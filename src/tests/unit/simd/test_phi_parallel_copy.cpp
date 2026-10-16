#include "ut/ut.hpp"
#include "simd_compiler.h"
#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <array>

using namespace luisa::compute;
using namespace luisa::compute::simd;
using namespace boost::ut;

namespace {

void run(uint32_t width, bool varying_trip_count) {
    xir::Module module;
    auto kernel = module.create_kernel();
    kernel->set_block_size(luisa::make_uint3(64u, 1u, 1u));
    auto buffer = kernel->create_resource_argument(Type::buffer(Type::of<uint32_t>()));
    auto preheader = kernel->create_body_block();
    auto header = kernel->create_basic_block();
    auto body = kernel->create_basic_block();
    auto exit = kernel->create_basic_block();
    xir::XIRBuilder builder;
    builder.set_insertion_point(preheader);
    auto constant = [&](uint32_t value) { return module.create_constant(Type::of<uint32_t>(), &value); };
    auto zero = constant(0u), one = constant(1u), three = constant(3u);
    using A = xir::ArithmeticOp;
    auto add = [&](xir::Value *a, xir::Value *b) { return builder.call(Type::of<uint32_t>(), A::BINARY_ADD, {a, b}); };
    auto lane = builder.call(Type::of<uint32_t>(), A::EXTRACT, {module.create_dispatch_id(), zero});
    auto a0 = add(lane, one), b0 = add(lane, constant(9u));
    xir::Value *limit = three;
    if (varying_trip_count) { limit = add(builder.call(Type::of<uint32_t>(), A::BINARY_MOD, {lane, three}), one); }
    builder.br(header);
    builder.set_insertion_point(header);
    auto index = builder.phi(Type::of<uint32_t>(), {{zero, preheader}});
    auto a = builder.phi(Type::of<uint32_t>(), {{a0, preheader}});
    auto b = builder.phi(Type::of<uint32_t>(), {{b0, preheader}});
    builder.cond_br(builder.call(Type::of<bool>(), A::BINARY_LESS, {index, limit}), body, exit);
    builder.set_insertion_point(body);
    // a = b; b = old_a. No arithmetic temporary obscures the PHI cycle.
    a->add_incoming(b, body);
    b->add_incoming(a, body);
    index->add_incoming(add(index, one), body);
    builder.br(header);
    builder.set_insertion_point(exit);
    auto address = builder.call(Type::of<uint32_t>(), A::BINARY_MUL, {lane, constant(2u)});
    builder.call(xir::ResourceWriteOp::BUFFER_WRITE, {buffer, address, a});
    builder.call(xir::ResourceWriteOp::BUFFER_WRITE, {buffer, add(address, one), b});
    builder.return_void();
    auto compiled = compile_simd_kernel(kernel, width, "parallel_phi_copy", false, true, true, false, 1u, false, false, true);
    expect(compiled.succeeded());
    if (!compiled.succeeded()) { return; }
    expect(!compiled.llvm_ir.empty());
    expect(compiled.assembly.empty());
    using Entry = void(const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    expect(entry != nullptr);
    if (entry == nullptr) { return; }
    for (auto active = 0u; active <= width; active++) {
        std::array<uint32_t, 32u> output;
        output.fill(0xdeadbeefu);
        SIMDHostBufferView view{output.data(), output.size() * sizeof(uint32_t)};
        SIMDPacketLaunchConfig launch{.dispatch_size = {active, 1u, 1u}, .block_size = {64u, 1u, 1u}};
        entry(&view, nullptr, &launch, active);
        auto correct = true;
        for (auto lane_id = 0u; lane_id < width; lane_id++) {
            auto iterations = varying_trip_count ? lane_id % 3u + 1u : 3u;
            auto a_expected = lane_id + (iterations % 2u == 0u ? 1u : 9u);
            auto b_expected = lane_id + (iterations % 2u == 0u ? 9u : 1u);
            correct &= output[lane_id * 2u] == (lane_id < active ? a_expected : 0xdeadbeefu);
            correct &= output[lane_id * 2u + 1u] == (lane_id < active ? b_expected : 0xdeadbeefu);
        }
        expect(correct) << "width=" << width << " active=" << active << " varying=" << varying_trip_count;
    }
}

}// namespace

int main() {
    "simd_simultaneous_phi_cycles"_test = [] {
        for (auto width : {1u, 2u, 4u, 8u, 16u}) {
            run(width, false);
            run(width, true);
        }
    };
}
