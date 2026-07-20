// Validates GPU-authored direct and batched indirect dispatch commands.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/dispatch_indirect.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_indirect(Device &device) {
    log_level_verbose();

    Stream stream = device.create_stream();
    constexpr auto kernel_block_size = make_uint3(64, 1, 1);
    constexpr auto dispatch_count = 16u;
    Kernel1D clear_kernel = [=](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.set_dispatch_count(dispatch_count);
    };
    Kernel1D emplace_kernel = [=](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.set_kernel(dispatch_id().x, kernel_block_size, make_uint3(dispatch_id().x, 1u, 1u), dispatch_id().x);
    };
    Kernel1D set_kernel = [=](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
        dispatch_buffer.set_kernel(dispatch_id().x, kernel_block_size, make_uint3(dispatch_id().x, 1u, 1u), dispatch_id().x);
    };
    Kernel1D dispatch_kernel = [=](BufferVar<uint> buffer) {
        set_block_size(kernel_block_size.x, kernel_block_size.y, kernel_block_size.z);
        buffer.atomic(kernel_id()).fetch_add(dispatch_size().x);
    };
    auto clear_shader = device.compile(clear_kernel);
    auto emplace_shader = device.compile(emplace_kernel);
    auto set_shader = device.compile(set_kernel);
    auto dispatch_shader = device.compile(dispatch_kernel);

    IndirectDispatchBuffer dispatch_buffer = device.create_indirect_dispatch_buffer(dispatch_count);
    Buffer<uint> buffer = device.create_buffer<uint>(dispatch_count);
    std::array<uint, dispatch_count> buffer_data{};
    CommandList cmdlist{};
    cmdlist << buffer.copy_from(luisa::span{buffer_data});
    // Single dispatch
    cmdlist << clear_shader(dispatch_buffer).dispatch(1)
            << set_shader(dispatch_buffer).dispatch(dispatch_count);
    for (auto i = 0; i < 16; ++i) {
        cmdlist << dispatch_shader(buffer).dispatch(dispatch_buffer, i, 1);
    }
    // Dispatch all
    cmdlist << emplace_shader(dispatch_buffer).dispatch(dispatch_count)
            << dispatch_shader(buffer).dispatch(dispatch_buffer);
    //  dispatch
    cmdlist << buffer.copy_to(luisa::span{buffer_data});
    stream << cmdlist.commit() << synchronize();
    luisa::string result;
    for (auto &i : buffer_data) {
        result += std::to_string(i) + " ";
    }
    LUISA_INFO("Result should be: 0 2 8 18 32 50 72 98 128 162 200 242 288 338 392 450");
    LUISA_INFO("Result: {}", result);
    constexpr std::array expected{0u, 2u, 8u, 18u, 32u, 50u, 72u, 98u,
                                  128u, 162u, 200u, 242u, 288u, 338u, 392u, 450u};
    for (auto i = 0u; i < dispatch_count; ++i) {
        expect(buffer_data[i] == expected[i]) << "indirect dispatch result mismatch at index " << i;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_indirect(device);
}
