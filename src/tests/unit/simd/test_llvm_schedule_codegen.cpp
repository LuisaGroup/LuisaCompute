#include "llvm_schedule_codegen.h"
#include "llvm_jit.h"
#include "simd_compiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>

#include "xir_to_schedule.h"

using namespace luisa::compute;
using namespace luisa::compute::simd;

namespace {

[[nodiscard]] bool check(bool condition, const char *expression,
                         const char *file, int line) noexcept {
    if (!condition) {
        std::cerr << file << ':' << line << ": check failed: "
                  << expression << '\n';
    }
    return condition;
}

#define CHECK(EXPR)                                                           \
    do {                                                                      \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) {     \
            return false;                                                     \
        }                                                                     \
    } while (false)

[[nodiscard]] SIMDPacketLaunchConfig launch_1d(
    uint32_t dispatch_size, uint32_t block_size) noexcept {
    SIMDPacketLaunchConfig config{};
    config.dispatch_size[0u] = dispatch_size;
    config.dispatch_size[1u] = 1u;
    config.dispatch_size[2u] = 1u;
    config.block_size[0u] = block_size;
    config.block_size[1u] = 1u;
    config.block_size[2u] = 1u;
    return config;
}

[[nodiscard]] std::string diagnostics_text(
    const schedule::XIRToScheduleResult &result) {
    std::string text;
    for (auto &&diagnostic : result.diagnostics) {
        text += schedule::to_string(diagnostic.code);
        text += ": ";
        text += diagnostic.message;
        text += '\n';
    }
    return text;
}

[[nodiscard]] std::optional<schedule::Function>
make_divergent_collective(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("divergent_collective");
    auto *entry = kernel->create_body_block();
    auto *left = kernel->create_basic_block();
    auto *right = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    left->set_name("left");
    right->set_name("right");
    merge->set_name("merge");

    auto *lane = module.create_warp_lane_id();
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    auto *two = module.create_constant(Type::of<uint32_t>(), &two_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS, {lane, two});
    builder.cond_br(condition, left, right);
    builder.set_insertion_point(left);
    auto *left_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD, {lane, two});
    auto *left_first = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE,
        {left_value});
    builder.br(merge);
    builder.set_insertion_point(right);
    auto *right_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_SUB, {lane, one});
    auto *right_first = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE,
        {right_value});
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{left_first, left}, {right_first, right}});
    selected->set_name("selected");
    auto *sum = builder.call(
        Type::of<uint32_t>(), xir::ThreadGroupOp::WARP_ACTIVE_SUM,
        {selected});
    sum->set_name("sum");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    auto function = std::move(*lowered.function);
    std::optional<schedule::ValueId> sum_id;
    for (auto &&value : function.values()) {
        if (value.name == "sum") { sum_id = value.id; }
    }
    if (!sum_id) { return std::nullopt; }
    for (auto &block : function.blocks()) {
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{sum_id};
        }
    }
    if (!schedule::verify(function).succeeded()) {
        return std::nullopt;
    }
    return function;
}

[[nodiscard]] std::optional<schedule::Function>
make_varying_loop(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("varying_loop");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *body = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("entry");
    header->set_name("header");
    body->set_name("body");
    exit->set_name("exit");
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.br(header);
    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    index->set_name("index");
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, one});
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(condition, body, exit);
    builder.set_insertion_point(body);
    auto *next = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one});
    builder.br(header);
    builder.set_insertion_point(exit);
    builder.return_void();
    index->add_incoming(zero, entry);
    index->add_incoming(next, body);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    auto function = std::move(*lowered.function);
    std::optional<schedule::ValueId> index_id;
    for (auto &&value : function.values()) {
        if (value.name == "index") { index_id = value.id; }
    }
    if (!index_id) { return std::nullopt; }
    for (auto &block : function.blocks()) {
        if (block.name == "exit") {
            block.terminator = schedule::ReturnTerminator{index_id};
        }
    }
    if (!schedule::verify(function).succeeded()) {
        return std::nullopt;
    }
    return function;
}

[[nodiscard]] std::optional<schedule::Function>
make_nested_divergence(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("nested_divergence");
    auto *entry = kernel->create_body_block();
    auto *nested = kernel->create_basic_block();
    auto *outer_right = kernel->create_basic_block();
    auto *inner_left = kernel->create_basic_block();
    auto *inner_right = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    nested->set_name("nested");
    outer_right->set_name("outer_right");
    inner_left->set_name("inner_left");
    inner_right->set_name("inner_right");
    merge->set_name("merge");
    auto *lane = module.create_warp_lane_id();
    uint32_t one_value = 1u;
    uint32_t three_value = 3u;
    auto *one = module.create_constant(
        Type::of<uint32_t>(), &one_value);
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *outer_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, three});
    builder.cond_br(outer_condition, nested, outer_right);
    builder.set_insertion_point(nested);
    auto *inner_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, one});
    builder.cond_br(inner_condition, inner_left, inner_right);
    builder.set_insertion_point(outer_right);
    builder.br(merge);
    builder.set_insertion_point(inner_left);
    builder.br(merge);
    builder.set_insertion_point(inner_right);
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, one});
    result->set_name("result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    auto function = std::move(*lowered.function);
    std::optional<schedule::ValueId> result_id;
    for (auto &&value : function.values()) {
        if (value.name == "result") { result_id = value.id; }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &block : function.blocks()) {
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!schedule::verify(function).succeeded()) {
        return std::nullopt;
    }
    return function;
}

[[nodiscard]] std::optional<schedule::Function>
make_large_cfg(uint32_t width, uint32_t block_count) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("large_cfg");
    std::vector<xir::BasicBlock *> blocks;
    blocks.reserve(block_count);
    blocks.emplace_back(kernel->create_body_block());
    for (auto i = uint32_t{1u}; i < block_count; i++) {
        blocks.emplace_back(kernel->create_basic_block());
    }
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    for (auto i = uint32_t{0u}; i + 1u < block_count; i++) {
        blocks[i]->set_name("chain_" + std::to_string(i));
        builder.set_insertion_point(blocks[i]);
        builder.br(blocks[i + 1u]);
    }
    blocks.back()->set_name("exit");
    builder.set_insertion_point(blocks.back());
    auto *result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, zero});
    result->set_name("result");
    builder.return_void();
    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    auto function = std::move(*lowered.function);
    std::optional<schedule::ValueId> result_id;
    for (auto &&value : function.values()) {
        if (value.name == "result") { result_id = value.id; }
    }
    if (!result_id) { return std::nullopt; }
    function.block(schedule::BlockId{block_count - 1u})->terminator =
        schedule::ReturnTerminator{result_id};
    return function;
}

template<size_t Width>
[[nodiscard]] bool run_codegen() {
    auto schedule_function = make_divergent_collective(Width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-schedule-codegen", *context);
    auto name = std::string{"schedule_divergent_w"} +
                std::to_string(Width);
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, Width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.argument_buffer_size == 0u);
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));

    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    auto lane_type = std::string{"<"} + std::to_string(Width) +
                     " x i32>";
    CHECK(ir.find(lane_type) != std::string::npos);
    CHECK(ir.find("llvm.vector.reduce.add.v" +
                  std::to_string(Width) + "i32") != std::string::npos);
    CHECK(ir.find("llvm.masked.store.v" +
                  std::to_string(Width) + "i32") != std::string::npos);
    CHECK(ir.find("llvm.x86.") == std::string::npos);
    CHECK(ir.find("llvm.aarch64.") == std::string::npos);
    CHECK(ir.find("llvm.arm.neon.") == std::string::npos);

    LLVMJIT jit;
    if (!jit.succeeded()) {
        std::cerr << jit.error() << '\n';
        return false;
    }
    if (!jit.add_module(std::move(module), std::move(context))) {
        std::cerr << jit.error() << '\n';
        return false;
    }
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    if (entry == nullptr) { std::cerr << jit.error() << '\n'; }
    CHECK(entry != nullptr);

    auto expected_sum = [](uint32_t active_lanes) noexcept {
        return active_lanes <= 2u ? active_lanes * 2u :
                                    active_lanes + 2u;
    };
    for (auto active_lanes :
         {static_cast<uint32_t>(Width),
          static_cast<uint32_t>(Width - 1u), uint32_t{0u}}) {
        std::array<uint32_t, Width> output{};
        output.fill(0xdeadbeefu);
        auto config = launch_1d(active_lanes, Width);
        entry(nullptr, output.data(), &config, active_lanes);
        auto sum = expected_sum(active_lanes);
        for (auto lane = uint32_t{0u}; lane < Width; lane++) {
            CHECK(output[lane] ==
                  (lane < active_lanes ? sum : 0xdeadbeefu));
        }
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool run_loop_codegen() {
    auto schedule_function = make_varying_loop(Width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-loop-codegen", *context);
    auto name = std::string{"schedule_loop_w"} +
                std::to_string(Width);
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, Width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    std::array<uint32_t, Width> output{};
    output.fill(0xdeadbeefu);
    auto config = launch_1d(Width, Width);
    entry(nullptr, output.data(), &config, Width);
    for (auto lane = uint32_t{0u}; lane < Width; lane++) {
        CHECK(output[lane] == lane + 1u);
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool run_control_fixture(
    std::optional<schedule::Function> schedule_function,
    std::string name, uint32_t increment) {
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(name, *context);
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, Width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    std::array<uint32_t, Width> output{};
    output.fill(0xdeadbeefu);
    auto config = launch_1d(Width, Width);
    entry(nullptr, output.data(), &config, Width);
    for (auto lane = uint32_t{0u}; lane < Width; lane++) {
        CHECK(output[lane] == lane + increment);
    }
    return true;
}

[[nodiscard]] bool run_nested_codegen() {
    return run_control_fixture<8u>(
        make_nested_divergence(8u), "schedule_nested_w8", 1u);
}

[[nodiscard]] bool run_large_cfg_codegen() {
    return run_control_fixture<4u>(
        make_large_cfg(4u, 96u), "schedule_large_cfg_w4", 0u);
}

[[nodiscard]] bool run_compiler_facade() {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("compiler_facade");
    auto *entry_block = kernel->create_body_block();
    auto *left = kernel->create_basic_block();
    auto *right = kernel->create_basic_block();
    auto *lane = module.create_warp_lane_id();
    uint32_t two_value = 2u;
    auto *two = module.create_constant(Type::of<uint32_t>(), &two_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS, {lane, two});
    builder.cond_br(condition, left, right);
    builder.set_insertion_point(left);
    builder.return_void();
    builder.set_insertion_point(right);
    builder.return_void();

    auto compiled = compile_simd_kernel(
        kernel, 8u, "simd_compiler_facade");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.argument_buffer_size == 0u);
    CHECK(!compiled.target_triple.empty());
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    auto config = launch_1d(8u, 8u);
    entry(nullptr, nullptr, &config, 8u);
    return true;
}

[[nodiscard]] bool run_buffer_vector_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto count = 11u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("buffer_vector_add");
    auto *buffer_type = Type::buffer(Type::of<luisa::uint4>());
    auto *lhs_argument = kernel->create_resource_argument(buffer_type);
    auto *rhs_argument = kernel->create_resource_argument(buffer_type);
    auto *output_argument = kernel->create_resource_argument(buffer_type);
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *lhs = builder.call(
        Type::of<luisa::uint4>(), xir::ResourceReadOp::BUFFER_READ,
        {lhs_argument, index});
    auto *rhs = builder.call(
        Type::of<luisa::uint4>(), xir::ResourceReadOp::BUFFER_READ,
        {rhs_argument, index});
    auto *sum = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::BINARY_ADD,
        {lhs, rhs});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output_argument, index, sum});
    builder.return_void();

    auto compiled = compile_simd_kernel(
        kernel, width, "simd_buffer_vector_add");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.argument_buffer_size ==
          3u * sizeof(SIMDHostBufferView));

    std::array<luisa::uint4, count> lhs_data{};
    std::array<luisa::uint4, count> rhs_data{};
    std::array<luisa::uint4, count> output_data{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        lhs_data[i] = luisa::make_uint4(i, i + 1u, i + 2u, i + 3u);
        rhs_data[i] = luisa::make_uint4(10u, 20u, 30u, 40u);
        output_data[i] = luisa::make_uint4(0xdeadbeefu);
    }
    alignas(16) std::array<SIMDHostBufferView, 3u> arguments{
        SIMDHostBufferView{lhs_data.data(), sizeof(lhs_data)},
        SIMDHostBufferView{rhs_data.data(), sizeof(rhs_data)},
        SIMDHostBufferView{output_data.data(), sizeof(output_data)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    config.thread_index = 0u;
    entry(arguments.data(), nullptr, &config, width);
    config.thread_index = width;
    entry(arguments.data(), nullptr, &config, width);
    for (auto i = uint32_t{0u}; i < count; i++) {
        CHECK(output_data[i].x == lhs_data[i].x + rhs_data[i].x);
        CHECK(output_data[i].y == lhs_data[i].y + rhs_data[i].y);
        CHECK(output_data[i].z == lhs_data[i].z + rhs_data[i].z);
        CHECK(output_data[i].w == lhs_data[i].w + rhs_data[i].w);
    }
    return true;
}

[[nodiscard]] bool run_ast_buffer_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferUInt lhs, BufferUInt rhs,
                         BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        output.write(index, lhs.read(index) + rhs.read(index));
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width, "simd_ast_buffer_add");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    std::array<uint32_t, count> lhs{};
    std::array<uint32_t, count> rhs{};
    std::array<uint32_t, count> output{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        lhs[i] = i * 3u;
        rhs[i] = 100u - i;
        output[i] = 0xdeadbeefu;
    }
    alignas(16) std::array<SIMDHostBufferView, 3u> arguments{
        SIMDHostBufferView{lhs.data(), sizeof(lhs)},
        SIMDHostBufferView{rhs.data(), sizeof(rhs)},
        SIMDHostBufferView{output.data(), sizeof(output)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(arguments.data(), nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        CHECK(output[i] == lhs[i] + rhs[i]);
    }
    return true;
}

[[nodiscard]] bool run_ast_select_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        auto even = (index & 1u) == 0u;
        output.write(index, ite(even, 17u, 29u));
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width, "simd_ast_select");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    std::array<uint32_t, count> output{};
    output.fill(0xdeadbeefu);
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(&argument, nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        CHECK(output[i] == (i % 2u == 0u ? 17u : 29u));
    }
    return true;
}

}// namespace

int main() {
    struct Test {
        std::string_view name;
        bool (*run)();
    };
    constexpr Test tests[]{
        {"Schedule IR vector warp1", &run_codegen<1u>},
        {"Schedule IR vector warp4", &run_codegen<4u>},
        {"Schedule IR vector warp8", &run_codegen<8u>},
        {"Schedule IR vector warp16", &run_codegen<16u>},
        {"Schedule IR loop warp4", &run_loop_codegen<4u>},
        {"Schedule IR loop warp8", &run_loop_codegen<8u>},
        {"Schedule IR nested convergence", &run_nested_codegen},
        {"Schedule IR 96-block CFG", &run_large_cfg_codegen},
        {"XIR compiler facade", &run_compiler_facade},
        {"XIR buffer vector gather/scatter", &run_buffer_vector_codegen},
        {"AST buffer dispatch", &run_ast_buffer_codegen},
        {"AST select operand order", &run_ast_select_codegen},
    };
    auto failures = 0u;
    for (auto test : tests) {
        if (test.run()) {
            std::cout << "[pass] " << test.name << '\n';
        } else {
            std::cerr << "[fail] " << test.name << '\n';
            ++failures;
        }
    }
    return failures == 0u ? 0 : 1;
}
