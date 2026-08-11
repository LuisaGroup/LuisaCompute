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
    using Entry = void(const void *, uint32_t *, uint32_t);
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
        entry(nullptr, output.data(), active_lanes);
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
    using Entry = void(const void *, uint32_t *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    std::array<uint32_t, Width> output{};
    output.fill(0xdeadbeefu);
    entry(nullptr, output.data(), Width);
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
    using Entry = void(const void *, uint32_t *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    std::array<uint32_t, Width> output{};
    output.fill(0xdeadbeefu);
    entry(nullptr, output.data(), Width);
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
    using Entry = void(const void *, void *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    entry(nullptr, nullptr, 8u);
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
