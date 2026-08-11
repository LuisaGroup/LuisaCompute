#include "llvm_schedule_codegen.h"
#include "llvm_jit.h"
#include "simd_compiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

#define CHECK(EXPR)                                                       \
    do {                                                                  \
        if (!check(static_cast<bool>(EXPR), #EXPR, __FILE__, __LINE__)) { \
            return false;                                                 \
        }                                                                 \
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

[[nodiscard]] size_t count_occurrences(
    std::string_view text, std::string_view needle) noexcept {
    auto count = size_t{0u};
    for (auto position = text.find(needle);
         position != std::string_view::npos;
         position = text.find(needle, position + needle.size())) {
        count++;
    }
    return count;
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
make_cold_state_pressure(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("cold_state_pressure");
    auto *entry = kernel->create_body_block();
    auto *left = kernel->create_basic_block();
    auto *right = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    left->set_name("left");
    right->set_name("right");
    merge->set_name("merge");

    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    std::array<xir::Value *, 8u> values{};
    for (auto i = size_t{0u}; i < values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 1u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
        values[i]->set_name("cold_" + std::to_string(i));
    }
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *take_left = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {parity, zero});
    builder.cond_br(take_left, left, right);

    builder.set_insertion_point(left);
    auto *left01 = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {values[0u], values[1u]});
    auto *left23 = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {values[2u], values[3u]});
    auto *left_sum = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {left01, left23});
    builder.br(merge);

    builder.set_insertion_point(right);
    auto *right45 = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {values[4u], values[5u]});
    auto *right67 = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {values[6u], values[7u]});
    auto *right_sum = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {right45, right67});
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{left_sum, left}, {right_sum, right}});
    selected->set_name("cold_state_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "cold_state_result") {
            result_id = value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
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
make_varying_loop_collective(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("varying_loop_collective");
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
    auto *sum = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_ACTIVE_SUM, {one});
    sum->set_name("loop_exit_sum");
    builder.return_void();
    index->add_incoming(zero, entry);
    index->add_incoming(next, body);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> sum_id;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "loop_exit_sum") { sum_id = value.id; }
    }
    // A post-dominator definition that treats the natural back-edge as a
    // virtual exit omits this gate. The first lane leaving the loop would then
    // execute the collective alone and observe one participant.
    auto header_has_exit_gate = false;
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "header") {
            if (auto *split = std::get_if<schedule::SplitTerminator>(
                    &block.terminator)) {
                header_has_exit_gate = split->convergence.has_value();
            }
        } else if (block.name == "exit") {
            block.terminator = schedule::ReturnTerminator{sum_id};
        }
    }
    if (!sum_id || !header_has_exit_gate ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_multiple_exit_loop_collective(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *dispatch = kernel->create_basic_block();
    auto *latch = kernel->create_basic_block();
    auto *side_exit = kernel->create_basic_block();
    auto *normal_exit = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    header->set_name("header");
    dispatch->set_name("dispatch");
    side_exit->set_name("side_exit");
    normal_exit->set_name("normal_exit");
    merge->set_name("merge");
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.br(header);
    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, one});
    auto *running = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(running, dispatch, normal_exit);
    builder.set_insertion_point(dispatch);
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *take_latch = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {parity, zero});
    builder.cond_br(take_latch, latch, side_exit);
    builder.set_insertion_point(latch);
    auto *next = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one});
    builder.br(header);
    builder.set_insertion_point(side_exit);
    builder.br(merge);
    builder.set_insertion_point(normal_exit);
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *sum = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_ACTIVE_SUM, {one});
    sum->set_name("multiple_exit_sum");
    builder.return_void();
    index->add_incoming(zero, entry);
    index->add_incoming(next, latch);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> sum_id;
    auto gated_splits = 0u;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "multiple_exit_sum") { sum_id = value.id; }
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "header" || block.name == "dispatch") {
            auto *split = std::get_if<schedule::SplitTerminator>(
                &block.terminator);
            if (split != nullptr && split->convergence) {
                ++gated_splits;
            }
        } else if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{sum_id};
        }
    }
    if (!sum_id || gated_splits != 2u ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
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

[[nodiscard]] std::optional<schedule::Function>
make_varying_switch(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("varying_switch");
    auto *entry = kernel->create_body_block();
    auto *case_zero = kernel->create_basic_block();
    auto *case_two = kernel->create_basic_block();
    auto *case_five = kernel->create_basic_block();
    auto *default_case = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    case_zero->set_name("case_zero");
    case_two->set_name("case_two");
    case_five->set_name("case_five");
    default_case->set_name("default_case");
    merge->set_name("merge");
    auto *lane = module.create_warp_lane_id();
    uint32_t ten_value = 10u;
    uint32_t twenty_value = 20u;
    uint32_t thirty_value = 30u;
    uint32_t forty_value = 40u;
    auto *ten = module.create_constant(
        Type::of<uint32_t>(), &ten_value);
    auto *twenty = module.create_constant(
        Type::of<uint32_t>(), &twenty_value);
    auto *thirty = module.create_constant(
        Type::of<uint32_t>(), &thirty_value);
    auto *forty = module.create_constant(
        Type::of<uint32_t>(), &forty_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *branch = builder.indexed_branch(lane);
    branch->set_default_block(default_case);
    branch->add_case(0u, case_zero);
    branch->add_case(2u, case_two);
    branch->add_case(5u, case_five);
    builder.set_insertion_point(case_zero);
    auto *zero_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, ten});
    builder.br(merge);
    builder.set_insertion_point(case_two);
    auto *two_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, twenty});
    builder.br(merge);
    builder.set_insertion_point(case_five);
    auto *five_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, thirty});
    builder.br(merge);
    builder.set_insertion_point(default_case);
    auto *default_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, forty});
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{zero_result, case_zero},
         {two_result, case_two},
         {five_result, case_five},
         {default_result, default_case}});
    selected->set_name("varying_switch_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    auto saw_convergent_switch = false;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "varying_switch_result") {
            result_id = value.id;
            if (value.value_class != schedule::ValueClass::varying) {
                return std::nullopt;
            }
        }
    }
    for (auto &block : lowered.function->blocks()) {
        if (auto *terminator = std::get_if<schedule::SwitchTerminator>(
                &block.terminator)) {
            saw_convergent_switch = terminator->convergence.has_value();
        }
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!result_id || !saw_convergent_switch ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_switch_loop_with_exits(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("switch_loop_with_exits");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *dispatch = kernel->create_basic_block();
    auto *continue_zero = kernel->create_basic_block();
    auto *early_return = kernel->create_basic_block();
    auto *break_return = kernel->create_basic_block();
    auto *continue_default = kernel->create_basic_block();
    auto *latch = kernel->create_basic_block();
    auto *normal_return = kernel->create_basic_block();
    entry->set_name("entry");
    header->set_name("header");
    dispatch->set_name("dispatch");
    continue_zero->set_name("continue_zero");
    early_return->set_name("early_return");
    break_return->set_name("break_return");
    continue_default->set_name("continue_default");
    latch->set_name("latch");
    normal_return->set_name("normal_return");
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t three_value = 3u;
    uint32_t four_value = 4u;
    uint32_t hundred_value = 100u;
    uint32_t two_hundred_value = 200u;
    uint32_t three_hundred_value = 300u;
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    auto *four = module.create_constant(
        Type::of<uint32_t>(), &four_value);
    auto *hundred = module.create_constant(
        Type::of<uint32_t>(), &hundred_value);
    auto *two_hundred = module.create_constant(
        Type::of<uint32_t>(), &two_hundred_value);
    auto *three_hundred = module.create_constant(
        Type::of<uint32_t>(), &three_hundred_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.br(header);
    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    index->set_name("switch_loop_index");
    auto *keep_running = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, four});
    builder.cond_br(keep_running, dispatch, normal_return);
    builder.set_insertion_point(dispatch);
    auto *lane_and_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, index});
    auto *selector = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane_and_index, three});
    auto *switch_inst = builder.indexed_branch(selector);
    switch_inst->set_default_block(continue_default);
    switch_inst->add_case(0u, continue_zero);
    switch_inst->add_case(1u, early_return);
    switch_inst->add_case(2u, break_return);
    builder.set_insertion_point(continue_zero);
    builder.br(latch);
    builder.set_insertion_point(continue_default);
    builder.br(latch);
    builder.set_insertion_point(latch);
    auto *next = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one});
    builder.br(header);
    builder.set_insertion_point(early_return);
    auto *early_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, hundred});
    early_value->set_name("early_value");
    builder.return_void();
    builder.set_insertion_point(break_return);
    auto *break_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, two_hundred});
    break_value->set_name("break_value");
    builder.return_void();
    builder.set_insertion_point(normal_return);
    auto *normal_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, three_hundred});
    normal_value->set_name("normal_value");
    builder.return_void();
    index->add_incoming(zero, entry);
    index->add_incoming(next, latch);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> early_id;
    std::optional<schedule::ValueId> break_id;
    std::optional<schedule::ValueId> normal_id;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "early_value") { early_id = value.id; }
        if (value.name == "break_value") { break_id = value.id; }
        if (value.name == "normal_value") { normal_id = value.id; }
    }
    if (!early_id || !break_id || !normal_id ||
        lowered.function->loops().empty()) {
        return std::nullopt;
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "early_return") {
            block.terminator = schedule::ReturnTerminator{early_id};
        } else if (block.name == "break_return") {
            block.terminator = schedule::ReturnTerminator{break_id};
        } else if (block.name == "normal_return") {
            block.terminator = schedule::ReturnTerminator{normal_id};
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_multiple_backedge_loop(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("multiple_backedge_loop");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *body = kernel->create_basic_block();
    auto *increment_one = kernel->create_basic_block();
    auto *increment_two = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("entry");
    header->set_name("header");
    body->set_name("body");
    increment_one->set_name("increment_one");
    increment_two->set_name("increment_two");
    exit->set_name("exit");
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.br(header);
    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    index->set_name("multiple_backedge_index");
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, one});
    auto *keep_running = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(keep_running, body, exit);
    builder.set_insertion_point(body);
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *take_one = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {parity, zero});
    builder.cond_br(take_one, increment_one, increment_two);
    builder.set_insertion_point(increment_one);
    auto *next_one = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one});
    builder.br(header);
    builder.set_insertion_point(increment_two);
    auto *next_two = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, two});
    builder.br(header);
    builder.set_insertion_point(exit);
    builder.return_void();
    index->add_incoming(zero, entry);
    index->add_incoming(next_one, increment_one);
    index->add_incoming(next_two, increment_two);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    auto loop_back_count = size_t{0u};
    for (auto &&value : lowered.function->values()) {
        if (value.name == "multiple_backedge_index") {
            result_id = value.id;
        }
    }
    for (auto &block : lowered.function->blocks()) {
        if (auto *branch = std::get_if<schedule::BranchTerminator>(
                &block.terminator);
            branch != nullptr && branch->edge.loop_back) {
            ++loop_back_count;
        }
        if (block.name == "exit") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!result_id || lowered.function->loops().size() != 1u ||
        loop_back_count != 2u ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_non_dominating_convergence(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("non_dominating_convergence");
    auto *entry = kernel->create_body_block();
    auto *split = kernel->create_basic_block();
    auto *bypass = kernel->create_basic_block();
    auto *shared = kernel->create_basic_block();
    auto *other = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    split->set_name("split");
    bypass->set_name("bypass");
    shared->set_name("shared");
    other->set_name("other");
    merge->set_name("merge");
    auto *lane = module.create_warp_lane_id();
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t three_value = 3u;
    uint32_t six_value = 6u;
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    auto *six = module.create_constant(
        Type::of<uint32_t>(), &six_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *enter_split = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, six});
    builder.cond_br(enter_split, split, bypass);
    builder.set_insertion_point(split);
    auto *take_shared = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, three});
    builder.cond_br(take_shared, shared, other);
    builder.set_insertion_point(bypass);
    builder.br(shared);
    builder.set_insertion_point(shared);
    builder.br(merge);
    builder.set_insertion_point(other);
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *sum = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_ACTIVE_SUM, {one});
    sum->set_name("non_dominating_sum");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> sum_id;
    std::optional<schedule::ConvergenceId> inner_convergence;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "non_dominating_sum") { sum_id = value.id; }
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "split") {
            auto *terminator = std::get_if<schedule::SplitTerminator>(
                &block.terminator);
            if (terminator != nullptr) {
                inner_convergence = terminator->convergence;
            }
        }
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{sum_id};
        }
    }
    if (!sum_id || !inner_convergence) { return std::nullopt; }

    // `shared` is intentionally not dominated by `split`, so the old static
    // dominator-subtree annotation does not mention the still-live inner
    // token on shared -> merge. Dynamic target matching must nevertheless
    // rendezvous that cohort before executing the collective at merge.
    auto static_join_is_missing = false;
    for (auto &&block : lowered.function->blocks()) {
        if (block.name != "shared") { continue; }
        auto *branch = std::get_if<schedule::BranchTerminator>(
            &block.terminator);
        if (branch != nullptr) {
            static_join_is_missing = std::find(
                                         branch->edge.joins.begin(),
                                         branch->edge.joins.end(),
                                         *inner_convergence) == branch->edge.joins.end();
        }
    }
    if (!static_join_is_missing ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_return_convergence_cascade(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("return_convergence_cascade");
    auto *entry = kernel->create_body_block();
    auto *nested = kernel->create_basic_block();
    auto *inner_live = kernel->create_basic_block();
    auto *early_return = kernel->create_basic_block();
    auto *outer_right = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    nested->set_name("nested");
    inner_live->set_name("inner_live");
    early_return->set_name("early_return");
    outer_right->set_name("outer_right");
    merge->set_name("merge");

    auto *lane = module.create_warp_lane_id();
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    uint32_t four_value = 4u;
    uint32_t hundred_value = 100u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *four = module.create_constant(
        Type::of<uint32_t>(), &four_value);
    auto *hundred = module.create_constant(
        Type::of<uint32_t>(), &hundred_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *outer_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, four});
    builder.cond_br(outer_condition, nested, outer_right);
    builder.set_insertion_point(nested);
    auto *inner_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, two});
    builder.cond_br(inner_condition, inner_live, early_return);
    builder.set_insertion_point(inner_live);
    builder.br(merge);
    builder.set_insertion_point(early_return);
    auto *early_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, hundred});
    early_value->set_name("early_value");
    builder.br(merge);
    builder.set_insertion_point(outer_right);
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *sum = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_ACTIVE_SUM, {one});
    sum->set_name("return_cascade_sum");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> early_id;
    std::optional<schedule::ValueId> sum_id;
    std::optional<schedule::ConvergenceId> outer_convergence;
    std::optional<schedule::ConvergenceId> inner_convergence;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "early_value") { early_id = value.id; }
        if (value.name == "return_cascade_sum") { sum_id = value.id; }
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "entry") {
            if (auto *split = std::get_if<schedule::SplitTerminator>(
                    &block.terminator)) {
                outer_convergence = split->convergence;
            }
        } else if (block.name == "nested") {
            if (auto *split = std::get_if<schedule::SplitTerminator>(
                    &block.terminator)) {
                inner_convergence = split->convergence;
            }
        } else if (block.name == "early_return") {
            block.terminator = schedule::ReturnTerminator{early_id};
        } else if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{sum_id};
        }
    }
    if (!early_id || !sum_id || !outer_convergence ||
        !inner_convergence) {
        return std::nullopt;
    }
    auto *outer = lowered.function->convergence(*outer_convergence);
    auto *inner = lowered.function->convergence(*inner_convergence);
    if (outer == nullptr || inner == nullptr ||
        outer->target != inner->target ||
        inner->parent != outer_convergence ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
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
    if constexpr (Width == 1u) {
        CHECK(ir.find("scheduler.loop") == std::string::npos);
        CHECK(ir.find("lane.convergence.token") == std::string::npos);
        CHECK(ir.find("frame.expected") == std::string::npos);
    } else {
        CHECK(ir.find("scheduler.loop") != std::string::npos);
        CHECK(ir.find("lane.pc") == std::string::npos);
        CHECK(ir.find("loop.epoch") == std::string::npos);
        CHECK(ir.find("lane.convergence.token") == std::string::npos);
        CHECK(ir.find("current.token") != std::string::npos);
        CHECK(ir.find("ready.mask") != std::string::npos);
        CHECK(ir.find("ready.token") != std::string::npos);
        CHECK(ir.find("frame.active = alloca i" +
                      std::to_string(Width)) != std::string::npos);
        CHECK(ir.find("frame.active = alloca <") == std::string::npos);
    }

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

[[nodiscard]] bool run_static_block_size_codegen() {
    static constexpr auto width = 8u;
    auto schedule_function = make_divergent_collective(width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-static-block-size", *context);
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width,
        "schedule_static_block_size", false, {32u, 2u, 1u});
    CHECK(codegen.succeeded());
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));

    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("thread.id.x") != std::string::npos);
    CHECK(ir.find("thread.id.yz") != std::string::npos);
    CHECK(ir.find("thread.id.y") != std::string::npos);
    CHECK(ir.find("thread.id.z") != std::string::npos);
    CHECK(ir.find(" urem ") == std::string::npos);
    CHECK(ir.find(" udiv ") == std::string::npos);

    auto invalid_context = std::make_unique<::llvm::LLVMContext>();
    auto invalid_module = std::make_unique<::llvm::Module>(
        "simd-invalid-static-block-size", *invalid_context);
    auto invalid = lower_schedule_to_llvm(
        *invalid_module, *schedule_function, width,
        "schedule_invalid_static_block_size", false,
        {48u, 2u, 1u});
    CHECK(!invalid.succeeded());
    CHECK(invalid.error.find("powers of two") != std::string::npos);
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

[[nodiscard]] bool run_loop_collective_codegen(
    std::optional<schedule::Function> schedule_function,
    std::string name) {
    static constexpr auto width = 8u;
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(name, *context);
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width, name);
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
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    for (auto active_lanes : {uint32_t{5u}, uint32_t{8u}}) {
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        auto config = launch_1d(active_lanes, width);
        function(nullptr, output.data(), &config, active_lanes);
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            CHECK(output[lane] ==
                  (lane < active_lanes ? active_lanes : 0xdeadbeefu));
        }
    }
    return true;
}

[[nodiscard]] bool run_varying_loop_collective_codegen() {
    return run_loop_collective_codegen(
        make_varying_loop_collective(8u),
        "simd_varying_loop_collective");
}

[[nodiscard]] bool run_multiple_exit_loop_collective_codegen() {
    return run_loop_collective_codegen(
        make_multiple_exit_loop_collective(8u),
        "simd_multiple_exit_loop_collective");
}

template<size_t Width>
[[nodiscard]] bool run_control_fixture(
    std::optional<schedule::Function> schedule_function,
    std::string name, uint32_t increment) {
    CHECK(schedule_function.has_value());
    std::vector<uint8_t> convergence_targets(
        schedule_function->blocks().size(), uint8_t{0u});
    for (auto &&point : schedule_function->convergence_points()) {
        convergence_targets[point.target.value] = 1u;
    }
    auto convergence_target_count = static_cast<size_t>(std::count(
        convergence_targets.begin(), convergence_targets.end(),
        uint8_t{1u}));
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(name, *context);
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, Width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("lane.convergence.token") == std::string::npos);
    CHECK(ir.find("ready.token") != std::string::npos);
    CHECK(count_occurrences(ir, "\nconvergence.cascade") ==
          2u * convergence_target_count);
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

[[nodiscard]] bool run_state_residency_codegen() {
    static constexpr auto width = 8u;
    auto schedule_function = make_cold_state_pressure(width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "schedule-state-residency", *context);
    auto name = std::string{"schedule_state_residency_w8"};
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("load volatile") != std::string::npos);
    CHECK(ir.find("store volatile") != std::string::npos);
    CHECK(ir.find("cold_0.spill") != std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(entry != nullptr);
    for (auto active_lanes : {uint32_t{5u}, uint32_t{8u}}) {
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        auto config = launch_1d(active_lanes, width);
        entry(nullptr, output.data(), &config, active_lanes);
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            auto expected = lane % 2u == 0u ?
                                4u * lane + 10u :
                                4u * lane + 26u;
            CHECK(output[lane] ==
                  (lane < active_lanes ? expected : 0xdeadbeefu));
        }
    }
    return true;
}

[[nodiscard]] bool run_uniform_value_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto left_addend = uint32_t{0x13579bdfu};
    static constexpr auto right_addend = uint32_t{0x2468ace0u};
    xir::Module xir_module;
    auto *kernel = xir_module.create_kernel();
    kernel->set_name("uniform_values");
    auto *condition = kernel->create_value_argument(Type::of<bool>());
    auto *bias = kernel->create_value_argument(Type::of<uint32_t>());
    auto *entry = kernel->create_body_block();
    auto *left = kernel->create_basic_block();
    auto *right = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    left->set_name("left");
    right->set_name("right");
    merge->set_name("merge");
    auto *left_constant = xir_module.create_constant(
        Type::of<uint32_t>(), &left_addend);
    auto *right_constant = xir_module.create_constant(
        Type::of<uint32_t>(), &right_addend);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    builder.cond_br(condition, left, right);
    builder.set_insertion_point(left);
    auto *left_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {bias, left_constant});
    left_value->set_name("left_uniform_add");
    builder.br(merge);
    builder.set_insertion_point(right);
    auto *right_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {bias, right_constant});
    right_value->set_name("right_uniform_add");
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{left_value, left}, {right_value, right}});
    selected->set_name("selected_uniform_phi");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    std::optional<schedule::ValueId> selected_id;
    for (auto &&value : lowered.function->values()) {
        if (value.origin == schedule::ValueOrigin::parameter) {
            CHECK(value.value_class == schedule::ValueClass::warp_uniform);
        }
        if (value.name == "left_uniform_add" ||
            value.name == "right_uniform_add") {
            CHECK(value.value_class == schedule::ValueClass::warp_uniform);
        }
        if (value.name == "selected_uniform_phi") {
            CHECK(value.value_class == schedule::ValueClass::warp_uniform);
            selected_id = value.id;
        }
    }
    CHECK(selected_id.has_value());
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{selected_id};
        }
    }

    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-uniform-values", *context);
    auto name = std::string{"simd_uniform_values"};
    auto codegen = lower_schedule_to_llvm(
        *module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.argument_buffer_size == 32u);
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));

    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("selected_uniform_phi.slot = alloca i32") !=
          std::string::npos);
    auto check_scalar_add = [&](uint32_t addend) noexcept {
        auto literal = std::to_string(addend);
        auto position = ir.find(literal);
        while (position != std::string::npos) {
            auto line_begin = ir.rfind('\n', position);
            line_begin = line_begin == std::string::npos ?
                             0u :
                             line_begin + 1u;
            auto line_end = ir.find('\n', position);
            auto line = std::string_view{ir}.substr(
                line_begin, line_end - line_begin);
            if (line.find("add i32") != std::string_view::npos) {
                return line.find("<8 x i32>") == std::string_view::npos;
            }
            position = ir.find(literal, position + literal.size());
        }
        return false;
    };
    CHECK(check_scalar_add(left_addend));
    CHECK(check_scalar_add(right_addend));

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    alignas(16) std::array<std::byte, 32u> arguments{};
    auto config = launch_1d(width, 32u);
    for (auto take_left : {false, true}) {
        arguments[0u] = static_cast<std::byte>(take_left);
        auto bias_value = uint32_t{17u};
        std::memcpy(arguments.data() + 16u,
                    &bias_value, sizeof(bias_value));
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        function(arguments.data(), output.data(), &config, width);
        auto expected = bias_value +
                        (take_left ? left_addend : right_addend);
        for (auto value : output) { CHECK(value == expected); }
    }
    return true;
}

[[nodiscard]] bool run_uniform_switch_codegen() {
    static constexpr auto width = 8u;
    xir::Module xir_module;
    auto *kernel = xir_module.create_kernel();
    kernel->set_name("uniform_switch");
    auto *selector = kernel->create_value_argument(Type::of<uint32_t>());
    auto *bias = kernel->create_value_argument(Type::of<uint32_t>());
    auto *entry = kernel->create_body_block();
    auto *case_zero = kernel->create_basic_block();
    auto *case_two = kernel->create_basic_block();
    auto *default_case = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    case_zero->set_name("case_zero");
    case_two->set_name("case_two");
    default_case->set_name("default_case");
    merge->set_name("merge");
    uint32_t eleven_value = 11u;
    uint32_t twenty_two_value = 22u;
    uint32_t thirty_three_value = 33u;
    auto *eleven = xir_module.create_constant(
        Type::of<uint32_t>(), &eleven_value);
    auto *twenty_two = xir_module.create_constant(
        Type::of<uint32_t>(), &twenty_two_value);
    auto *thirty_three = xir_module.create_constant(
        Type::of<uint32_t>(), &thirty_three_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *switch_inst = builder.indexed_branch(selector);
    switch_inst->set_default_block(default_case);
    switch_inst->add_case(0u, case_zero);
    switch_inst->add_case(2u, case_two);
    builder.set_insertion_point(case_zero);
    auto *zero_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {bias, eleven});
    builder.br(merge);
    builder.set_insertion_point(case_two);
    auto *two_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {bias, twenty_two});
    builder.br(merge);
    builder.set_insertion_point(default_case);
    auto *default_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {bias, thirty_three});
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{zero_result, case_zero},
         {two_result, case_two},
         {default_result, default_case}});
    selected->set_name("uniform_switch_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    std::optional<schedule::ValueId> result_id;
    auto saw_uniform_switch = false;
    for (auto &&value : lowered.function->values()) {
        if (value.origin == schedule::ValueOrigin::parameter) {
            CHECK(value.value_class == schedule::ValueClass::warp_uniform);
        }
        if (value.name == "uniform_switch_result") {
            CHECK(value.value_class == schedule::ValueClass::warp_uniform);
            result_id = value.id;
        }
    }
    for (auto &block : lowered.function->blocks()) {
        if (std::holds_alternative<schedule::SwitchTerminator>(
                block.terminator)) {
            saw_uniform_switch =
                block.strategy == schedule::RegionStrategy::uniform_control;
        }
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    CHECK(result_id.has_value());
    CHECK(saw_uniform_switch);

    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-uniform-switch", *context);
    auto name = std::string{"simd_uniform_switch"};
    auto codegen = lower_schedule_to_llvm(
        *module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.argument_buffer_size == 32u);
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("uniform_switch_result.slot = alloca i32") !=
          std::string::npos);
    CHECK(ir.find("uniform.switch.default") != std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    alignas(16) std::array<std::byte, 32u> arguments{};
    auto config = launch_1d(width, 32u);
    auto bias_value = uint32_t{7u};
    std::memcpy(arguments.data() + 16u,
                &bias_value, sizeof(bias_value));
    for (auto selector_value : {0u, 2u, 99u}) {
        std::memcpy(arguments.data(),
                    &selector_value, sizeof(selector_value));
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        function(arguments.data(), output.data(), &config, width);
        auto addend = selector_value == 0u ? 11u :
                      selector_value == 2u ? 22u :
                                             33u;
        for (auto value : output) {
            CHECK(value == bias_value + addend);
        }
    }
    return true;
}

[[nodiscard]] bool run_varying_switch_codegen() {
    static constexpr auto width = 8u;
    auto schedule_function = make_varying_switch(width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-varying-switch", *context);
    auto name = std::string{"simd_varying_switch"};
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("varying.switch.coherent") != std::string::npos);
    CHECK(ir.find("varying.switch.divergent") != std::string::npos);
    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(module), std::move(context)));
    using Entry = void(
        const void *, uint32_t *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    for (auto active_lanes :
         {uint32_t{8u}, uint32_t{6u}, uint32_t{1u}}) {
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        auto config = launch_1d(active_lanes, width);
        function(nullptr, output.data(), &config, active_lanes);
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            if (lane >= active_lanes) {
                CHECK(output[lane] == 0xdeadbeefu);
                continue;
            }
            auto addend = lane == 0u ? 10u :
                          lane == 2u ? 20u :
                          lane == 5u ? 30u :
                                       40u;
            CHECK(output[lane] == lane + addend);
        }
    }
    return true;
}

template<size_t Width>
[[nodiscard]] bool run_switch_loop_exits_codegen() {
    auto schedule_function = make_switch_loop_with_exits(Width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-switch-loop-exits", *context);
    auto name = std::string{"simd_switch_loop_exits_w"} +
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
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    std::array<uint32_t, Width> output{};
    output.fill(0xdeadbeefu);
    auto config = launch_1d(Width, Width);
    function(nullptr, output.data(), &config, Width);
    for (auto lane = uint32_t{0u}; lane < Width; lane++) {
        auto expected = lane % 4u == 2u ? lane + 200u :
                                          lane + 100u;
        CHECK(output[lane] == expected);
    }
    return true;
}

[[nodiscard]] bool run_multiple_backedge_loop_codegen() {
    static constexpr auto width = 8u;
    auto schedule_function = make_multiple_backedge_loop(width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-multiple-backedge-loop", *context);
    auto name = std::string{"simd_multiple_backedge_loop"};
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width, name);
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
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    std::array<uint32_t, width> output{};
    output.fill(0xdeadbeefu);
    auto config = launch_1d(width, width);
    function(nullptr, output.data(), &config, width);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        CHECK(output[lane] == lane + 1u);
    }
    return true;
}

[[nodiscard]] bool run_non_dominating_convergence_codegen() {
    static constexpr auto width = 8u;
    auto schedule_function = make_non_dominating_convergence(width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-non-dominating-convergence", *context);
    auto name = std::string{"simd_non_dominating_convergence"};
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width, name);
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
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    for (auto active_lanes : {uint32_t{5u}, uint32_t{8u}}) {
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        auto config = launch_1d(active_lanes, width);
        function(nullptr, output.data(), &config, active_lanes);
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            CHECK(output[lane] ==
                  (lane < active_lanes ? active_lanes : 0xdeadbeefu));
        }
    }
    return true;
}

[[nodiscard]] bool run_return_convergence_cascade_codegen() {
    static constexpr auto width = 8u;
    auto schedule_function = make_return_convergence_cascade(width);
    CHECK(schedule_function.has_value());
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto module = std::make_unique<::llvm::Module>(
        "simd-return-convergence-cascade", *context);
    auto name = std::string{"simd_return_convergence_cascade"};
    auto codegen = lower_schedule_to_llvm(
        *module, *schedule_function, width, name);
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
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    for (auto active_lanes : {uint32_t{5u}, uint32_t{8u}}) {
        std::array<uint32_t, width> output{};
        output.fill(0xdeadbeefu);
        auto config = launch_1d(active_lanes, width);
        function(nullptr, output.data(), &config, active_lanes);
        auto expected_live = active_lanes - 2u;
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            auto expected = lane >= active_lanes    ? 0xdeadbeefu :
                            lane >= 2u && lane < 4u ? lane + 100u :
                                                      expected_live;
            CHECK(output[lane] == expected);
        }
    }
    return true;
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

struct TexturePacketProbe {
    bool valid{true};
    uint32_t read_calls{0u};
    uint32_t write_calls{0u};
    uint32_t lane_count{0u};
    uint64_t read_mask{0u};
    uint64_t write_mask{0u};
    std::array<uint32_t, 8u> x{};
    std::array<uint32_t, 8u> y{};
    std::array<uint32_t, 8u> z{};
};

void texture_packet_read_probe(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, void *values) {
    auto *probe = static_cast<TexturePacketProbe *>(texture);
    probe->read_calls++;
    probe->lane_count = lane_count;
    probe->read_mask = active_mask_bits;
    probe->valid &= level == 3u && lane_count == 8u;
    auto *components = static_cast<float *>(values);
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if ((active_mask_bits & (uint64_t{1u} << lane)) == 0u) {
            continue;
        }
        probe->x[lane] = x[lane];
        probe->y[lane] = y[lane];
        probe->z[lane] = z[lane];
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            components[component * lane_count + lane] =
                static_cast<float>(100u * component + x[lane]);
        }
    }
}

void texture_packet_write_probe(
    void *texture, uint32_t level, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *x,
    const uint32_t *y, const uint32_t *z, const void *values) {
    auto *probe = static_cast<TexturePacketProbe *>(texture);
    probe->write_calls++;
    probe->write_mask = active_mask_bits;
    probe->valid &= level == 3u && lane_count == 8u;
    auto *components = static_cast<const float *>(values);
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if ((active_mask_bits & (uint64_t{1u} << lane)) == 0u) {
            continue;
        }
        probe->valid &= x[lane] == probe->x[lane] &&
                        y[lane] == probe->y[lane] &&
                        z[lane] == probe->z[lane];
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            probe->valid &=
                components[component * lane_count + lane] ==
                static_cast<float>(100u * component + x[lane]);
        }
    }
}

uint32_t texture_packet_size_probe(
    void *, uint32_t, uint32_t) {
    return 8u;
}

[[nodiscard]] bool run_texture_packet_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("texture_packet_read_write");
    auto *texture_type = Type::texture(Type::of<float>(), 2u);
    auto *texture = kernel->create_resource_argument(texture_type);
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *y = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, one});
    auto *coordinate = builder.call(
        Type::of<luisa::uint2>(), xir::ArithmeticOp::AGGREGATE,
        {x, y});
    auto *pixel = builder.call(
        Type::of<luisa::float4>(),
        xir::ResourceReadOp::TEXTURE2D_READ,
        {texture, coordinate});
    builder.call(
        xir::ResourceWriteOp::TEXTURE2D_WRITE,
        {texture, coordinate, pixel});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-texture-packet", *context);
    auto name = std::string{"simd_texture_packet"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.argument_buffer_size == sizeof(SIMDHostTextureView));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("texture.read.packet") != std::string::npos);
    CHECK(ir.find("texture.write.packet") != std::string::npos);
    CHECK(ir.find("texture.read.lane") == std::string::npos);
    CHECK(ir.find("texture.write.lane") == std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    TexturePacketProbe probe;
    auto texture_view = SIMDHostTextureView{
        .texture = &probe,
        .read_float = texture_packet_read_probe,
        .read_uint = texture_packet_read_probe,
        .write_float = texture_packet_write_probe,
        .write_uint = texture_packet_write_probe,
        .size = texture_packet_size_probe,
        .level = 3u,
        .dimension = 2u,
    };
    auto config = launch_1d(active_lanes, width);
    function(&texture_view, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.read_calls == 1u);
    CHECK(probe.write_calls == 1u);
    CHECK(probe.lane_count == width);
    CHECK(probe.read_mask == 0x1fu);
    CHECK(probe.write_mask == 0x1fu);
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        CHECK(probe.x[lane] == lane);
        CHECK(probe.y[lane] == 0u);
        CHECK(probe.z[lane] == 0u);
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

[[nodiscard]] bool run_ast_fast_math_canonicalization() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferFloat output) noexcept {
        auto index = dispatch_id().x;
        auto exponent = cast<float>(index) * 0.25f - 1.0f;
        output.write(
            index,
            pow(2.0f, exponent) + pow(10.0f, exponent));
    };
    auto precise = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_radix_pow_precise", false);
    CHECK(precise.succeeded());
    CHECK(precise.fast_math_identity_count == 0u);
    CHECK(precise.fast_math_radix_pow_count == 0u);

    auto fast = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_radix_pow_fast", true);
    if (!fast.succeeded()) {
        for (auto &&diagnostic : fast.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(fast.fast_math_identity_count == 0u);
    CHECK(fast.fast_math_radix_pow_count == 2u);
    CHECK(fast.argument_buffer_size == sizeof(SIMDHostBufferView));

    std::array<float, count> output{};
    output.fill(-1.0f);
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(fast.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(&argument, nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto exponent = static_cast<float>(i) * 0.25f - 1.0f;
        auto expected =
            std::exp2(exponent) + std::pow(10.0f, exponent);
        auto error = std::abs(output[i] - expected);
        CHECK(error <= 2.0e-3f * (1.0f + std::abs(expected)));
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
        {"Schedule IR vector warp2", &run_codegen<2u>},
        {"Schedule IR vector warp4", &run_codegen<4u>},
        {"Schedule IR vector warp8", &run_codegen<8u>},
        {"Schedule IR vector warp16", &run_codegen<16u>},
        {"static power-of-two block size",
         &run_static_block_size_codegen},
        {"Schedule IR loop warp1", &run_loop_codegen<1u>},
        {"Schedule IR loop warp2", &run_loop_codegen<2u>},
        {"Schedule IR loop warp4", &run_loop_codegen<4u>},
        {"Schedule IR loop warp8", &run_loop_codegen<8u>},
        {"varying loop exit collective",
         &run_varying_loop_collective_codegen},
        {"multiple-exit loop collective",
         &run_multiple_exit_loop_collective_codegen},
        {"Schedule IR nested convergence", &run_nested_codegen},
        {"Schedule IR 96-block CFG", &run_large_cfg_codegen},
        {"scheduler state residency", &run_state_residency_codegen},
        {"scalar uniform values", &run_uniform_value_codegen},
        {"scalar uniform switch", &run_uniform_switch_codegen},
        {"varying switch convergence", &run_varying_switch_codegen},
        {"scalar switch loop exits",
         &run_switch_loop_exits_codegen<1u>},
        {"switch loop exits", &run_switch_loop_exits_codegen<8u>},
        {"multiple loop backedges", &run_multiple_backedge_loop_codegen},
        {"dynamic non-dominating convergence",
         &run_non_dominating_convergence_codegen},
        {"return convergence cascade",
         &run_return_convergence_cascade_codegen},
        {"XIR compiler facade", &run_compiler_facade},
        {"XIR buffer vector gather/scatter", &run_buffer_vector_codegen},
        {"XIR texture packet callback", &run_texture_packet_codegen},
        {"AST buffer dispatch", &run_ast_buffer_codegen},
        {"AST select operand order", &run_ast_select_codegen},
        {"AST fast radix pow canonicalization",
         &run_ast_fast_math_canonicalization},
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
