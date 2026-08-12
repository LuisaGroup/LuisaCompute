#include "llvm_schedule_codegen.h"
#include "llvm_jit.h"
#include "predicated_if_conversion.h"
#include "simd_compiler.h"
#include "warp_uniformity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

void set_environment_variable(
    const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

struct ScopedEnvironmentVariable {
    std::string name;
    std::optional<std::string> previous;

    explicit ScopedEnvironmentVariable(
        const char *env_name, const char *value)
        : name{env_name} {
        if (auto *old_value = std::getenv(env_name)) {
            previous.emplace(old_value);
        }
        set_environment_variable(name.c_str(), value);
    }

    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            name.c_str(),
            previous ? previous->c_str() : nullptr);
    }

    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

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
    if (codegen.direct_control_flow) {
        CHECK(ir.find("ready.token") == std::string::npos);
        CHECK(convergence_target_count == 0u);
        CHECK(count_occurrences(ir, "\nconvergence.cascade") == 0u);
    } else {
        CHECK(ir.find("ready.token") != std::string::npos);
        CHECK(count_occurrences(ir, "\nconvergence.cascade") ==
              2u * convergence_target_count);
    }
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
    CHECK(codegen.direct_control_flow);
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
    CHECK(ir.find("direct.switch.default") != std::string::npos);

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
        kernel, 8u, "simd_compiler_facade", false,
        true, true, true);
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.argument_buffer_size == 0u);
    CHECK(!compiled.target_triple.empty());
    CHECK(!compiled.direct_control_flow);
    CHECK(!compiled.assembly.empty());
    CHECK(compiled.assembly.find("simd_compiler_facade") !=
          std::string::npos);
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
    CHECK(compiled.uniform_buffer_broadcast_count == 0u);

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

[[nodiscard]] bool run_faceforward_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("faceforward");
    auto *buffer_type = Type::buffer(Type::of<luisa::float3>());
    auto *normal_argument = kernel->create_resource_argument(buffer_type);
    auto *incident_argument = kernel->create_resource_argument(buffer_type);
    auto *reference_argument = kernel->create_resource_argument(buffer_type);
    auto *output_argument = kernel->create_resource_argument(buffer_type);
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *normal = builder.call(
        Type::of<luisa::float3>(), xir::ResourceReadOp::BUFFER_READ,
        {normal_argument, index});
    auto *incident = builder.call(
        Type::of<luisa::float3>(), xir::ResourceReadOp::BUFFER_READ,
        {incident_argument, index});
    auto *reference = builder.call(
        Type::of<luisa::float3>(), xir::ResourceReadOp::BUFFER_READ,
        {reference_argument, index});
    auto *result = builder.call(
        Type::of<luisa::float3>(), xir::ArithmeticOp::FACEFORWARD,
        {normal, incident, reference});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output_argument, index, result});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-faceforward", *context);
    auto name = std::string{"simd_faceforward"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.argument_buffer_size ==
          4u * sizeof(SIMDHostBufferView));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(count_occurrences(ir, "fcmp olt <8 x float>") == 1u);
    CHECK(count_occurrences(ir, "select <8 x i1>") >= 3u);
    CHECK(count_occurrences(ir, "fneg <8 x float>") == 3u);
    CHECK(ir.find("llvm.x86.") == std::string::npos);
    CHECK(ir.find("llvm.aarch64.") == std::string::npos);
    CHECK(ir.find("llvm.arm.neon.") == std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    std::array<luisa::float3, count> normals{};
    std::array<luisa::float3, count> incidents{};
    std::array<luisa::float3, count> references{};
    std::array<luisa::float3, count> output{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        normals[i] = luisa::make_float3(
            static_cast<float>(i) + 0.25f,
            1.0f - static_cast<float>(i) * 0.125f,
            -0.75f);
        incidents[i] = luisa::make_float3(
            (i & 1u) == 0u ? -1.0f : 1.0f, 0.0f, 0.0f);
        references[i] = luisa::make_float3(1.0f, 0.0f, 0.0f);
        output[i] = luisa::make_float3(1234.0f);
    }
    alignas(16) std::array<SIMDHostBufferView, 4u> arguments{
        SIMDHostBufferView{normals.data(), sizeof(normals)},
        SIMDHostBufferView{incidents.data(), sizeof(incidents)},
        SIMDHostBufferView{references.data(), sizeof(references)},
        SIMDHostBufferView{output.data(), sizeof(output)},
    };
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        function(arguments.data(), nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto expected = (i & 1u) == 0u ? normals[i] : -normals[i];
        CHECK(output[i].x == expected.x);
        CHECK(output[i].y == expected.y);
        CHECK(output[i].z == expected.z);
    }
    return true;
}

[[nodiscard]] std::optional<schedule::Function>
make_lane_affine_buffer_schedule(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("lane_affine_buffer");
    kernel->set_block_size(luisa::make_uint3(64u, 1u, 1u));
    auto *buffer_type = Type::buffer(Type::of<float>());
    auto *lhs = kernel->create_resource_argument(buffer_type);
    auto *rhs = kernel->create_resource_argument(buffer_type);
    auto *output = kernel->create_resource_argument(buffer_type);
    auto *entry = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    uint32_t five_value = 5u;
    uint32_t nine_value = 9u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *five = module.create_constant(
        Type::of<uint32_t>(), &five_value);
    auto *nine = module.create_constant(
        Type::of<uint32_t>(), &nine_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *column = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *row = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, one});
    auto *lhs_row = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_MUL,
        {row, five});
    auto *lhs_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lhs_row, two});
    auto *rhs_row = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_MUL,
        {two, nine});
    auto *rhs_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {rhs_row, column});
    auto *output_row = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_MUL,
        {row, nine});
    auto *output_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {output_row, column});
    auto *a = builder.call(
        Type::of<float>(), xir::ResourceReadOp::BUFFER_READ,
        {lhs, lhs_index});
    auto *b = builder.call(
        Type::of<float>(), xir::ResourceReadOp::BUFFER_READ,
        {rhs, rhs_index});
    auto *product = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL, {a, b});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, output_index, product});
    builder.return_void();
    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] bool run_lane_affine_buffer_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto count = 9u;
    auto schedule_function =
        make_lane_affine_buffer_schedule(width);
    CHECK(schedule_function.has_value());

    struct ModuleBundle {
        std::unique_ptr<::llvm::LLVMContext> context;
        std::unique_ptr<::llvm::Module> module;
        LLVMScheduleCodegenResult codegen;
    };
    auto make_module = [&](std::string_view module_name,
                           std::string_view entry_name,
                           bool enable_lane_affine) {
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            std::string{module_name}, *context);
        auto codegen = lower_schedule_to_llvm(
            *module, *schedule_function, width, entry_name,
            false, {64u, 1u, 1u}, true,
            enable_lane_affine);
        return ModuleBundle{
            std::move(context), std::move(module),
            std::move(codegen)};
    };

    auto enabled_bundle = make_module(
        "simd-lane-affine-buffer",
        "simd_lane_affine_buffer", true);
    CHECK(enabled_bundle.codegen.succeeded());
    CHECK(enabled_bundle.codegen.uniform_buffer_broadcast_count == 1u);
    CHECK(enabled_bundle.codegen.contiguous_buffer_read_count == 1u);
    CHECK(enabled_bundle.codegen.contiguous_buffer_write_count == 1u);
    CHECK(!::llvm::verifyModule(
        *enabled_bundle.module, &::llvm::errs()));
    std::string enabled_ir;
    ::llvm::raw_string_ostream enabled_stream{enabled_ir};
    enabled_bundle.module->print(enabled_stream, nullptr);
    enabled_stream.flush();
    CHECK(enabled_ir.find("llvm.masked.load") != std::string::npos);
    CHECK(enabled_ir.find("llvm.masked.store") != std::string::npos);
    CHECK(enabled_ir.find("llvm.masked.gather") == std::string::npos);
    CHECK(enabled_ir.find("llvm.masked.scatter") == std::string::npos);

    auto assembly_bundle = make_module(
        "simd-lane-affine-assembly",
        "simd_lane_affine_assembly", true);
    CHECK(assembly_bundle.codegen.succeeded());
    LLVMJIT assembly_target;
    CHECK(assembly_target.succeeded());
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_bundle.module),
        std::move(assembly_bundle.context));
    CHECK(!assembly.empty());
    CHECK(assembly.find("gather") == std::string::npos);
    CHECK(assembly.find("scatter") == std::string::npos);

    auto disabled_bundle = make_module(
        "simd-lane-affine-disabled",
        "simd_lane_affine_disabled", false);
    CHECK(disabled_bundle.codegen.succeeded());
    CHECK(disabled_bundle.codegen.uniform_buffer_broadcast_count == 1u);
    CHECK(disabled_bundle.codegen.contiguous_buffer_read_count == 0u);
    CHECK(disabled_bundle.codegen.contiguous_buffer_write_count == 0u);
    std::string disabled_ir;
    ::llvm::raw_string_ostream disabled_stream{disabled_ir};
    disabled_bundle.module->print(disabled_stream, nullptr);
    disabled_stream.flush();
    CHECK(disabled_ir.find("llvm.masked.gather") != std::string::npos);
    CHECK(disabled_ir.find("llvm.masked.scatter") != std::string::npos);

    std::array<float, 5u> lhs{};
    std::array<float, 45u> rhs{};
    std::array<float, count> output{};
    for (auto i = size_t{0u}; i < lhs.size(); i++) {
        lhs[i] = static_cast<float>(i + 1u);
    }
    for (auto i = size_t{0u}; i < rhs.size(); i++) {
        rhs[i] = static_cast<float>(i) * 0.25f - 3.0f;
    }
    output.fill(-1234.0f);
    alignas(16) std::array<SIMDHostBufferView, 3u> arguments{
        SIMDHostBufferView{lhs.data(), sizeof(lhs)},
        SIMDHostBufferView{rhs.data(), sizeof(rhs)},
        SIMDHostBufferView{output.data(), sizeof(output)},
    };
    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(enabled_bundle.module),
        std::move(enabled_bundle.context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(
        jit.lookup("simd_lane_affine_buffer"));
    CHECK(function != nullptr);
    auto config = launch_1d(count, 64u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        function(arguments.data(), nullptr, &config, width);
    }
    for (auto i = size_t{0u}; i < output.size(); i++) {
        CHECK(output[i] == lhs[2u] * rhs[18u + i]);
    }

    auto w2_schedule = make_lane_affine_buffer_schedule(2u);
    CHECK(w2_schedule.has_value());
    auto w2_context = std::make_unique<::llvm::LLVMContext>();
    auto w2_module = std::make_unique<::llvm::Module>(
        "simd-lane-affine-w2-policy", *w2_context);
    auto w2_codegen = lower_schedule_to_llvm(
        *w2_module, *w2_schedule, 2u,
        "simd_lane_affine_w2_policy",
        false, {64u, 1u, 1u}, true, true);
    CHECK(w2_codegen.succeeded());
    CHECK(w2_codegen.contiguous_buffer_read_count == 0u);
    CHECK(w2_codegen.contiguous_buffer_write_count == 0u);
    std::string w2_ir;
    ::llvm::raw_string_ostream w2_stream{w2_ir};
    w2_module->print(w2_stream, nullptr);
    w2_stream.flush();
    CHECK(w2_ir.find("llvm.masked.gather") != std::string::npos);
    CHECK(w2_ir.find("llvm.masked.scatter") != std::string::npos);

    Kernel1D sparse_kernel = [](BufferFloat input,
                                BufferFloat result) noexcept {
        auto lane = dispatch_id().x;
        $if ((lane & 1u) != 0u) {
            auto index = lane - 1u;
            result.write(index, input.read(index));
        };
    };
    auto sparse = compile_simd_kernel(
        sparse_kernel.function()->function(), width,
        "simd_lane_affine_sparse_cohort");
    CHECK(sparse.succeeded());
    CHECK(sparse.contiguous_buffer_read_count == 1u);
    CHECK(sparse.contiguous_buffer_write_count == 1u);
    std::array<float, width> sparse_input{};
    std::array<float, width> sparse_output{};
    for (auto i = size_t{0u}; i < width; i++) {
        sparse_input[i] = static_cast<float>(i) + 0.25f;
    }
    sparse_output.fill(-777.0f);
    alignas(16) std::array<SIMDHostBufferView, 2u> sparse_arguments{
        SIMDHostBufferView{sparse_input.data(), sizeof(sparse_input)},
        SIMDHostBufferView{sparse_output.data(), sizeof(sparse_output)},
    };
    auto sparse_entry = reinterpret_cast<Entry *>(sparse.entry);
    CHECK(sparse_entry != nullptr);
    auto sparse_config = launch_1d(width, 64u);
    sparse_entry(
        sparse_arguments.data(), nullptr, &sparse_config, width);
    for (auto i = size_t{0u}; i < width; i++) {
        auto expected = (i & 1u) == 0u ?
                            sparse_input[i] :
                            -777.0f;
        CHECK(sparse_output[i] == expected);
    }
    return true;
}

[[nodiscard]] std::optional<schedule::Function>
make_uniform_buffer_broadcast_schedule(
    uint32_t width, bool volatile_read) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name(volatile_read ?
                         "volatile_uniform_buffer_read" :
                         "uniform_buffer_broadcast");
    auto *buffer_type = Type::buffer(Type::of<uint32_t>());
    auto *input = kernel->create_resource_argument(buffer_type);
    auto *output = kernel->create_resource_argument(buffer_type);
    auto *uniform_index = kernel->create_value_argument(
        Type::of<uint32_t>());
    auto *entry = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *output_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto read_op = volatile_read ?
                       xir::ResourceReadOp::BUFFER_VOLATILE_READ :
                       xir::ResourceReadOp::BUFFER_READ;
    auto *argument_read = builder.call(
        Type::of<uint32_t>(), read_op,
        {input, uniform_index});
    auto *cohort_index = builder.call(
        Type::of<uint32_t>(),
        xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE,
        {lane});
    auto *cohort_read = builder.call(
        Type::of<uint32_t>(), read_op,
        {input, cohort_index});
    auto *sum = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {argument_read, cohort_read});
    auto *result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {sum, output_index});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, output_index, result});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] bool run_uniform_buffer_broadcast_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto count = 11u;
    auto schedule_function =
        make_uniform_buffer_broadcast_schedule(width, false);
    CHECK(schedule_function.has_value());

    struct ModuleBundle {
        std::unique_ptr<::llvm::LLVMContext> context;
        std::unique_ptr<::llvm::Module> module;
        LLVMScheduleCodegenResult codegen;
    };
    auto make_module = [&](const schedule::Function &source,
                           std::string_view module_name,
                           std::string_view entry_name,
                           bool enable_broadcast) {
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            std::string{module_name}, *context);
        auto codegen = lower_schedule_to_llvm(
            *module, source, width, entry_name,
            false, {}, enable_broadcast);
        return ModuleBundle{
            std::move(context), std::move(module),
            std::move(codegen)};
    };

    auto enabled_bundle = make_module(
        *schedule_function,
        "simd-uniform-buffer-broadcast",
        "simd_uniform_buffer_broadcast", true);
    auto &enabled = enabled_bundle.codegen;
    auto &enabled_module = enabled_bundle.module;
    CHECK(enabled.succeeded());
    CHECK(enabled.argument_buffer_size == 48u);
    CHECK(enabled.uniform_buffer_broadcast_count == 2u);
    CHECK(enabled.contiguous_buffer_write_count == 1u);
    CHECK(!::llvm::verifyModule(*enabled_module, &::llvm::errs()));
    std::string enabled_ir;
    ::llvm::raw_string_ostream enabled_stream{enabled_ir};
    enabled_module->print(enabled_stream, nullptr);
    enabled_stream.flush();
    CHECK(enabled_ir.find("llvm.masked.gather") == std::string::npos);
    CHECK(enabled_ir.find("llvm.masked.load") == std::string::npos);
    CHECK(enabled_ir.find("llvm.masked.store") != std::string::npos);
    CHECK(enabled_ir.find("llvm.masked.scatter") == std::string::npos);

    auto use_site_schedule =
        make_uniform_buffer_broadcast_schedule(width, false);
    CHECK(use_site_schedule.has_value());
    auto annotated_read_count = size_t{0u};
    for (auto &block : use_site_schedule->blocks()) {
        for (auto &instruction : block.instructions) {
            if (instruction.opcode != schedule::Opcode::resource_read ||
                instruction.operands.size() != 2u) {
                continue;
            }
            auto *index = use_site_schedule->value(
                instruction.operands[1u]);
            if (index == nullptr ||
                index->origin != schedule::ValueOrigin::instruction) {
                continue;
            }
            index->value_class = schedule::ValueClass::varying;
            instruction.cohort_uniform_operand_index = 1u;
            annotated_read_count++;
        }
    }
    CHECK(annotated_read_count == 1u);
    auto assembly_bundle = make_module(
        *use_site_schedule,
        "simd-uniform-buffer-broadcast-assembly",
        "simd_uniform_buffer_broadcast_assembly", true);
    CHECK(assembly_bundle.codegen.succeeded());
    LLVMJIT assembly_target;
    CHECK(assembly_target.succeeded());
    auto assembly = assembly_target.emit_assembly(
        std::move(assembly_bundle.module),
        std::move(assembly_bundle.context));
    CHECK(!assembly.empty());
    CHECK(assembly.find("gather") == std::string::npos);

    auto disabled_bundle = make_module(
        *schedule_function,
        "simd-uniform-buffer-gather",
        "simd_uniform_buffer_gather", false);
    auto &disabled = disabled_bundle.codegen;
    auto &disabled_module = disabled_bundle.module;
    CHECK(disabled.succeeded());
    CHECK(disabled.uniform_buffer_broadcast_count == 0u);
    CHECK(!::llvm::verifyModule(*disabled_module, &::llvm::errs()));
    std::string disabled_ir;
    ::llvm::raw_string_ostream disabled_stream{disabled_ir};
    disabled_module->print(disabled_stream, nullptr);
    disabled_stream.flush();
    CHECK(count_occurrences(
              disabled_ir, "llvm.masked.gather") >= 2u);

    auto volatile_schedule =
        make_uniform_buffer_broadcast_schedule(width, true);
    CHECK(volatile_schedule.has_value());
    auto volatile_context =
        std::make_unique<::llvm::LLVMContext>();
    auto volatile_module = std::make_unique<::llvm::Module>(
        "simd-volatile-uniform-buffer-read",
        *volatile_context);
    auto volatile_codegen = lower_schedule_to_llvm(
        *volatile_module, *volatile_schedule, width,
        "simd_volatile_uniform_buffer_read",
        false, {}, true);
    CHECK(volatile_codegen.succeeded());
    CHECK(volatile_codegen.uniform_buffer_broadcast_count == 0u);
    std::string volatile_ir;
    ::llvm::raw_string_ostream volatile_stream{volatile_ir};
    volatile_module->print(volatile_stream, nullptr);
    volatile_stream.flush();
    CHECK(count_occurrences(
              volatile_ir, "llvm.masked.gather") >= 2u);

    struct alignas(16) Arguments {
        SIMDHostBufferView input{};
        SIMDHostBufferView output{};
        uint32_t index{0u};
        std::array<std::byte, 12u> padding{};
    };
    static_assert(sizeof(Arguments) == 48u);
    std::array<uint32_t, 16u> input{};
    std::array<uint32_t, count> output{};
    for (auto i = uint32_t{0u}; i < input.size(); i++) {
        input[i] = 7u + i * 13u;
    }
    output.fill(0xdeadbeefu);
    Arguments arguments{
        .input = {input.data(), sizeof(input)},
        .output = {output.data(), sizeof(output)},
        .index = 5u,
    };
    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(
        std::move(enabled_module),
        std::move(enabled_bundle.context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(
        jit.lookup("simd_uniform_buffer_broadcast"));
    CHECK(function != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        function(&arguments, nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        CHECK(output[i] == input[5u] + input[0u] + i);
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

[[nodiscard]] bool run_accel_instance_metadata_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("accel_instance_metadata");
    auto *accel = kernel->create_resource_argument(Type::of<Accel>());
    auto *metadata_output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::uint4>()));
    auto *transform_output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::float4x4>()));
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *instance_id = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {x, one});
    auto *user_id = builder.call(
        Type::of<uint32_t>(),
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
        {accel, instance_id});
    auto *visibility = builder.call(
        Type::of<uint32_t>(),
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK,
        {accel, instance_id});
    auto *uniform_user_id = builder.call(
        Type::of<uint32_t>(),
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
        {accel, zero});
    auto *metadata = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::AGGREGATE,
        {instance_id, user_id, visibility, uniform_user_id});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {metadata_output, x, metadata});
    auto *transform = builder.call(
        Type::of<luisa::float4x4>(),
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM,
        {accel, instance_id});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {transform_output, x, transform});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {accel, instance_id, transform});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK,
        {accel, instance_id, visibility});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID,
        {accel, instance_id, user_id});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID,
        {accel, zero, uniform_user_id});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-accel-instance-metadata", *context);
    auto name = std::string{"simd_accel_instance_metadata"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        SIMDHostBufferView metadata_output;
        SIMDHostBufferView transform_output;
    };
    CHECK(codegen.argument_buffer_size == sizeof(Arguments));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(count_occurrences(ir, "llvm.masked.gather") >= 14u);
    CHECK(count_occurrences(ir, "llvm.masked.scatter") >= 17u);
    CHECK(ir.find("accel.instance.scalar.load") != std::string::npos);
    CHECK(ir.find("accel.instance.scalar.store") != std::string::npos);
    CHECK(ir.find("call void %") == std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);

    std::array<SIMDHostAccelInstance, 2u> instances{};
    std::array affine0{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f};
    std::array affine1{
        2.0f, 7.0f, 17.0f, 29.0f,
        3.0f, 11.0f, 19.0f, 31.0f,
        5.0f, 13.0f, 23.0f, 37.0f};
    std::memcpy(
        instances[0u].affine, affine0.data(), sizeof(affine0));
    std::memcpy(
        instances[1u].affine, affine1.data(), sizeof(affine1));
    instances[0u].user_id = 11u;
    instances[0u].mask = 0x1u;
    instances[1u].user_id = 22u;
    instances[1u].mask = 0x2u;
    SIMDHostAccelInstanceTable instance_table{
        .data = instances.data(),
        .size = instances.size(),
    };
    std::array<luisa::uint4, width> metadata_values{};
    std::array<luisa::float4x4, width> transform_values{};
    Arguments arguments{
        .accel = {
            .instances = &instance_table,
        },
        .metadata_output = {metadata_values.data(), sizeof(metadata_values)},
        .transform_output = {transform_values.data(), sizeof(transform_values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    auto expected0 = luisa::make_float4x4(1.0f);
    auto expected1 = luisa::make_float4x4(
        luisa::make_float4(2.0f, 3.0f, 5.0f, 0.0f),
        luisa::make_float4(7.0f, 11.0f, 13.0f, 0.0f),
        luisa::make_float4(17.0f, 19.0f, 23.0f, 0.0f),
        luisa::make_float4(29.0f, 31.0f, 37.0f, 1.0f));
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        auto instance_id = lane & 1u;
        CHECK(luisa::all(
            metadata_values[lane] == luisa::make_uint4(
                                         instance_id,
                                         instance_id == 0u ? 11u : 22u,
                                         instance_id == 0u ? 0x1u : 0x2u,
                                         11u)));
        auto expected = instance_id == 0u ? expected0 : expected1;
        for (auto column = uint32_t{0u}; column < 4u; column++) {
            CHECK(luisa::all(
                transform_values[lane][column] == expected[column]));
        }
    }
    CHECK(instances[0u].dirty == 1u);
    CHECK(instances[1u].dirty == 1u);
    return true;
}

struct RayQueryPacketProbe {
    uint32_t calls{0u};
    uint32_t lane_count{0u};
    uint64_t mask{0u};
    uint32_t expected_lane_count{0u};
    uint64_t expected_mask{0u};
    SIMDHostAccelRayQueryProceed *expected_proceed{nullptr};
    bool valid{true};
};

void ray_query_packet_probe_impl(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    auto *probe = static_cast<RayQueryPacketProbe *>(states[0u]->accel);
    probe->calls++;
    probe->lane_count = lane_count;
    probe->mask = active_mask_bits;
    probe->valid &= lane_count == probe->expected_lane_count &&
                    active_mask_bits == probe->expected_mask;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        auto active = (active_mask_bits & (uint64_t{1u} << lane)) != 0u;
        if (!active) {
            probe->valid &= states[lane] == nullptr;
            continue;
        }
        auto *state = states[lane];
        probe->valid &= state != nullptr && state->accel == probe &&
                        state->proceed == probe->expected_proceed &&
                        state->time == 0.0f &&
                        state->visibility_mask == 0x5au &&
                        state->terminate_on_first == 0u &&
                        state->cursor_valid == 0u &&
                        state->candidate_kind == 0u &&
                        state->candidate_committed == 0u &&
                        state->terminated == 0u &&
                        state->candidate_batch_count == 0u &&
                        state->candidate_batch_index == 0u &&
                        state->candidate_batch_has_more == 0u &&
                        state->candidate_batch_initialized == 0u &&
                        state->committed.inst == ~0u &&
                        state->committed.prim == ~0u &&
                        state->committed.kind == 0u &&
                        state->committed.t == 0.0f;
        constexpr std::array expected_ray{
            1.0f, 2.0f, 3.0f, 0.25f,
            4.0f, 5.0f, 6.0f, 7.0f};
        for (auto component = uint32_t{0u};
             component < expected_ray.size(); component++) {
            probe->valid &=
                state->world_ray[component] == expected_ray[component];
        }
        for (auto previous = uint32_t{0u}; previous < lane; previous++) {
            probe->valid &= states[previous] != state;
        }
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = 10u + lane,
            .prim = 20u + lane,
            .bary = {0.125f, 0.25f},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = 30.0f + static_cast<float>(lane),
        };
        state->terminated = 1u;
    }
}

void ray_query_packet_probe_narrow(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    ray_query_packet_probe_impl(
        lane_count, active_mask_bits, states);
}

void ray_query_packet_probe_wide(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    ray_query_packet_probe_impl(
        lane_count, active_mask_bits, states);
}

[[nodiscard]] bool run_ray_query_packet_codegen_case(
    uint32_t width, uint32_t active_lanes,
    bool expect_wide) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("ray_query_packet");
    auto *accel = kernel->create_resource_argument(Type::of<Accel>());
    auto *ray = kernel->create_value_argument(Type::of<Ray>());
    auto *output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::uint4>()));
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto three_value = uint32_t{3u};
    auto visibility_value = uint32_t{0x5au};
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    auto *visibility = module.create_constant(
        Type::of<uint32_t>(), &visibility_value);
    auto *query_type = Type::custom("LC_RayQueryAll");
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *query_value = builder.call(
        query_type, xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
        {accel, ray, visibility});
    auto *query = builder.alloca_local(query_type);
    builder.store(query, query_value);
    builder.call(
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
        {query});
    auto *committed = builder.call(
        Type::of<CommittedHit>(),
        xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
        {query});
    auto *inst = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {committed, zero});
    auto *prim = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {committed, one});
    auto *kind = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {committed, three});
    auto *metadata = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::AGGREGATE,
        {inst, prim, kind, x});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, x, metadata});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-ray-query-packet", *context);
    auto name = std::string{"simd_ray_query_packet"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        Ray ray;
        SIMDHostBufferView output;
    };
    CHECK(codegen.argument_buffer_size == sizeof(Arguments));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("ray.query.state") != std::string::npos);
    CHECK(ir.find("ray.query.packet") != std::string::npos);
    CHECK(ir.find("ray.query.proceed.lane") == std::string::npos);
    CHECK(count_occurrences(ir, "call void %") == 1u);
    CHECK(count_occurrences(ir, "llvm.masked.scatter") >= 20u);
    CHECK(count_occurrences(ir, "llvm.masked.gather") >= 9u);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);

    RayQueryPacketProbe probe;
    luisa::vector<luisa::uint4> values(width);
    std::fill(
        values.begin(), values.end(),
        luisa::make_uint4(0xdeadbeefu));
    auto expected_proceed = expect_wide ?
                                ray_query_packet_probe_wide :
                                ray_query_packet_probe_narrow;
    probe.expected_lane_count = width;
    probe.expected_mask = (uint64_t{1u} << active_lanes) - 1u;
    probe.expected_proceed = expected_proceed;
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .ray_query_proceed = ray_query_packet_probe_narrow,
            .ray_query_proceed_wide = ray_query_packet_probe_wide,
        },
        .ray = {
            .compressed_origin = {1.0f, 2.0f, 3.0f},
            .compressed_t_min = 0.25f,
            .compressed_direction = {4.0f, 5.0f, 6.0f},
            .compressed_t_max = 7.0f,
        },
        .output = {values.data(), sizeof(values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.calls == 1u);
    CHECK(probe.lane_count == width);
    CHECK(probe.mask == probe.expected_mask);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        auto expected = lane < active_lanes ?
                            luisa::make_uint4(
                                10u + lane, 20u + lane,
                                static_cast<uint32_t>(
                                    SIMDHostRayQueryCandidateKind::surface),
                                lane) :
                            luisa::make_uint4(0xdeadbeefu);
        CHECK(luisa::all(values[lane] == expected));
    }
    return true;
}

[[nodiscard]] bool run_ray_query_packet_codegen() {
    return run_ray_query_packet_codegen_case(4u, 3u, false) &&
           run_ray_query_packet_codegen_case(8u, 5u, true);
}

struct RayQueryScratchProbe {
    uint32_t calls{0u};
    uint64_t mask{0u};
    bool valid{true};
};

void ray_query_scratch_probe(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    SIMDHostRayQueryState *first_state = nullptr;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if (states[lane] != nullptr) {
            first_state = states[lane];
            break;
        }
    }
    if (first_state == nullptr) { return; }
    auto *probe = static_cast<RayQueryScratchProbe *>(first_state->accel);
    probe->calls++;
    probe->mask |= active_mask_bits;
    probe->valid &= lane_count == 8u && active_mask_bits != 0u;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        auto active = (active_mask_bits & (uint64_t{1u} << lane)) != 0u;
        if (!active) {
            probe->valid &= states[lane] == nullptr;
            continue;
        }
        auto *state = states[lane];
        probe->valid &= state != nullptr && state->accel == probe &&
                        state->proceed == ray_query_scratch_probe &&
                        (state->visibility_mask == 0x31u ||
                         state->visibility_mask == 0x72u);
        state->committed = SIMDHostRayQueryCommittedHit{
            .inst = state->visibility_mask,
            .prim = lane,
            .bary = {0.0f, 0.0f},
            .kind = static_cast<uint32_t>(
                SIMDHostRayQueryCandidateKind::surface),
            .t = 1.0f,
        };
        state->terminated = 1u;
    }
}

[[nodiscard]] bool run_ray_query_scratch_coloring_codegen(
    bool overlapping, bool disable_coloring = false) {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    ScopedEnvironmentVariable coloring{
        "LUISA_SIMD_DISABLE_RAY_QUERY_SCRATCH_COLORING",
        disable_coloring ? "1" : nullptr};
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name(
        overlapping ? "ray_query_scratch_overlap" :
                      "ray_query_scratch_sequential");
    auto *accel = kernel->create_resource_argument(Type::of<Accel>());
    auto *ray = kernel->create_value_argument(Type::of<Ray>());
    auto *output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::uint4>()));
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto visibility1_value = uint32_t{0x31u};
    auto visibility2_value = uint32_t{0x72u};
    auto *visibility1 = module.create_constant(
        Type::of<uint32_t>(), &visibility1_value);
    auto *visibility2 = module.create_constant(
        Type::of<uint32_t>(), &visibility2_value);
    auto *query_type = Type::custom("LC_RayQueryAll");
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *query_value1 = builder.call(
        query_type, xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
        {accel, ray, visibility1});
    auto *query1 = builder.alloca_local(query_type);
    builder.store(query1, query_value1);
    xir::Value *committed1 = nullptr;
    if (!overlapping) {
        builder.call(
            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
            {query1});
        committed1 = builder.call(
            Type::of<CommittedHit>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
            {query1});
    }
    auto *query_value2 = builder.call(
        query_type, xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
        {accel, ray, visibility2});
    auto *query2 = builder.alloca_local(query_type);
    builder.store(query2, query_value2);
    if (overlapping) {
        builder.call(
            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
            {query1});
        committed1 = builder.call(
            Type::of<CommittedHit>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
            {query1});
    }
    builder.call(
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
        {query2});
    auto *committed2 = builder.call(
        Type::of<CommittedHit>(),
        xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
        {query2});
    auto *inst1 = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {committed1, zero});
    auto *inst2 = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {committed2, zero});
    auto *metadata = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::AGGREGATE,
        {inst1, inst2, x, one});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, x, metadata});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        overlapping ? "simd-ray-query-scratch-overlap" :
                      "simd-ray-query-scratch-sequential",
        *context);
    auto name = std::string{
        overlapping ? "simd_ray_query_scratch_overlap" :
                      "simd_ray_query_scratch_sequential"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    auto expected_slots = overlapping || disable_coloring ? 2u : 1u;
    CHECK(codegen.ray_query_count == 2u);
    CHECK(codegen.ray_query_scratch_slot_count == expected_slots);
    CHECK(codegen.ray_query_scratch_bytes ==
          expected_slots * width * sizeof(SIMDHostRayQueryState));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(count_occurrences(ir, "alloca [9728 x i8]") ==
          expected_slots);
    CHECK(count_occurrences(ir, "call void %") == 2u);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        Ray ray;
        SIMDHostBufferView output;
    };
    RayQueryScratchProbe probe;
    std::array<luisa::uint4, width> values{};
    values.fill(luisa::make_uint4(0xdeadbeefu));
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .ray_query_proceed = ray_query_scratch_probe,
            .ray_query_proceed_wide = ray_query_scratch_probe,
        },
        .ray = {
            .compressed_origin = {1.0f, 2.0f, 3.0f},
            .compressed_t_min = 0.25f,
            .compressed_direction = {4.0f, 5.0f, 6.0f},
            .compressed_t_max = 7.0f,
        },
        .output = {values.data(), sizeof(values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.calls == 2u);
    CHECK(probe.mask == 0x1fu);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        auto expected = lane < active_lanes ?
                            luisa::make_uint4(
                                visibility1_value,
                                visibility2_value, lane, 1u) :
                            luisa::make_uint4(0xdeadbeefu);
        CHECK(luisa::all(values[lane] == expected));
    }
    return true;
}

[[nodiscard]] bool run_sequential_ray_query_scratch_coloring_codegen() {
    return run_ray_query_scratch_coloring_codegen(false);
}

[[nodiscard]] bool run_overlapping_ray_query_scratch_coloring_codegen() {
    return run_ray_query_scratch_coloring_codegen(true);
}

[[nodiscard]] bool run_disabled_ray_query_scratch_coloring_codegen() {
    return run_ray_query_scratch_coloring_codegen(false, true);
}

[[nodiscard]] bool run_divergent_ray_query_scratch_coloring_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    ScopedEnvironmentVariable coloring{
        "LUISA_SIMD_DISABLE_RAY_QUERY_SCRATCH_COLORING", nullptr};
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("ray_query_scratch_divergent");
    auto *accel = kernel->create_resource_argument(Type::of<Accel>());
    auto *ray = kernel->create_value_argument(Type::of<Ray>());
    auto *output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::uint4>()));
    auto *entry = kernel->create_body_block();
    auto *left = kernel->create_basic_block();
    auto *right = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto two_value = uint32_t{2u};
    auto visibility1_value = uint32_t{0x31u};
    auto visibility2_value = uint32_t{0x72u};
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *visibility1 = module.create_constant(
        Type::of<uint32_t>(), &visibility1_value);
    auto *visibility2 = module.create_constant(
        Type::of<uint32_t>(), &visibility2_value);
    auto *query_type = Type::custom("LC_RayQueryAll");
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {x, two});
    builder.cond_br(condition, left, right);

    auto emit_query = [&](xir::BasicBlock *block,
                          xir::Value *visibility) {
        builder.set_insertion_point(block);
        auto *query_value = builder.call(
            query_type,
            xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
            {accel, ray, visibility});
        auto *query = builder.alloca_local(query_type);
        builder.store(query, query_value);
        builder.call(
            xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
            {query});
        auto *committed = builder.call(
            Type::of<CommittedHit>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT,
            {query});
        auto *inst = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
            {committed, zero});
        builder.br(merge);
        return inst;
    };
    auto *inst1 = emit_query(left, visibility1);
    auto *inst2 = emit_query(right, visibility2);
    builder.set_insertion_point(merge);
    auto *inst = builder.phi(
        Type::of<uint32_t>(), {{inst1, left}, {inst2, right}});
    auto *metadata = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::AGGREGATE,
        {inst, x, one, zero});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, x, metadata});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-ray-query-scratch-divergent", *context);
    auto name = std::string{"simd_ray_query_scratch_divergent"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.ray_query_count == 2u);
    CHECK(codegen.ray_query_scratch_slot_count == 1u);
    CHECK(codegen.ray_query_scratch_bytes ==
          width * sizeof(SIMDHostRayQueryState));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(count_occurrences(ir, "alloca [9728 x i8]") == 1u);
    CHECK(count_occurrences(ir, "call void %") == 2u);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        Ray ray;
        SIMDHostBufferView output;
    };
    RayQueryScratchProbe probe;
    std::array<luisa::uint4, width> values{};
    values.fill(luisa::make_uint4(0xdeadbeefu));
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .ray_query_proceed = ray_query_scratch_probe,
            .ray_query_proceed_wide = ray_query_scratch_probe,
        },
        .ray = {
            .compressed_origin = {1.0f, 2.0f, 3.0f},
            .compressed_t_min = 0.25f,
            .compressed_direction = {4.0f, 5.0f, 6.0f},
            .compressed_t_max = 7.0f,
        },
        .output = {values.data(), sizeof(values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.calls == 2u);
    CHECK(probe.mask == 0x1fu);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        auto expected = lane < active_lanes ?
                            luisa::make_uint4(
                                lane < two_value ?
                                    visibility1_value :
                                    visibility2_value,
                                lane, 1u, 0u) :
                            luisa::make_uint4(0xdeadbeefu);
        CHECK(luisa::all(values[lane] == expected));
    }
    return true;
}

[[nodiscard]] bool run_accel_motion_metadata_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("accel_motion_metadata");
    auto *accel = kernel->create_resource_argument(Type::of<Accel>());
    auto *matrix_output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::float4x4>()));
    auto *srt_type = Type::of<MotionInstanceTransformSRT>();
    auto *srt_output = kernel->create_resource_argument(
        Type::buffer(srt_type));
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto eight_value = uint32_t{8u};
    auto *eight = module.create_constant(
        Type::of<uint32_t>(), &eight_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *keyframe = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {x, one});
    auto *srt_instance = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {x, eight});
    auto *matrix = builder.call(
        Type::of<luisa::float4x4>(),
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX,
        {accel, x, keyframe});
    auto *srt = builder.call(
        srt_type,
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT,
        {accel, srt_instance, keyframe});
    auto *uniform_matrix = builder.call(
        Type::of<luisa::float4x4>(),
        xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX,
        {accel, zero, zero});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {matrix_output, x, matrix});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {srt_output, x, srt});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX,
        {accel, x, keyframe, matrix});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT,
        {accel, srt_instance, keyframe, srt});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX,
        {accel, zero, zero, uniform_matrix});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-accel-motion-metadata", *context);
    auto name = std::string{"simd_accel_motion_metadata"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        SIMDHostBufferView matrix_output;
        SIMDHostBufferView srt_output;
    };
    CHECK(codegen.argument_buffer_size == sizeof(Arguments));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("accel.safe.motion.keyframe") != std::string::npos);
    CHECK(ir.find("accel.motion.scalar.frame") != std::string::npos);
    CHECK(count_occurrences(ir, "llvm.masked.gather") >= 32u);
    CHECK(count_occurrences(ir, "llvm.masked.scatter") >= 32u);
    CHECK(ir.find("call void %") == std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);

    std::array<SIMDHostAccelInstance, 16u> instances{};
    std::array<std::array<MotionInstanceTransform, 2u>, 16u> frames{};
    for (auto instance = uint32_t{0u}; instance < 8u; instance++) {
        for (auto key = uint32_t{0u}; key < 2u; key++) {
            auto matrix = luisa::make_float4x4(1.0f);
            matrix[3u].x = 1.0f + static_cast<float>(instance);
            matrix[3u].y = 2.0f + static_cast<float>(key);
            frames[instance][key].as_matrix() = matrix;
            auto &srt = frames[instance + 8u][key].as_srt();
            srt.pivot[0u] = static_cast<float>(instance) + 0.25f;
            srt.pivot[1u] = static_cast<float>(key) + 0.5f;
            srt.translation[2u] =
                static_cast<float>(10u * instance + key);
        }
        instances[instance].motion_frames = frames[instance].data();
        instances[instance].motion_keyframe_count = 2u;
        instances[instance].motion_mode = static_cast<uint32_t>(
            SIMDHostAccelMotionMode::matrix);
        instances[instance + 8u].motion_frames =
            frames[instance + 8u].data();
        instances[instance + 8u].motion_keyframe_count = 2u;
        instances[instance + 8u].motion_mode = static_cast<uint32_t>(
            SIMDHostAccelMotionMode::srt);
    }
    SIMDHostAccelInstanceTable instance_table{
        .data = instances.data(),
        .size = instances.size(),
    };
    std::array<luisa::float4x4, width> matrix_values{};
    std::array<MotionInstanceTransformSRT, width> srt_values{};
    Arguments arguments{
        .accel = {
            .instances = &instance_table,
        },
        .matrix_output = {matrix_values.data(), sizeof(matrix_values)},
        .srt_output = {srt_values.data(), sizeof(srt_values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        auto key = lane & 1u;
        auto &expected_matrix = frames[lane][key].as_matrix();
        for (auto column = uint32_t{0u}; column < 4u; column++) {
            CHECK(luisa::all(
                matrix_values[lane][column] ==
                expected_matrix[column]));
        }
        CHECK(std::memcmp(
                  &srt_values[lane],
                  &frames[lane + 8u][key].as_srt(),
                  sizeof(MotionInstanceTransformSRT)) == 0);
        CHECK(instances[lane].dirty == 1u);
        CHECK(instances[lane + 8u].dirty == 1u);
    }
    for (auto lane = active_lanes; lane < 8u; lane++) {
        CHECK(instances[lane].dirty == 0u);
        CHECK(instances[lane + 8u].dirty == 0u);
    }
    return true;
}

struct BindlessTexturePacketProbe {
    bool valid{true};
    uint32_t calls{0u};
    std::array<uint64_t, 2u> masks{};
    std::array<std::array<uint32_t, 8u>, 2u> slots{};
};

void bindless_texture_sample_probe(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *sampler_codes,
    const float *u, const float *v, const float *w,
    const float *levels, float *values) {
    auto *probe = static_cast<BindlessTexturePacketProbe *>(
        slots[0u].texture2d.texture);
    auto call = probe->calls++;
    probe->valid &= call < probe->masks.size() && slot_count == 2u &&
                    dimension == 2u && lane_count == 8u &&
                    sampler_codes == nullptr && levels == nullptr &&
                    u != nullptr && v != nullptr && w != nullptr;
    if (call >= probe->masks.size()) { return; }
    probe->masks[call] = active_mask_bits;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if ((active_mask_bits & (uint64_t{1u} << lane)) == 0u) {
            continue;
        }
        probe->slots[call][lane] = slot_indices[lane];
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            values[component * lane_count + lane] =
                static_cast<float>(
                    call * 100u + component * 10u + slot_indices[lane]);
        }
    }
}

[[nodiscard]] bool run_bindless_texture_packet_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("bindless_texture_packet");
    auto *bindless = kernel->create_resource_argument(
        Type::of<BindlessArray>());
    auto *output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::float4>()));
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto uv_value = luisa::make_float2(0.25f, 0.75f);
    auto *uv = module.create_constant(
        Type::of<luisa::float2>(), &uv_value);
    auto output_offset_value = uint32_t{8u};
    auto *output_offset = module.create_constant(
        Type::of<uint32_t>(), &output_offset_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *slot = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {x, one});
    auto *varying_pixel = builder.call(
        Type::of<luisa::float4>(),
        xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE,
        {bindless, slot, uv});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, x, varying_pixel});
    auto *uniform_pixel = builder.call(
        Type::of<luisa::float4>(),
        xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE,
        {bindless, zero, uv});
    auto *uniform_output_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {x, output_offset});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, uniform_output_index, uniform_pixel});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-bindless-texture-packet", *context);
    auto name = std::string{"simd_bindless_texture_packet"};
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    struct alignas(16) Arguments {
        SIMDHostBindlessArrayView bindless;
        SIMDHostBufferView output;
    };
    CHECK(codegen.argument_buffer_size == sizeof(Arguments));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("bindless.texture.sample.result") != std::string::npos);
    CHECK(ir.find("bindless.texture.sample.lane") == std::string::npos);
    CHECK(ir.find("bindless.uniform.callback.mask") != std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);

    BindlessTexturePacketProbe probe;
    std::array<SIMDHostBindlessSlot, 2u> slots{};
    for (auto &slot_descriptor : slots) {
        slot_descriptor.texture2d.texture = &probe;
    }
    std::array<luisa::float4, 16u> output_values{};
    Arguments arguments{
        .bindless = {
            .slots = slots.data(),
            .size = slots.size(),
            .sample_texture = bindless_texture_sample_probe,
        },
        .output = {output_values.data(), sizeof(output_values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.calls == 2u);
    CHECK(probe.masks[0u] == 0x1fu);
    CHECK(probe.masks[1u] == 0x01u);
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        auto expected_slot = lane & 1u;
        CHECK(probe.slots[0u][lane] == expected_slot);
        CHECK(luisa::all(output_values[lane] == luisa::make_float4(
                                                    static_cast<float>(expected_slot),
                                                    static_cast<float>(10u + expected_slot),
                                                    static_cast<float>(20u + expected_slot),
                                                    static_cast<float>(30u + expected_slot))));
        CHECK(luisa::all(output_values[8u + lane] ==
                         luisa::make_float4(
                             100.0f, 110.0f, 120.0f, 130.0f)));
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

[[nodiscard]] bool run_ast_uniform_loop_buffer_broadcast_width(
    uint32_t width, uint64_t expected_broadcast_count) {
    static constexpr auto count = 13u;
    static constexpr auto input_count = 16u;
    Kernel1D kernel = [](BufferUInt input, BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint sum = 0u;
        $for (other, input_count) {
            $if (other != index) {
                sum += input.read(other);
            };
        };
        output.write(index, sum);
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_uniform_loop_buffer_broadcast_w" +
            std::to_string(width));
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.uniform_buffer_broadcast_count ==
          expected_broadcast_count);
    if (width == 8u) {
        ScopedEnvironmentVariable disable{
            "LUISA_SIMD_DISABLE_UNIFORM_BUFFER_BROADCAST", "1"};
        auto gathered = compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_uniform_loop_buffer_gather_w8");
        CHECK(gathered.succeeded());
        CHECK(gathered.uniform_buffer_broadcast_count == 0u);
    }

    std::array<uint32_t, input_count> input{};
    std::array<uint32_t, count> output{};
    auto total = uint32_t{0u};
    for (auto i = uint32_t{0u}; i < input_count; i++) {
        input[i] = i + 1u;
        total += input[i];
    }
    output.fill(0xdeadbeefu);
    alignas(16) std::array<SIMDHostBufferView, 2u> arguments{
        SIMDHostBufferView{input.data(), sizeof(input)},
        SIMDHostBufferView{output.data(), sizeof(output)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(arguments.data(), nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        CHECK(output[i] == total - input[i]);
    }
    return true;
}

[[nodiscard]] bool run_ast_uniform_loop_buffer_broadcast() {
    return run_ast_uniform_loop_buffer_broadcast_width(1u, 1u) &&
           run_ast_uniform_loop_buffer_broadcast_width(2u, 0u) &&
           run_ast_uniform_loop_buffer_broadcast_width(4u, 1u) &&
           run_ast_uniform_loop_buffer_broadcast_width(8u, 1u) &&
           run_ast_uniform_loop_buffer_broadcast_width(16u, 1u);
}

[[nodiscard]] bool run_ast_coherent_loop_direct_control() {
    static constexpr auto count = 13u;
    static constexpr auto input_count = 16u;
    Kernel1D kernel = [](BufferUInt input, BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint sum = 0u;
        $for (inner, input_count) {
            sum += input.read(inner) * (index + 1u);
        };
        output.write(index, sum);
    };
    for (auto width : {1u, 2u, 4u, 8u, 16u}) {
        auto compiled = compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_coherent_loop_direct_w" +
                std::to_string(width),
            false, width == 8u);
        CHECK(compiled.succeeded());
        CHECK(compiled.direct_control_flow);
        if (width == 8u) {
            CHECK(!compiled.assembly.empty());
            CHECK(compiled.assembly.find(".LJTI") ==
                  std::string::npos);
            CHECK(compiled.assembly.find("jmpq\t*") ==
                  std::string::npos);
        }

        std::array<uint32_t, input_count> input{};
        std::array<uint32_t, count> output{};
        auto total = uint32_t{0u};
        for (auto i = uint32_t{0u}; i < input_count; i++) {
            input[i] = i + 1u;
            total += input[i];
        }
        output.fill(0xdeadbeefu);
        alignas(16) std::array<SIMDHostBufferView, 2u> arguments{
            SIMDHostBufferView{input.data(), sizeof(input)},
            SIMDHostBufferView{output.data(), sizeof(output)},
        };
        using Entry = void(
            const void *, void *, const SIMDPacketLaunchConfig *,
            uint32_t);
        auto entry = reinterpret_cast<Entry *>(compiled.entry);
        auto config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += width) {
            config.thread_index = first;
            entry(arguments.data(), nullptr, &config, width);
        }
        for (auto i = uint32_t{0u}; i < count; i++) {
            CHECK(output[i] == total * (i + 1u));
        }
    }
    {
        ScopedEnvironmentVariable disable{
            "LUISA_SIMD_DISABLE_COHERENT_DIRECT_CFG", "1"};
        auto scheduled = compile_simd_kernel(
            kernel.function()->function(), 8u,
            "simd_ast_coherent_loop_scheduled_w8");
        CHECK(scheduled.succeeded());
        CHECK(!scheduled.direct_control_flow);
    }
    {
        Kernel1D cohort_branch = [](BufferUInt output) noexcept {
            auto index = dispatch_id().x;
            auto enabled =
                warp_active_sum(warp_lane_id()) > 20u;
            $if (enabled) {
                output.write(index, index + 7u);
            }
            $else {
                output.write(index, index + 11u);
            };
        };
        auto compiled = compile_simd_kernel(
            cohort_branch.function()->function(), 8u,
            "simd_ast_cohort_branch_direct_w8");
        CHECK(compiled.succeeded());
        CHECK(compiled.direct_control_flow);

        std::array<uint32_t, count> output{};
        alignas(16) std::array<SIMDHostBufferView, 1u> arguments{
            SIMDHostBufferView{output.data(), sizeof(output)},
        };
        using Entry = void(
            const void *, void *, const SIMDPacketLaunchConfig *,
            uint32_t);
        auto entry = reinterpret_cast<Entry *>(compiled.entry);
        auto config = launch_1d(count, 16u);
        output.fill(0xdeadbeefu);
        for (auto first = uint32_t{0u}; first < 16u;
             first += 8u) {
            config.thread_index = first;
            entry(arguments.data(), nullptr, &config, 8u);
        }
        for (auto i = uint32_t{0u}; i < count; i++) {
            auto expected_offset = i < 8u ? 7u : 11u;
            CHECK(output[i] == i + expected_offset);
        }
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

[[nodiscard]] bool run_ast_predicated_diamond() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D varying_kernel = [](BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0u;
        $if ((index & 1u) == 0u) {
            value = index * 3u + 1u;
        }
        $else {
            value = index * 5u + 1u;
        };
        output.write(index, value);
    };
    auto varying = compile_simd_kernel(
        varying_kernel.function()->function(), width,
        "simd_ast_predicated_diamond");
    if (!varying.succeeded()) {
        for (auto &&diagnostic : varying.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(varying.predicated_diamond_count == 1u);
    CHECK(varying.predicated_instruction_count == 4u);
    CHECK(varying.predicated_phi_count == 1u);
    CHECK(varying.factored_select_count == 2u);

    auto scalar = compile_simd_kernel(
        varying_kernel.function()->function(), 1u,
        "simd_ast_scalar_diamond");
    CHECK(scalar.succeeded());
    CHECK(scalar.predicated_diamond_count == 0u);
    CHECK(scalar.predicated_instruction_count == 0u);
    CHECK(scalar.predicated_phi_count == 0u);
    CHECK(scalar.factored_select_count == 0u);

    {
        ScopedEnvironmentVariable disable_predication{
            "LUISA_SIMD_DISABLE_PREDICATED_IF", "1"};
        auto scheduled = compile_simd_kernel(
            varying_kernel.function()->function(), width,
            "simd_ast_scheduled_diamond");
        CHECK(scheduled.succeeded());
        CHECK(scheduled.predicated_diamond_count == 0u);
        CHECK(scheduled.predicated_instruction_count == 0u);
        CHECK(scheduled.predicated_phi_count == 0u);
        CHECK(scheduled.factored_select_count == 0u);
    }

    std::array<uint32_t, count> output{};
    output.fill(0xdeadbeefu);
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(varying.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(&argument, nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto expected = (i & 1u) == 0u ?
                            i * 3u + 1u :
                            i * 5u + 1u;
        CHECK(output[i] == expected);
    }

    Kernel1D total_cast_kernel = [](BufferFloat output) noexcept {
        auto index = dispatch_id().x;
        auto signed_index = cast<int>(index);
        $float value = 0.0f;
        $if ((index & 1u) == 0u) {
            value = cast<float>(signed_index) + 1.0f;
        }
        $else {
            value = cast<float>(-signed_index) + 2.0f;
        };
        output.write(index, value);
    };
    auto total_cast = compile_simd_kernel(
        total_cast_kernel.function()->function(), width,
        "simd_ast_predicated_total_cast_diamond");
    CHECK(total_cast.succeeded());
    CHECK(total_cast.predicated_diamond_count == 1u);
    CHECK(total_cast.predicated_phi_count == 1u);
    std::array<float, count> cast_output{};
    cast_output.fill(-1234.0f);
    alignas(16) SIMDHostBufferView cast_argument{
        cast_output.data(), sizeof(cast_output)};
    auto cast_entry = reinterpret_cast<Entry *>(total_cast.entry);
    CHECK(cast_entry != nullptr);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        cast_entry(&cast_argument, nullptr, &config, width);
    }
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto expected = (i & 1u) == 0u ?
                            static_cast<float>(i) + 1.0f :
                            -static_cast<float>(i) + 2.0f;
        CHECK(cast_output[i] == expected);
    }

    Kernel1D uniform_kernel = [](BufferUInt output,
                                 UInt selector) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0u;
        $if (selector == 0u) {
            value = index * 3u + 1u;
        }
        $else {
            value = index * 5u + 7u;
        };
        output.write(index, value);
    };
    auto uniform = compile_simd_kernel(
        uniform_kernel.function()->function(), width,
        "simd_ast_uniform_diamond");
    CHECK(uniform.succeeded());
    CHECK(uniform.predicated_diamond_count == 0u);
    CHECK(uniform.predicated_instruction_count == 0u);
    CHECK(uniform.predicated_phi_count == 0u);
    CHECK(uniform.factored_select_count == 0u);

    xir::Module cohort_module;
    auto *cohort_kernel = cohort_module.create_kernel();
    auto *cohort_entry = cohort_kernel->create_body_block();
    auto *cohort_true = cohort_kernel->create_basic_block();
    auto *cohort_false = cohort_kernel->create_basic_block();
    auto *cohort_merge = cohort_kernel->create_basic_block();
    auto *lane = cohort_module.create_warp_lane_id();
    auto *zero = cohort_module.create_constant_zero(
        Type::of<uint32_t>());
    auto *one = cohort_module.create_constant_one(
        Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(cohort_entry);
    auto *lane_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
        {lane, zero});
    auto *cohort_condition = builder.call(
        Type::of<bool>(),
        xir::ThreadGroupOp::WARP_ACTIVE_ANY,
        {lane_condition});
    builder.cond_br(
        cohort_condition, cohort_true, cohort_false);
    builder.set_insertion_point(cohort_true);
    builder.br(cohort_merge);
    builder.set_insertion_point(cohort_false);
    builder.br(cohort_merge);
    builder.set_insertion_point(cohort_merge);
    static_cast<void>(builder.phi(
        Type::of<uint32_t>(),
        {{one, cohort_true}, {zero, cohort_false}}));
    builder.return_void();
    schedule::WarpUniformityAnalysis cohort_uniformity;
    cohort_uniformity.analyze(cohort_kernel);
    CHECK(cohort_uniformity.classify(cohort_condition) ==
          schedule::ValueClass::cohort_uniform);
    auto cohort =
        schedule::predicate_small_varying_diamonds(cohort_kernel);
    CHECK(!cohort.changed());
    CHECK(cohort_entry->terminator()->isa<xir::ConditionalBranchInst>());
    return true;
}

[[nodiscard]] bool run_ast_loop_unswitch() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D varying_kernel = [](BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        auto choose_left = (index & 1u) == 0u;
        $uint value = 1u;
        $for (iteration, 8u) {
            $if (choose_left) {
                value = value * 3u + index;
                value = value ^ (iteration + 9u);
            }
            $else {
                value = value * 5u + index;
                value = value ^ (iteration + 17u);
            };
        };
        output.write(index, value);
    };
    auto varying = compile_simd_kernel(
        varying_kernel.function()->function(), width,
        "simd_ast_loop_unswitch");
    if (!varying.succeeded()) {
        for (auto &&diagnostic : varying.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(varying.predicated_diamond_count == 0u);
    CHECK(varying.unswitched_loop_count == 1u);
    CHECK(varying.unswitched_cloned_block_count != 0u);
    CHECK(varying.unswitched_cloned_instruction_count != 0u);
    CHECK(varying.unswitched_live_out_count != 0u);

    {
        ScopedEnvironmentVariable disable_loop_unswitch{
            "LUISA_SIMD_DISABLE_LOOP_UNSWITCH", "1"};
        auto scheduled = compile_simd_kernel(
            varying_kernel.function()->function(), width,
            "simd_ast_scheduled_loop");
        CHECK(scheduled.succeeded());
        CHECK(scheduled.unswitched_loop_count == 0u);
        CHECK(scheduled.unswitched_cloned_block_count == 0u);
        CHECK(scheduled.unswitched_cloned_instruction_count == 0u);
        CHECK(scheduled.unswitched_live_out_count == 0u);
    }

    std::array<uint32_t, count> output{};
    output.fill(0xdeadbeefu);
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(varying.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(&argument, nullptr, &config, width);
    }
    for (auto index = uint32_t{0u}; index < count; index++) {
        auto value = uint32_t{1u};
        auto choose_left = (index & 1u) == 0u;
        for (auto iteration = uint32_t{0u}; iteration < 8u;
             iteration++) {
            if (choose_left) {
                value = value * 3u + index;
                value = value ^ (iteration + 9u);
            } else {
                value = value * 5u + index;
                value = value ^ (iteration + 17u);
            }
        }
        CHECK(output[index] == value);
    }
    return true;
}

[[nodiscard]] bool run_ast_shader_execution_reorder() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        reorder_shader_execution(index & 3u, 2u);
        output.write(index, index * 7u + 3u);
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_shader_execution_reorder");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    std::array<uint32_t, count> output{};
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(&argument, nullptr, &config, width);
    }
    for (auto index = uint32_t{0u}; index < count; index++) {
        CHECK(output[index] == index * 7u + 3u);
    }
    return true;
}

[[nodiscard]] bool run_ast_boolean_vector_reduction() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferUInt2 output) noexcept {
        auto index = dispatch_id().x;
        auto bits = make_bool3(
            (index & 1u) != 0u,
            (index & 2u) != 0u,
            (index & 4u) != 0u);
        output.write(
            index,
            make_uint2(cast<uint>(any(bits)), cast<uint>(all(bits))));
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_boolean_vector_reduction");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    std::array<uint2, count> output{};
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(&argument, nullptr, &config, width);
    }
    for (auto index = uint32_t{0u}; index < count; index++) {
        CHECK(output[index].x == ((index & 7u) != 0u));
        CHECK(output[index].y == ((index & 7u) == 7u));
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
        {"XIR faceforward fixed-vector arithmetic",
         &run_faceforward_codegen},
        {"lane-affine scalar buffer load/store",
         &run_lane_affine_buffer_codegen},
        {"uniform buffer read broadcast",
         &run_uniform_buffer_broadcast_codegen},
        {"XIR texture packet callback", &run_texture_packet_codegen},
        {"XIR accel instance metadata",
         &run_accel_instance_metadata_codegen},
        {"XIR ray-query packet callback",
         &run_ray_query_packet_codegen},
        {"sequential ray-query scratch coloring",
         &run_sequential_ray_query_scratch_coloring_codegen},
        {"overlapping ray-query scratch coloring",
         &run_overlapping_ray_query_scratch_coloring_codegen},
        {"disabled ray-query scratch coloring",
         &run_disabled_ray_query_scratch_coloring_codegen},
        {"divergent ray-query scratch coloring",
         &run_divergent_ray_query_scratch_coloring_codegen},
        {"XIR accel motion metadata",
         &run_accel_motion_metadata_codegen},
        {"XIR bindless texture packet callback",
         &run_bindless_texture_packet_codegen},
        {"AST buffer dispatch", &run_ast_buffer_codegen},
        {"AST uniform-loop buffer broadcast",
         &run_ast_uniform_loop_buffer_broadcast},
        {"AST coherent-loop direct control",
         &run_ast_coherent_loop_direct_control},
        {"AST select operand order", &run_ast_select_codegen},
        {"AST fast radix pow canonicalization",
         &run_ast_fast_math_canonicalization},
        {"AST predicated varying diamond",
         &run_ast_predicated_diamond},
        {"AST invariant varying loop unswitch",
         &run_ast_loop_unswitch},
        {"AST shader execution reorder hint",
         &run_ast_shader_execution_reorder},
        {"AST boolean-vector all/any",
         &run_ast_boolean_vector_reduction},
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
