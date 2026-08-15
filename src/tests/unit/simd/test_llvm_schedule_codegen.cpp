#include "llvm_schedule_codegen.h"
#include "llvm_jit.h"
#include "predicated_if_conversion.h"
#include "simd_compiler.h"
#include "warp_uniformity.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <csignal>
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

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "xir_to_schedule.h"

using namespace luisa::compute;
using namespace luisa::compute::simd;

struct SIMDAggregatePromotionProbe {
    float x;
    float y;
    float z;
    uint32_t tag;
};

LUISA_STRUCT(SIMDAggregatePromotionProbe, x, y, z, tag) {};

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

[[nodiscard]] std::string_view line_containing(
    std::string_view text, std::string_view needle) noexcept {
    auto position = text.find(needle);
    if (position == std::string_view::npos) { return {}; }
    auto begin = text.rfind('\n', position);
    begin = begin == std::string_view::npos ? 0u : begin + 1u;
    auto end = text.find('\n', position);
    if (end == std::string_view::npos) { end = text.size(); }
    return text.substr(begin, end - begin);
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
make_state_phi_coalescing(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("state_phi_coalescing");
    auto *entry = kernel->create_body_block();
    auto *first_true = kernel->create_basic_block();
    auto *first_false = kernel->create_basic_block();
    auto *first_merge = kernel->create_basic_block();
    auto *second_true = kernel->create_basic_block();
    auto *second_false = kernel->create_basic_block();
    auto *second_merge = kernel->create_basic_block();
    entry->set_name("entry");
    first_true->set_name("first_true");
    first_false->set_name("first_false");
    first_merge->set_name("first_merge");
    second_true->set_name("second_true");
    second_false->set_name("second_false");
    second_merge->set_name("second_merge");

    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    uint32_t ten_value = 10u;
    uint32_t hundred_value = 100u;
    uint32_t two_hundred_value = 200u;
    uint32_t thousand_value = 1000u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *ten = module.create_constant(
        Type::of<uint32_t>(), &ten_value);
    auto *hundred = module.create_constant(
        Type::of<uint32_t>(), &hundred_value);
    auto *two_hundred = module.create_constant(
        Type::of<uint32_t>(), &two_hundred_value);
    auto *thousand = module.create_constant(
        Type::of<uint32_t>(), &thousand_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *initial = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, ten});
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *first_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {parity, zero});
    builder.cond_br(first_condition, first_true, first_false);

    builder.set_insertion_point(first_true);
    auto *first_true_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {initial, hundred});
    auto *first_true_left = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {initial, ten});
    auto *first_true_right = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {initial, two_hundred});
    builder.br(first_merge);
    builder.set_insertion_point(first_false);
    auto *first_false_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {initial, two_hundred});
    auto *first_false_left = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {initial, hundred});
    auto *first_false_right = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {initial, thousand});
    builder.br(first_merge);

    builder.set_insertion_point(first_merge);
    auto *first_state = builder.phi(
        Type::of<uint32_t>(),
        {{first_true_value, first_true},
         {first_false_value, first_false}});
    first_state->set_name("state_chain");
    auto *first_left = builder.phi(
        Type::of<uint32_t>(),
        {{first_true_left, first_true},
         {first_false_left, first_false}});
    auto *first_right = builder.phi(
        Type::of<uint32_t>(),
        {{first_true_right, first_true},
         {first_false_right, first_false}});
    first_left->set_name("parallel_chain");
    first_right->set_name("parallel_chain");
    auto *second_bits = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, two});
    auto *second_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {second_bits, zero});
    builder.cond_br(second_condition, second_true, second_false);

    builder.set_insertion_point(second_true);
    auto *second_true_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {first_state, thousand});
    builder.br(second_merge);
    builder.set_insertion_point(second_false);
    builder.br(second_merge);

    builder.set_insertion_point(second_merge);
    auto *second_state = builder.phi(
        Type::of<uint32_t>(),
        {{second_true_value, second_true},
         {first_state, second_false}});
    second_state->set_name("state_chain");
    auto *second_left = builder.phi(
        Type::of<uint32_t>(),
        {{first_right, second_true},
         {first_left, second_false}});
    auto *second_right = builder.phi(
        Type::of<uint32_t>(),
        {{first_left, second_true},
         {first_right, second_false}});
    second_left->set_name("parallel_chain");
    second_right->set_name("parallel_chain");
    auto *scaled_state = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_MUL,
        {second_state, thousand});
    auto *result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {scaled_state, second_right});
    result->set_name("coalescing_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name != "second_merge") { continue; }
        for (auto &&value : lowered.function->values()) {
            if (value.defining_block == block.id &&
                value.name == "coalescing_result") {
                block.terminator = schedule::ReturnTerminator{value.id};
                break;
            }
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_general_state_coloring_pressure(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("general_state_coloring_pressure");
    auto *entry = kernel->create_body_block();
    auto *first_low = kernel->create_basic_block();
    auto *first_high = kernel->create_basic_block();
    auto *first_merge = kernel->create_basic_block();
    auto *first_consume = kernel->create_basic_block();
    auto *second_prepare = kernel->create_basic_block();
    auto *second_low = kernel->create_basic_block();
    auto *second_high = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("entry");
    first_low->set_name("first_low");
    first_high->set_name("first_high");
    first_merge->set_name("first_merge");
    first_consume->set_name("first_consume");
    second_prepare->set_name("second_prepare");
    second_low->set_name("second_low");
    second_high->set_name("second_high");
    exit->set_name("exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    xir::XIRBuilder builder;
    auto make_sum = [&](const std::array<xir::Value *, 16u> &values) noexcept {
        auto *sum = values.front();
        for (auto i = size_t{1u}; i < values.size(); i++) {
            sum = builder.call(
                Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
                {sum, values[i]});
        }
        return sum;
    };

    builder.set_insertion_point(entry);
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *first_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {parity, zero});
    builder.cond_br(first_condition, first_low, first_high);

    std::array<xir::Value *, 16u> first_low_values{};
    builder.set_insertion_point(first_low);
    for (auto i = size_t{0u}; i < first_low_values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 1u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        first_low_values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
    }
    builder.br(first_merge);

    std::array<xir::Value *, 16u> first_high_values{};
    builder.set_insertion_point(first_high);
    for (auto i = size_t{0u}; i < first_high_values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 101u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        first_high_values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
    }
    builder.br(first_merge);

    std::array<xir::Value *, 16u> first_states{};
    builder.set_insertion_point(first_merge);
    for (auto i = size_t{0u}; i < first_states.size(); i++) {
        first_states[i] = builder.phi(
            Type::of<uint32_t>(),
            {{first_low_values[i], first_low},
             {first_high_values[i], first_high}});
        first_states[i]->set_name(
            "first_state_" + std::to_string(i));
    }
    builder.br(first_consume);

    builder.set_insertion_point(first_consume);
    static_cast<void>(make_sum(first_states));
    builder.br(second_prepare);

    builder.set_insertion_point(second_prepare);
    auto *second_bits = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, two});
    auto *second_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {second_bits, zero});
    builder.cond_br(second_condition, second_low, second_high);

    std::array<xir::Value *, 16u> second_low_values{};
    builder.set_insertion_point(second_low);
    for (auto i = size_t{0u}; i < second_low_values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 201u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        second_low_values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
    }
    builder.br(exit);

    std::array<xir::Value *, 16u> second_high_values{};
    builder.set_insertion_point(second_high);
    for (auto i = size_t{0u}; i < second_high_values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 301u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        second_high_values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
    }
    builder.br(exit);

    std::array<xir::Value *, 16u> second_states{};
    builder.set_insertion_point(exit);
    for (auto i = size_t{0u}; i < second_states.size(); i++) {
        second_states[i] = builder.phi(
            Type::of<uint32_t>(),
            {{second_low_values[i], second_low},
             {second_high_values[i], second_high}});
        second_states[i]->set_name(
            "second_state_" + std::to_string(i));
    }
    auto *result = make_sum(second_states);
    result->set_name("general_coloring_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name != "exit") { continue; }
        for (auto &&value : lowered.function->values()) {
            if (value.defining_block == block.id &&
                value.name == "general_coloring_result") {
                block.terminator =
                    schedule::ReturnTerminator{value.id};
                break;
            }
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_single_general_state_coloring_candidate(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("single_general_state_coloring_candidate");
    auto *entry = kernel->create_body_block();
    auto *first_low = kernel->create_basic_block();
    auto *first_high = kernel->create_basic_block();
    auto *middle = kernel->create_basic_block();
    auto *first_consume = kernel->create_basic_block();
    auto *second_prepare = kernel->create_basic_block();
    auto *second_low = kernel->create_basic_block();
    auto *second_high = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("entry");
    first_low->set_name("first_low");
    first_high->set_name("first_high");
    middle->set_name("middle");
    first_consume->set_name("first_consume");
    second_prepare->set_name("second_prepare");
    second_low->set_name("second_low");
    second_high->set_name("second_high");
    exit->set_name("exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *first_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {parity, zero});
    builder.cond_br(first_condition, first_low, first_high);

    builder.set_insertion_point(first_low);
    auto *first_low_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, one});
    builder.br(middle);
    builder.set_insertion_point(first_high);
    auto *first_high_value = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, two});
    builder.br(middle);

    builder.set_insertion_point(middle);
    auto *first_state = builder.phi(
        Type::of<uint32_t>(),
        {{first_low_value, first_low},
         {first_high_value, first_high}});
    first_state->set_name("single_first_state");
    builder.br(first_consume);

    builder.set_insertion_point(first_consume);
    static_cast<void>(builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {first_state, one}));
    builder.br(second_prepare);

    builder.set_insertion_point(second_prepare);
    auto *second_bits = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, two});
    auto *second_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {second_bits, zero});
    builder.cond_br(second_condition, second_low, second_high);

    std::array<xir::Value *, 31u> second_low_values{};
    builder.set_insertion_point(second_low);
    for (auto i = size_t{0u}; i < second_low_values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 101u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        second_low_values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
    }
    builder.br(exit);

    std::array<xir::Value *, 31u> second_high_values{};
    builder.set_insertion_point(second_high);
    for (auto i = size_t{0u}; i < second_high_values.size(); i++) {
        auto constant_value = static_cast<uint32_t>(i + 201u);
        auto *constant = module.create_constant(
            Type::of<uint32_t>(), &constant_value);
        second_high_values[i] = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {lane, constant});
    }
    builder.br(exit);

    builder.set_insertion_point(exit);
    std::array<xir::Value *, 31u> second_states{};
    for (auto i = size_t{0u}; i < second_states.size(); i++) {
        second_states[i] = builder.phi(
            Type::of<uint32_t>(),
            {{second_low_values[i], second_low},
             {second_high_values[i], second_high}});
        second_states[i]->set_name(
            "single_second_state_" + std::to_string(i));
    }
    auto *result = second_states.front();
    for (auto i = size_t{1u}; i < second_states.size(); i++) {
        result = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
            {result, second_states[i]});
    }
    result->set_name("single_general_coloring_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name != "exit") { continue; }
        for (auto &&value : lowered.function->values()) {
            if (value.defining_block == block.id &&
                value.name == "single_general_coloring_result") {
                block.terminator =
                    schedule::ReturnTerminator{value.id};
                break;
            }
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
make_local_predicated_sqrt_loop(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("local_predicated_sqrt_loop");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *body = kernel->create_basic_block();
    auto *sqrt_arm = kernel->create_basic_block();
    auto *miss_arm = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *second_body = kernel->create_basic_block();
    auto *second_sqrt_arm = kernel->create_basic_block();
    auto *second_miss_arm = kernel->create_basic_block();
    auto *second_merge = kernel->create_basic_block();
    auto *nested_outer = kernel->create_basic_block();
    auto *nested_calc = kernel->create_basic_block();
    auto *inner_true = kernel->create_basic_block();
    auto *inner_false = kernel->create_basic_block();
    auto *inner_merge = kernel->create_basic_block();
    auto *outer_false = kernel->create_basic_block();
    auto *outer_merge = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("entry");
    header->set_name("header");
    body->set_name("body");
    sqrt_arm->set_name("sqrt_arm");
    miss_arm->set_name("miss_arm");
    merge->set_name("merge");
    second_body->set_name("second_body");
    second_sqrt_arm->set_name("second_sqrt_arm");
    second_miss_arm->set_name("second_miss_arm");
    second_merge->set_name("second_merge");
    nested_outer->set_name("nested_outer");
    nested_calc->set_name("nested_calc");
    inner_true->set_name("inner_true");
    inner_false->set_name("inner_false");
    inner_merge->set_name("inner_merge");
    outer_false->set_name("outer_false");
    outer_merge->set_name("outer_merge");
    exit->set_name("exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero_u32 = module.create_constant_zero(Type::of<uint32_t>());
    auto *one_u32 = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_u32_value = 2u;
    auto *two_u32 = module.create_constant(
        Type::of<uint32_t>(), &two_u32_value);
    float zero_f32_value = 0.0f;
    float one_f32_value = 1.0f;
    float two_f32_value = 2.0f;
    auto *zero_f32 = module.create_constant(
        Type::of<float>(), &zero_f32_value);
    auto *one_f32 = module.create_constant(
        Type::of<float>(), &one_f32_value);
    auto *two_f32 = module.create_constant(
        Type::of<float>(), &two_f32_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *lane_f32 = builder.cast_(
        Type::of<float>(), xir::CastOp::STATIC_CAST, lane);
    auto *initial = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {lane_f32, one_f32});
    builder.br(header);

    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    auto *value = builder.phi(Type::of<float>());
    value->set_name("local_region_result");
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one_u32});
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {parity, two_u32});
    auto *loop_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(loop_condition, body, exit);

    builder.set_insertion_point(body);
    auto *square = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {value, value});
    auto *discriminant = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_SUB,
        {square, one_f32});
    auto *hit = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
        {discriminant, zero_f32});
    builder.cond_br(hit, sqrt_arm, miss_arm);

    builder.set_insertion_point(sqrt_arm);
    auto *root = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SQRT,
        {discriminant});
    auto *negative = builder.call(
        Type::of<float>(), xir::ArithmeticOp::UNARY_MINUS,
        {root});
    auto *offset = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {value, negative});
    auto *scaled = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {offset, two_f32});
    builder.br(merge);

    builder.set_insertion_point(miss_arm);
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *first_value = builder.phi(
        Type::of<float>(),
        {{scaled, sqrt_arm}, {value, miss_arm}});
    builder.br(second_body);

    builder.set_insertion_point(second_body);
    auto *second_square = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {first_value, first_value});
    auto *second_discriminant = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_SUB,
        {second_square, one_f32});
    auto *second_hit = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
        {second_discriminant, zero_f32});
    builder.cond_br(
        second_hit, second_sqrt_arm, second_miss_arm);

    builder.set_insertion_point(second_sqrt_arm);
    auto *second_quotient = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_DIV,
        {second_discriminant, two_f32});
    auto *second_clamped = builder.call(
        Type::of<float>(), xir::ArithmeticOp::MAX,
        {second_quotient, one_f32});
    auto *second_offset = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {first_value, second_clamped});
    auto *second_scaled = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {second_offset, two_f32});
    builder.br(second_merge);

    builder.set_insertion_point(second_miss_arm);
    builder.br(second_merge);

    builder.set_insertion_point(second_merge);
    auto *second_value = builder.phi(
        Type::of<float>(),
        {{second_scaled, second_sqrt_arm},
         {first_value, second_miss_arm}});
    builder.br(nested_outer);

    builder.set_insertion_point(nested_outer);
    auto *outer_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
        {parity, zero_u32});
    builder.cond_br(outer_condition, nested_calc, outer_false);

    builder.set_insertion_point(nested_calc);
    auto *advanced = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {second_value, lane_f32});
    auto *inner_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
        {advanced, two_f32});
    builder.cond_br(inner_condition, inner_true, inner_false);

    builder.set_insertion_point(inner_true);
    builder.br(inner_merge);
    builder.set_insertion_point(inner_false);
    builder.br(inner_merge);

    builder.set_insertion_point(inner_merge);
    auto *inner_value = builder.phi(
        Type::of<float>(),
        {{advanced, inner_true}, {second_value, inner_false}});
    builder.br(outer_merge);

    builder.set_insertion_point(outer_false);
    builder.br(outer_merge);

    builder.set_insertion_point(outer_merge);
    auto *next_value = builder.phi(
        Type::of<float>(),
        {{inner_value, inner_merge}, {second_value, outer_false}});
    auto *next_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one_u32});
    builder.br(header);

    builder.set_insertion_point(exit);
    builder.return_void();
    index->add_incoming(zero_u32, entry);
    index->add_incoming(next_index, outer_merge);
    value->add_incoming(initial, entry);
    value->add_incoming(next_value, outer_merge);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    for (auto &&schedule_value : lowered.function->values()) {
        if (schedule_value.name == "local_region_result") {
            result_id = schedule_value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &schedule_block : lowered.function->blocks()) {
        if (schedule_block.name == "exit") {
            schedule_block.terminator =
                schedule::ReturnTerminator{result_id};
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_local_predicated_terminal_bridge_loop(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("local_predicated_terminal_bridge_loop");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *body = kernel->create_basic_block();
    auto *sqrt_arm = kernel->create_basic_block();
    auto *miss_arm = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("terminal_entry");
    header->set_name("terminal_header");
    body->set_name("terminal_body");
    sqrt_arm->set_name("terminal_sqrt_arm");
    miss_arm->set_name("terminal_miss_arm");
    merge->set_name("terminal_merge");
    exit->set_name("terminal_exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero_u32 = module.create_constant_zero(Type::of<uint32_t>());
    auto *one_u32 = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_u32_value = 2u;
    auto *two_u32 = module.create_constant(
        Type::of<uint32_t>(), &two_u32_value);
    float zero_f32_value = 0.0f;
    float half_f32_value = 0.5f;
    float one_f32_value = 1.0f;
    float two_f32_value = 2.0f;
    auto *zero_f32 = module.create_constant(
        Type::of<float>(), &zero_f32_value);
    auto *half_f32 = module.create_constant(
        Type::of<float>(), &half_f32_value);
    auto *one_f32 = module.create_constant(
        Type::of<float>(), &one_f32_value);
    auto *two_f32 = module.create_constant(
        Type::of<float>(), &two_f32_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *lane_f32 = builder.cast_(
        Type::of<float>(), xir::CastOp::STATIC_CAST, lane);
    auto *initial = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {lane_f32, one_f32});
    builder.br(header);

    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    auto *value = builder.phi(Type::of<float>());
    value->set_name("terminal_bridge_result");
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one_u32});
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {parity, two_u32});
    auto *loop_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(loop_condition, body, exit);

    builder.set_insertion_point(body);
    auto *selector = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, index});
    auto *selector_bit = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {selector, one_u32});
    auto *take_sqrt = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
        {selector_bit, zero_u32});
    builder.cond_br(take_sqrt, sqrt_arm, miss_arm);

    builder.set_insertion_point(sqrt_arm);
    auto *square = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {value, value});
    auto *discriminant = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {square, one_f32});
    auto *root = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SQRT,
        {discriminant});
    auto *scaled = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {root, half_f32});
    builder.br(merge);

    builder.set_insertion_point(miss_arm);
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *next_value = static_cast<xir::Value *>(builder.phi(
        Type::of<float>(),
        {{scaled, sqrt_arm}, {value, miss_arm}}));
    for (auto i = 0u; i < 20u; i++) {
        next_value = builder.call(
            Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
            {next_value, lane_f32});
        next_value = builder.call(
            Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
            {next_value, half_f32});
    }
    auto *next_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one_u32});
    builder.br(header);

    builder.set_insertion_point(exit);
    auto *positive = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
        {value, zero_f32});
    auto *result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SELECT,
        {value, two_f32, positive});
    result->set_name("terminal_bridge_return");
    builder.return_void();
    index->add_incoming(zero_u32, entry);
    index->add_incoming(next_index, merge);
    value->add_incoming(initial, entry);
    value->add_incoming(next_value, merge);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    for (auto &&schedule_value : lowered.function->values()) {
        if (schedule_value.name == "terminal_bridge_return") {
            result_id = schedule_value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &schedule_block : lowered.function->blocks()) {
        if (schedule_block.name == "terminal_exit") {
            schedule_block.terminator =
                schedule::ReturnTerminator{result_id};
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_two_sided_local_predicated_loop(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("two_sided_local_predicated_loop");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *body = kernel->create_basic_block();
    auto *true_arm = kernel->create_basic_block();
    auto *false_arm = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("two_sided_entry");
    header->set_name("two_sided_header");
    body->set_name("two_sided_body");
    true_arm->set_name("two_sided_true");
    false_arm->set_name("two_sided_false");
    merge->set_name("two_sided_merge");
    exit->set_name("two_sided_exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero_u32 = module.create_constant_zero(Type::of<uint32_t>());
    auto *one_u32 = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_u32_value = 2u;
    auto *two_u32 = module.create_constant(
        Type::of<uint32_t>(), &two_u32_value);
    float zero_f32_value = 0.0f;
    float half_f32_value = 0.5f;
    float one_f32_value = 1.0f;
    float nan_f32_value = std::bit_cast<float>(0x7fc01234u);
    auto *zero_f32 = module.create_constant(
        Type::of<float>(), &zero_f32_value);
    auto *half_f32 = module.create_constant(
        Type::of<float>(), &half_f32_value);
    auto *one_f32 = module.create_constant(
        Type::of<float>(), &one_f32_value);
    auto *nan_f32 = module.create_constant(
        Type::of<float>(), &nan_f32_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *lane_f32 = builder.cast_(
        Type::of<float>(), xir::CastOp::STATIC_CAST, lane);
    auto *initial = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {lane_f32, one_f32});
    builder.br(header);

    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    auto *value = builder.phi(Type::of<float>());
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one_u32});
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {parity, two_u32});
    auto *loop_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(loop_condition, body, exit);

    builder.set_insertion_point(body);
    auto *selector = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, index});
    auto *selector_bit = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {selector, one_u32});
    auto *take_true = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
        {selector_bit, zero_u32});
    auto *true_cast_source = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SELECT,
        {nan_f32, value, take_true});
    builder.cond_br(take_true, true_arm, false_arm);

    builder.set_insertion_point(true_arm);
    auto *integer_value = builder.cast_(
        Type::of<uint32_t>(), xir::CastOp::STATIC_CAST,
        true_cast_source);
    auto *rounded_value = builder.cast_(
        Type::of<float>(), xir::CastOp::STATIC_CAST,
        integer_value);
    auto *true_offset = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {rounded_value, one_f32});
    auto *true_value = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {true_offset, half_f32});
    builder.br(merge);

    builder.set_insertion_point(false_arm);
    auto *negative = builder.call(
        Type::of<float>(), xir::ArithmeticOp::UNARY_MINUS,
        {value});
    auto *square = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {negative, negative});
    auto *false_offset = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {square, one_f32});
    auto *false_value = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_MUL,
        {false_offset, half_f32});
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *selected_value = builder.phi(
        Type::of<float>(),
        {{true_value, true_arm}, {false_value, false_arm}});
    auto *next_value = builder.call(
        Type::of<float>(), xir::ArithmeticOp::BINARY_ADD,
        {selected_value, half_f32});
    auto *next_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one_u32});
    builder.br(header);

    builder.set_insertion_point(exit);
    auto *positive = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_GREATER,
        {value, zero_f32});
    auto *result = builder.call(
        Type::of<float>(), xir::ArithmeticOp::SELECT,
        {one_f32, value, positive});
    result->set_name("two_sided_return");
    builder.return_void();
    index->add_incoming(zero_u32, entry);
    index->add_incoming(next_index, merge);
    value->add_incoming(initial, entry);
    value->add_incoming(next_value, merge);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    for (auto &&schedule_value : lowered.function->values()) {
        if (schedule_value.name == "two_sided_return") {
            result_id = schedule_value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &schedule_block : lowered.function->blocks()) {
        if (schedule_block.name == "two_sided_exit") {
            schedule_block.terminator =
                schedule::ReturnTerminator{result_id};
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_nested_local_predicated_loop(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("nested_local_predicated_loop");
    auto *entry = kernel->create_body_block();
    auto *header = kernel->create_basic_block();
    auto *body = kernel->create_basic_block();
    auto *nested = kernel->create_basic_block();
    auto *inner_true = kernel->create_basic_block();
    auto *inner_false = kernel->create_basic_block();
    auto *inner_merge = kernel->create_basic_block();
    auto *outer_false = kernel->create_basic_block();
    auto *outer_merge = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("entry");
    header->set_name("header");
    body->set_name("body");
    nested->set_name("nested");
    inner_true->set_name("inner_true");
    inner_false->set_name("inner_false");
    inner_merge->set_name("inner_merge");
    outer_false->set_name("outer_false");
    outer_merge->set_name("outer_merge");
    exit->set_name("exit");

    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t two_value = 2u;
    uint32_t three_value = 3u;
    uint32_t eleven_value = 11u;
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    auto *eleven = module.create_constant(
        Type::of<uint32_t>(), &eleven_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *initial = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, eleven});
    builder.br(header);

    builder.set_insertion_point(header);
    auto *index = builder.phi(Type::of<uint32_t>());
    auto *value = builder.phi(Type::of<uint32_t>());
    value->set_name("nested_local_region_result");
    auto *parity = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_BIT_AND,
        {lane, one});
    auto *bound = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {parity, two});
    auto *loop_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {index, bound});
    builder.cond_br(loop_condition, body, exit);

    builder.set_insertion_point(body);
    auto *outer_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
        {parity, zero});
    builder.cond_br(outer_condition, nested, outer_false);

    builder.set_insertion_point(nested);
    auto *advanced = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {value, index});
    auto *inner_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {advanced, lane});
    builder.cond_br(
        inner_condition, inner_true, inner_false);

    builder.set_insertion_point(inner_true);
    builder.br(inner_merge);
    builder.set_insertion_point(inner_false);
    builder.br(inner_merge);

    builder.set_insertion_point(inner_merge);
    auto *inner_value = builder.phi(
        Type::of<uint32_t>(),
        {{advanced, inner_true}, {three, inner_false}});
    builder.br(outer_merge);

    builder.set_insertion_point(outer_false);
    builder.br(outer_merge);

    builder.set_insertion_point(outer_merge);
    auto *next_value = builder.phi(
        Type::of<uint32_t>(),
        {{inner_value, inner_merge}, {value, outer_false}});
    auto *next_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {index, one});
    builder.br(header);

    builder.set_insertion_point(exit);
    builder.return_void();
    index->add_incoming(zero, entry);
    index->add_incoming(next_index, outer_merge);
    value->add_incoming(initial, entry);
    value->add_incoming(next_value, outer_merge);

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    for (auto &&schedule_value : lowered.function->values()) {
        if (schedule_value.name ==
            "nested_local_region_result") {
            result_id = schedule_value.id;
        }
    }
    if (!result_id) { return std::nullopt; }
    for (auto &schedule_block : lowered.function->blocks()) {
        if (schedule_block.name == "exit") {
            schedule_block.terminator =
                schedule::ReturnTerminator{result_id};
        }
    }
    if (!schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
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
make_runtime_coherent_branch(uint32_t width) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("runtime_coherent_branch");
    auto *entry = kernel->create_body_block();
    auto *true_block = kernel->create_basic_block();
    auto *false_block = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    entry->set_name("entry");
    true_block->set_name("true_block");
    false_block->set_name("false_block");
    merge->set_name("merge");

    auto *lane = module.create_warp_lane_id();
    auto *dispatch_id = module.create_dispatch_id();
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    uint32_t threshold_value = 2u;
    uint32_t true_addend_value = 10u;
    uint32_t false_addend_value = 20u;
    auto *threshold = module.create_constant(
        Type::of<uint32_t>(), &threshold_value);
    auto *true_addend = module.create_constant(
        Type::of<uint32_t>(), &true_addend_value);
    auto *false_addend = module.create_constant(
        Type::of<uint32_t>(), &false_addend_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *selector = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, one});
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {selector, threshold});
    builder.cond_br(condition, true_block, false_block);
    builder.set_insertion_point(true_block);
    auto *true_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, true_addend});
    builder.br(merge);
    builder.set_insertion_point(false_block);
    auto *false_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, false_addend});
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{true_result, true_block}, {false_result, false_block}});
    selected->set_name("runtime_coherent_branch_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    auto saw_varying_split = false;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "runtime_coherent_branch_result") {
            result_id = value.id;
        }
    }
    for (auto &block : lowered.function->blocks()) {
        if (auto *terminator = std::get_if<schedule::SplitTerminator>(
                &block.terminator)) {
            auto *schedule_condition =
                lowered.function->value(terminator->condition);
            saw_varying_split =
                terminator->convergence.has_value() &&
                schedule_condition != nullptr &&
                schedule_condition->value_class ==
                    schedule::ValueClass::varying;
        }
        if (block.name == "merge") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!result_id || !saw_varying_split ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_coherent_all_on_region(uint32_t width,
                            bool divergent_entry,
                            bool short_region = false) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("coherent_all_on_region");
    auto *entry = kernel->create_body_block();
    auto *true_arm = kernel->create_basic_block();
    auto *false_arm = kernel->create_basic_block();
    auto *merge = kernel->create_basic_block();
    auto *next_test = short_region ?
                          nullptr :
                          kernel->create_basic_block();
    auto *next_true = kernel->create_basic_block();
    auto *next_false = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    entry->set_name("all_on_entry");
    true_arm->set_name("all_on_expensive_arm");
    false_arm->set_name("all_on_cheap_arm");
    merge->set_name("all_on_merge");
    if (next_test != nullptr) {
        next_test->set_name("all_on_next_test");
    }
    next_true->set_name("all_on_next_true");
    next_false->set_name("all_on_next_false");
    exit->set_name("all_on_exit");

    auto *lane = module.create_warp_lane_id();
    auto *dispatch_id = module.create_dispatch_id();
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto two_value = uint32_t{2u};
    auto ten_value = uint32_t{10u};
    auto thousand_value = uint32_t{1000u};
    auto two_thousand_value = uint32_t{2000u};
    auto half_value = std::max(1u, width / 2u);
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *ten = module.create_constant(
        Type::of<uint32_t>(), &ten_value);
    auto *thousand = module.create_constant(
        Type::of<uint32_t>(), &thousand_value);
    auto *two_thousand = module.create_constant(
        Type::of<uint32_t>(), &two_thousand_value);
    auto *half = module.create_constant(
        Type::of<uint32_t>(), &half_value);

    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *selector = static_cast<xir::Value *>(lane);
    if (!divergent_entry) {
        selector = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
            {dispatch_id, one});
    }
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {selector, half});
    builder.cond_br(condition, true_arm, false_arm);

    builder.set_insertion_point(true_arm);
    auto *true_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_DIV,
        {lane, two});
    builder.br(merge);

    builder.set_insertion_point(false_arm);
    auto *false_result = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {lane, ten});
    builder.br(merge);

    builder.set_insertion_point(merge);
    auto *selected = builder.phi(
        Type::of<uint32_t>(),
        {{true_result, true_arm}, {false_result, false_arm}});
    auto *advanced = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {selected, one});
    if (next_test != nullptr) {
        builder.br(next_test);
        builder.set_insertion_point(next_test);
    }
    auto *next_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {lane, half});
    builder.cond_br(next_condition, next_true, next_false);

    builder.set_insertion_point(next_true);
    auto *upper = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {advanced, thousand});
    builder.br(exit);

    builder.set_insertion_point(next_false);
    auto *lower = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {advanced, two_thousand});
    builder.br(exit);

    builder.set_insertion_point(exit);
    auto *result = builder.phi(
        Type::of<uint32_t>(),
        {{upper, next_true}, {lower, next_false}});
    result->set_name("all_on_region_result");
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return std::nullopt;
    }
    std::optional<schedule::ValueId> result_id;
    for (auto &&value : lowered.function->values()) {
        if (value.name == "all_on_region_result") {
            result_id = value.id;
        }
    }
    for (auto &block : lowered.function->blocks()) {
        if (block.name == "all_on_exit") {
            block.terminator = schedule::ReturnTerminator{result_id};
        }
    }
    if (!result_id ||
        !schedule::verify(*lowered.function).succeeded()) {
        return std::nullopt;
    }
    return std::move(*lowered.function);
}

[[nodiscard]] std::optional<schedule::Function>
make_varying_switch(uint32_t width,
                    bool runtime_coherent = false) {
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name(runtime_coherent ?
                         "runtime_coherent_switch" :
                         "varying_switch");
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
    auto *selector = static_cast<xir::Value *>(lane);
    if (runtime_coherent) {
        auto *dispatch_id = module.create_dispatch_id();
        auto *one = module.create_constant_one(Type::of<uint32_t>());
        selector = builder.call(
            Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
            {dispatch_id, one});
    }
    auto *branch = builder.indexed_branch(selector);
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
            auto *schedule_selector =
                lowered.function->value(terminator->selector);
            saw_convergent_switch =
                terminator->convergence.has_value() &&
                schedule_selector != nullptr &&
                schedule_selector->value_class ==
                    schedule::ValueClass::varying;
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
    CHECK(codegen.direct_divergent_child_count == 0u);
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

[[nodiscard]] bool run_direct_divergent_child_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        auto schedule_function = make_divergent_collective(width);
        CHECK(schedule_function.has_value());
        std::string candidate_ir;
        std::string oracle_ir;
        for (auto disable_direct_child : {false, true}) {
            ScopedEnvironmentVariable force{
                "LUISA_SIMD_FORCE_DIRECT_DIVERGENT_CHILD",
                disable_direct_child ? nullptr : "1"};
            ScopedEnvironmentVariable disable{
                "LUISA_SIMD_DISABLE_DIRECT_DIVERGENT_CHILD",
                disable_direct_child ? "1" : nullptr};
            auto context = std::make_unique<::llvm::LLVMContext>();
            auto module = std::make_unique<::llvm::Module>(
                "simd-direct-divergent-child", *context);
            auto name =
                std::string{"simd_direct_divergent_child_w"} +
                std::to_string(width);
            auto codegen = lower_schedule_to_llvm(
                *module, *schedule_function, width, name);
            if (!codegen.succeeded()) {
                std::cerr << codegen.error << '\n';
                return false;
            }
            auto expected_count =
                width >= 4u && !disable_direct_child ? 1u : 0u;
            CHECK(codegen.direct_divergent_child_count == expected_count);
            CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
            auto &ir = disable_direct_child ? oracle_ir : candidate_ir;
            ::llvm::raw_string_ostream stream{ir};
            module->print(stream, nullptr);
            stream.flush();
            LLVMJIT jit;
            CHECK(jit.succeeded());
            CHECK(jit.add_module(
                std::move(module), std::move(context)));
            using Entry = void(
                const void *, uint32_t *,
                const SIMDPacketLaunchConfig *, uint32_t);
            auto function = reinterpret_cast<Entry *>(jit.lookup(name));
            CHECK(function != nullptr);
            for (auto active_lanes = uint32_t{0u};
                 active_lanes <= width; active_lanes++) {
                std::vector<uint32_t> output(width, 0xdeadbeefu);
                auto config = launch_1d(active_lanes, width);
                function(
                    nullptr, output.data(),
                    &config, active_lanes);
                auto expected_sum = active_lanes <= 2u ?
                                        active_lanes * 2u :
                                        active_lanes + 2u;
                for (auto lane = uint32_t{0u}; lane < width; lane++) {
                    auto expected = lane < active_lanes ?
                                        expected_sum :
                                        0xdeadbeefu;
                    CHECK(output[lane] == expected);
                }
            }
        }
        if (width == 2u) {
            CHECK(candidate_ir == oracle_ir);
        } else {
            CHECK(candidate_ir != oracle_ir);
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

[[nodiscard]] bool run_local_predicated_region_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        auto schedule_function =
            make_local_predicated_sqrt_loop(width);
        CHECK(schedule_function.has_value());
        auto run_variant = [&](bool disable_regions,
                               bool disable_chaining,
                               bool disable_nested_tail,
                               bool disable_terminal_bridge,
                               std::array<float, 16u> &output,
                               LLVMScheduleCodegenResult &result) {
            auto context = std::make_unique<::llvm::LLVMContext>();
            auto module = std::make_unique<::llvm::Module>(
                disable_regions ?
                    "simd-local-region-oracle" :
                disable_chaining ?
                    "simd-local-region-chain-oracle" :
                disable_nested_tail ?
                    "simd-local-region-tail-oracle" :
                    "simd-local-region",
                *context);
            auto name = std::string{
                            disable_regions ?
                                "schedule_local_region_oracle_w" :
                            disable_chaining ?
                                "schedule_local_region_chain_oracle_w" :
                            disable_nested_tail ?
                                "schedule_local_region_tail_oracle_w" :
                                "schedule_local_region_w"} +
                        std::to_string(width);
            {
                ScopedEnvironmentVariable policy{
                    "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_REGIONS",
                    disable_regions ? "1" : nullptr};
                ScopedEnvironmentVariable chain_policy{
                    "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_CHAINING",
                    disable_chaining ? "1" : nullptr};
                ScopedEnvironmentVariable nested_tail_policy{
                    "LUISA_SIMD_DISABLE_CHAINED_NESTED_TAIL",
                    disable_nested_tail ? "1" : nullptr};
                ScopedEnvironmentVariable terminal_bridge_policy{
                    "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_TERMINAL_BRIDGE",
                    disable_terminal_bridge ? "1" : nullptr};
                result = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
            }
            if (!result.succeeded() ||
                ::llvm::verifyModule(*module, &::llvm::errs())) {
                return false;
            }
            LLVMJIT jit;
            if (!jit.succeeded() ||
                !jit.add_module(
                    std::move(module), std::move(context))) {
                return false;
            }
            using Entry = void(
                const void *, float *,
                const SIMDPacketLaunchConfig *, uint32_t);
            auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
            if (entry == nullptr) { return false; }
            output.fill(std::bit_cast<float>(0x7fc01234u));
            auto active_lanes = width - 1u;
            auto config = launch_1d(active_lanes, 16u);
            entry(nullptr, output.data(), &config, active_lanes);
            return true;
        };

        std::array<float, 16u> candidate_output{};
        std::array<float, 16u> tail_oracle_output{};
        std::array<float, 16u> chain_oracle_output{};
        std::array<float, 16u> terminal_oracle_output{};
        std::array<float, 16u> oracle_output{};
        LLVMScheduleCodegenResult candidate;
        LLVMScheduleCodegenResult tail_oracle;
        LLVMScheduleCodegenResult chain_oracle;
        LLVMScheduleCodegenResult terminal_oracle;
        LLVMScheduleCodegenResult oracle;
        CHECK(run_variant(
            false, false, false, false, candidate_output, candidate));
        CHECK(run_variant(
            false, false, true, false, tail_oracle_output, tail_oracle));
        CHECK(run_variant(
            false, true, false, true,
            chain_oracle_output, chain_oracle));
        CHECK(run_variant(
            false, false, false, true,
            terminal_oracle_output, terminal_oracle));
        CHECK(run_variant(
            true, false, false, false, oracle_output, oracle));
        CHECK(candidate.local_predicated_diamond_count == 3u);
        CHECK(candidate.local_predicated_assignment_diamond_count == 1u);
        CHECK(candidate.local_predicated_block_count >= 6u);
        CHECK(candidate.local_predicated_instruction_count == 8u);
        CHECK(candidate.nested_predicated_region_count == 1u);
        CHECK(candidate.nested_predicated_instruction_count == 2u);
        CHECK(candidate.chained_predicated_region_count ==
              (width >= 4u ? 1u : 0u));
        CHECK(candidate.chained_predicated_transition_count ==
              (width == 8u ? 2u : width >= 4u ? 1u :
                                                0u));
        CHECK(candidate.chained_predicated_nested_tail_count ==
              (width == 8u ? 1u : 0u));
        CHECK(candidate.chained_predicated_terminal_block_count == 0u ||
              width >= 8u);
        CHECK(candidate.chained_predicated_terminal_instruction_count == 0u ||
              width >= 8u);
        if (width >= 4u) {
            CHECK(candidate.chained_predicated_block_count >= 5u);
        } else {
            CHECK(candidate.chained_predicated_block_count == 0u);
        }
        CHECK(tail_oracle.local_predicated_diamond_count == 3u);
        CHECK(tail_oracle.nested_predicated_region_count == 1u);
        CHECK(tail_oracle.chained_predicated_region_count ==
              (width >= 4u ? 1u : 0u));
        CHECK(tail_oracle.chained_predicated_transition_count ==
              (width >= 4u ? 1u : 0u));
        CHECK(tail_oracle.chained_predicated_nested_tail_count == 0u);
        CHECK(chain_oracle.local_predicated_diamond_count == 3u);
        CHECK(chain_oracle.local_predicated_assignment_diamond_count == 1u);
        CHECK(chain_oracle.local_predicated_instruction_count == 8u);
        CHECK(chain_oracle.nested_predicated_region_count == 1u);
        CHECK(chain_oracle.chained_predicated_region_count == 0u);
        CHECK(chain_oracle.chained_predicated_transition_count == 0u);
        CHECK(chain_oracle.chained_predicated_block_count == 0u);
        CHECK(chain_oracle.chained_predicated_terminal_block_count == 0u);
        CHECK(chain_oracle.chained_predicated_terminal_instruction_count == 0u);
        CHECK(terminal_oracle.local_predicated_diamond_count == 3u);
        CHECK(terminal_oracle.chained_predicated_terminal_block_count == 0u);
        CHECK(terminal_oracle.chained_predicated_terminal_instruction_count ==
              0u);
        CHECK(oracle.local_predicated_diamond_count == 0u);
        CHECK(oracle.local_predicated_assignment_diamond_count == 0u);
        CHECK(oracle.local_predicated_block_count == 0u);
        CHECK(oracle.local_predicated_instruction_count == 0u);
        CHECK(oracle.nested_predicated_region_count == 0u);
        CHECK(oracle.chained_predicated_nested_tail_count == 0u);
        for (auto lane = size_t{0u}; lane < candidate_output.size(); lane++) {
            CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                  std::bit_cast<uint32_t>(oracle_output[lane]));
            CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                  std::bit_cast<uint32_t>(chain_oracle_output[lane]));
            CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                  std::bit_cast<uint32_t>(tail_oracle_output[lane]));
            CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                  std::bit_cast<uint32_t>(terminal_oracle_output[lane]));
        }
    }
    return true;
}

[[nodiscard]] bool run_local_predicated_terminal_bridge_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        auto schedule_function =
            make_local_predicated_terminal_bridge_loop(width);
        CHECK(schedule_function.has_value());
        auto run_variant = [&](bool disable_terminal_bridge,
                               std::array<float, 16u> &output,
                               LLVMScheduleCodegenResult &result) {
            auto context = std::make_unique<::llvm::LLVMContext>();
            auto module = std::make_unique<::llvm::Module>(
                disable_terminal_bridge ?
                    "simd-local-terminal-bridge-oracle" :
                    "simd-local-terminal-bridge",
                *context);
            auto name = std::string{
                            disable_terminal_bridge ?
                                "schedule_local_terminal_bridge_oracle_w" :
                                "schedule_local_terminal_bridge_w"} +
                        std::to_string(width);
            {
                ScopedEnvironmentVariable force{
                    "LUISA_SIMD_FORCE_LOCAL_PREDICATED_TERMINAL_BRIDGE",
                    width == 2u ? "1" : nullptr};
                ScopedEnvironmentVariable disable{
                    "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_TERMINAL_BRIDGE",
                    disable_terminal_bridge ? "1" : nullptr};
                result = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
            }
            if (!result.succeeded() ||
                ::llvm::verifyModule(*module, &::llvm::errs())) {
                return false;
            }
            LLVMJIT jit;
            if (!jit.succeeded()) { return false; }
            auto assembly = jit.emit_assembly_copy(*module);
            if (assembly.empty() ||
                assembly.find("sqrtf") != std::string::npos ||
                !jit.add_module(std::move(module), std::move(context))) {
                return false;
            }
            using Entry = void(
                const void *, float *,
                const SIMDPacketLaunchConfig *, uint32_t);
            auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
            if (entry == nullptr) { return false; }
            output.fill(std::bit_cast<float>(0x7fc01234u));
            auto active_lanes = width - 1u;
            auto config = launch_1d(active_lanes, 16u);
            entry(nullptr, output.data(), &config, active_lanes);
            return true;
        };

        std::array<float, 16u> candidate_output{};
        std::array<float, 16u> oracle_output{};
        LLVMScheduleCodegenResult candidate;
        LLVMScheduleCodegenResult oracle;
        CHECK(run_variant(false, candidate_output, candidate));
        CHECK(run_variant(true, oracle_output, oracle));
        CHECK(candidate.local_predicated_diamond_count == 1u);
        CHECK(candidate.chained_predicated_region_count == 1u);
        CHECK(candidate.chained_predicated_transition_count == 0u);
        CHECK(candidate.chained_predicated_terminal_block_count == 1u);
        CHECK(candidate.chained_predicated_terminal_instruction_count >= 40u);
        CHECK(oracle.local_predicated_diamond_count == 1u);
        CHECK(oracle.chained_predicated_region_count == 0u);
        CHECK(oracle.chained_predicated_terminal_block_count == 0u);
        CHECK(oracle.chained_predicated_terminal_instruction_count == 0u);
        for (auto lane = size_t{0u}; lane < candidate_output.size(); lane++) {
            CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                  std::bit_cast<uint32_t>(oracle_output[lane]));
        }
    }
    return true;
}

[[nodiscard]] bool run_two_sided_local_predicated_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        auto schedule_function =
            make_two_sided_local_predicated_loop(width);
        CHECK(schedule_function.has_value());
        auto run_variant = [&](bool disable,
                               std::array<float, 16u> &output,
                               LLVMScheduleCodegenResult &result) {
            auto context = std::make_unique<::llvm::LLVMContext>();
            auto module = std::make_unique<::llvm::Module>(
                disable ? "simd-two-sided-local-oracle" :
                          "simd-two-sided-local",
                *context);
            auto name = std::string{
                            disable ?
                                "schedule_two_sided_local_oracle_w" :
                                "schedule_two_sided_local_w"} +
                        std::to_string(width);
            {
                ScopedEnvironmentVariable force{
                    "LUISA_SIMD_FORCE_TWO_SIDED_LOCAL_PREDICATION",
                    width == 16u ? "1" : nullptr};
                ScopedEnvironmentVariable policy{
                    "LUISA_SIMD_DISABLE_TWO_SIDED_LOCAL_PREDICATION",
                    disable ? "1" : nullptr};
                result = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
            }
            if (!result.succeeded() ||
                ::llvm::verifyModule(*module, &::llvm::errs())) {
                return false;
            }
            LLVMJIT jit;
            if (!jit.succeeded() ||
                !jit.add_module(std::move(module), std::move(context))) {
                return false;
            }
            using Entry = void(
                const void *, float *,
                const SIMDPacketLaunchConfig *, uint32_t);
            auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
            if (entry == nullptr) { return false; }
            output.fill(std::bit_cast<float>(0x7fc01234u));
            auto active_lanes = width - 1u;
            auto config = launch_1d(active_lanes, 16u);
            entry(nullptr, output.data(), &config, active_lanes);
            return true;
        };

        std::array<float, 16u> candidate_output{};
        std::array<float, 16u> oracle_output{};
        LLVMScheduleCodegenResult candidate;
        LLVMScheduleCodegenResult oracle;
        CHECK(run_variant(false, candidate_output, candidate));
        CHECK(run_variant(true, oracle_output, oracle));
        CHECK(candidate.local_predicated_diamond_count == 1u);
        CHECK(candidate.local_predicated_two_sided_diamond_count == 1u);
        CHECK(candidate.local_predicated_assignment_diamond_count == 0u);
        CHECK(candidate.local_predicated_block_count == 2u);
        CHECK(candidate.local_predicated_instruction_count == 8u);
        CHECK(candidate.chained_predicated_region_count ==
              (width >= 8u ? 1u : 0u));
        CHECK(candidate.chained_predicated_terminal_block_count ==
              (width >= 8u ? 1u : 0u));
        CHECK(candidate.chained_predicated_terminal_instruction_count ==
              (width >= 8u ? 2u : 0u));
        CHECK(oracle.local_predicated_diamond_count == 0u);
        CHECK(oracle.local_predicated_two_sided_diamond_count == 0u);
        CHECK(oracle.chained_predicated_region_count == 0u);
        auto active_lanes = static_cast<size_t>(width - 1u);
        for (auto lane = size_t{0u}; lane < candidate_output.size(); lane++) {
            if (std::bit_cast<uint32_t>(candidate_output[lane]) !=
                std::bit_cast<uint32_t>(oracle_output[lane])) {
                std::cerr << "two-sided local mismatch: width=" << width
                          << ", lane=" << lane
                          << ", candidate=0x" << std::hex
                          << std::bit_cast<uint32_t>(candidate_output[lane])
                          << ", oracle=0x"
                          << std::bit_cast<uint32_t>(oracle_output[lane])
                          << std::dec << '\n';
            }
            CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                  std::bit_cast<uint32_t>(oracle_output[lane]));
            if (lane >= active_lanes) {
                CHECK(std::bit_cast<uint32_t>(candidate_output[lane]) ==
                      0x7fc01234u);
            }
        }
    }
    return true;
}

[[nodiscard]] bool run_nested_local_predicated_region_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        auto schedule_function =
            make_nested_local_predicated_loop(width);
        CHECK(schedule_function.has_value());
        auto run_variant = [&](bool disable,
                               std::array<uint32_t, 16u> &output,
                               LLVMScheduleCodegenResult &result) {
            auto context = std::make_unique<::llvm::LLVMContext>();
            auto module = std::make_unique<::llvm::Module>(
                disable ? "simd-nested-local-region-oracle" :
                          "simd-nested-local-region",
                *context);
            auto name = std::string{
                            disable ?
                                "schedule_nested_local_region_oracle_w" :
                                "schedule_nested_local_region_w"} +
                        std::to_string(width);
            {
                ScopedEnvironmentVariable policy{
                    "LUISA_SIMD_DISABLE_LOCAL_PREDICATED_REGIONS",
                    disable ? "1" : nullptr};
                result = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
            }
            if (!result.succeeded() ||
                ::llvm::verifyModule(*module, &::llvm::errs())) {
                return false;
            }
            LLVMJIT jit;
            if (!jit.succeeded() ||
                !jit.add_module(
                    std::move(module), std::move(context))) {
                return false;
            }
            using Entry = void(
                const void *, uint32_t *,
                const SIMDPacketLaunchConfig *, uint32_t);
            auto entry = reinterpret_cast<Entry *>(jit.lookup(name));
            if (entry == nullptr) { return false; }
            output.fill(0xdeadbeefu);
            auto active_lanes = width - 1u;
            auto config = launch_1d(active_lanes, 16u);
            entry(nullptr, output.data(), &config, active_lanes);
            return true;
        };

        std::array<uint32_t, 16u> candidate_output{};
        std::array<uint32_t, 16u> oracle_output{};
        LLVMScheduleCodegenResult candidate;
        LLVMScheduleCodegenResult oracle;
        CHECK(run_variant(false, candidate_output, candidate));
        CHECK(run_variant(true, oracle_output, oracle));
        CHECK(candidate.local_predicated_diamond_count == 1u);
        CHECK(candidate.local_predicated_assignment_diamond_count == 1u);
        CHECK(candidate.local_predicated_instruction_count == 0u);
        CHECK(candidate.nested_predicated_region_count == 1u);
        CHECK(candidate.nested_predicated_block_count >= 5u);
        CHECK(candidate.nested_predicated_instruction_count == 2u);
        CHECK(candidate.chained_predicated_region_count == 0u);
        CHECK(candidate.chained_predicated_transition_count == 0u);
        CHECK(oracle.local_predicated_diamond_count == 0u);
        CHECK(oracle.local_predicated_assignment_diamond_count == 0u);
        CHECK(oracle.local_predicated_instruction_count == 0u);
        CHECK(oracle.nested_predicated_region_count == 0u);
        CHECK(oracle.nested_predicated_block_count == 0u);
        CHECK(oracle.nested_predicated_instruction_count == 0u);
        CHECK(candidate_output == oracle_output);
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
        if constexpr (Width >= 4u) {
            CHECK(ir.find(
                      "@convergence.targets = private unnamed_addr "
                      "constant") != std::string::npos);
            CHECK(ir.find(
                      "convergence.dynamic.target = load i32") !=
                  std::string::npos);
        } else {
            CHECK(ir.find("@convergence.targets =") ==
                  std::string::npos);
            CHECK(ir.find(
                      "convergence.dynamic.target = extractelement") !=
                  std::string::npos);
        }
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

[[nodiscard]] bool run_nested_w2_codegen() {
    return run_control_fixture<2u>(
        make_nested_divergence(2u), "schedule_nested_w2", 1u);
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

template<size_t Width>
[[nodiscard]] bool run_state_phi_coalescing_width() {
    auto schedule_function = make_state_phi_coalescing(Width);
    CHECK(schedule_function.has_value());
    struct RunResult {
        size_t state_slots{0u};
        size_t coalesced_slots{0u};
        bool direct_control_flow{false};
        std::string assembly{};
        std::array<std::array<uint32_t, Width>, Width + 1u> outputs{};
    };
    auto run = [&](bool disable) -> std::optional<RunResult> {
        ScopedEnvironmentVariable setting{
            "LUISA_SIMD_DISABLE_STATE_PHI_COALESCING",
            disable ? "1" : "0"};
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            "state-phi-coalescing", *context);
        auto name = std::string{"state_phi_coalescing_w"} +
                    std::to_string(Width);
        auto codegen = lower_schedule_to_llvm(
            *module, *schedule_function, Width, name);
        if (!codegen.succeeded() ||
            ::llvm::verifyModule(*module, &::llvm::errs())) {
            if (!codegen.error.empty()) {
                std::cerr << codegen.error << '\n';
            }
            return std::nullopt;
        }
        LLVMJIT jit;
        if (!jit.succeeded()) { return std::nullopt; }
        auto assembly = jit.emit_assembly_copy(*module);
        if (assembly.empty() ||
            !jit.add_module(std::move(module), std::move(context))) {
            return std::nullopt;
        }
        using Entry = void(
            const void *, uint32_t *,
            const SIMDPacketLaunchConfig *, uint32_t);
        auto *entry = reinterpret_cast<Entry *>(jit.lookup(name));
        if (entry == nullptr) { return std::nullopt; }
        RunResult result{
            .state_slots = codegen.state_slot_count,
            .coalesced_slots = codegen.coalesced_state_slot_count,
            .direct_control_flow = codegen.direct_control_flow,
            .assembly = std::move(assembly),
        };
        for (auto active_lanes = uint32_t{0u};
             active_lanes <= Width; active_lanes++) {
            auto &output = result.outputs[active_lanes];
            output.fill(0xdeadbeefu);
            auto config = launch_1d(active_lanes, Width);
            entry(nullptr, output.data(), &config, active_lanes);
        }
        return result;
    };

    auto candidate = run(false);
    auto oracle = run(true);
    CHECK(candidate.has_value());
    CHECK(oracle.has_value());
    CHECK(candidate->state_slots == oracle->state_slots);
    CHECK(oracle->coalesced_slots == 0u);
    if constexpr (Width == 1u) {
        CHECK(candidate->direct_control_flow);
        CHECK(candidate->coalesced_slots == 0u);
        CHECK(candidate->assembly == oracle->assembly);
    } else {
        CHECK(!candidate->direct_control_flow);
        CHECK(candidate->coalesced_slots != 0u);
        CHECK(candidate->assembly != oracle->assembly);
    }
    CHECK(candidate->outputs == oracle->outputs);
    for (auto active_lanes = uint32_t{0u};
         active_lanes <= Width; active_lanes++) {
        for (auto lane = uint32_t{0u}; lane < Width; lane++) {
            auto initial = lane + 10u;
            auto even = lane % 2u == 0u;
            auto low_pair = (lane & 2u) == 0u;
            auto first_state = initial + (even ? 100u : 200u);
            auto second_state = first_state +
                                (low_pair ? 1000u : 0u);
            auto first_left = initial + (even ? 10u : 100u);
            auto first_right = initial + (even ? 200u : 1000u);
            auto expected = second_state * 1000u +
                            (low_pair ? first_left : first_right);
            CHECK(candidate->outputs[active_lanes][lane] ==
                  (lane < active_lanes ? expected : 0xdeadbeefu));
        }
    }
    return true;
}

[[nodiscard]] bool run_state_phi_coalescing_codegen() {
    return run_state_phi_coalescing_width<1u>() &&
           run_state_phi_coalescing_width<2u>() &&
           run_state_phi_coalescing_width<4u>() &&
           run_state_phi_coalescing_width<8u>() &&
           run_state_phi_coalescing_width<16u>();
}

template<size_t Width>
[[nodiscard]] bool run_general_state_coloring_width() {
    auto schedule_function =
        make_general_state_coloring_pressure(Width);
    CHECK(schedule_function.has_value());
    enum class Mode : uint8_t { production,
                                forced,
                                disabled };
    struct RunResult {
        size_t state_slots{0u};
        size_t coalesced_slots{0u};
        size_t general_colored_slots{0u};
        bool direct_control_flow{false};
        std::string assembly{};
        std::array<std::array<uint32_t, Width>, Width + 1u> outputs{};
    };
    auto run = [&](Mode mode) -> std::optional<RunResult> {
        ScopedEnvironmentVariable master{
            "LUISA_SIMD_DISABLE_STATE_PHI_COALESCING", "0"};
        ScopedEnvironmentVariable force{
            "LUISA_SIMD_FORCE_GENERAL_STATE_COLORING",
            mode == Mode::forced ? "1" : "0"};
        ScopedEnvironmentVariable disable{
            "LUISA_SIMD_DISABLE_GENERAL_STATE_COLORING",
            mode == Mode::disabled ? "1" : "0"};
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            "general-state-coloring", *context);
        auto name = std::string{"general_state_coloring_w"} +
                    std::to_string(Width);
        auto codegen = lower_schedule_to_llvm(
            *module, *schedule_function, Width, name);
        if (!codegen.succeeded() ||
            ::llvm::verifyModule(*module, &::llvm::errs())) {
            if (!codegen.error.empty()) {
                std::cerr << codegen.error << '\n';
            }
            return std::nullopt;
        }
        LLVMJIT jit;
        if (!jit.succeeded()) { return std::nullopt; }
        auto assembly = jit.emit_assembly_copy(*module);
        if (assembly.empty() ||
            !jit.add_module(std::move(module), std::move(context))) {
            return std::nullopt;
        }
        using Entry = void(
            const void *, uint32_t *,
            const SIMDPacketLaunchConfig *, uint32_t);
        auto *entry = reinterpret_cast<Entry *>(jit.lookup(name));
        if (entry == nullptr) { return std::nullopt; }
        RunResult result{
            .state_slots = codegen.state_slot_count,
            .coalesced_slots = codegen.coalesced_state_slot_count,
            .general_colored_slots =
                codegen.general_colored_state_slot_count,
            .direct_control_flow = codegen.direct_control_flow,
            .assembly = std::move(assembly),
        };
        for (auto active_lanes = uint32_t{0u};
             active_lanes <= Width; active_lanes++) {
            auto &output = result.outputs[active_lanes];
            output.fill(0xdeadbeefu);
            auto config = launch_1d(active_lanes, Width);
            entry(nullptr, output.data(), &config, active_lanes);
        }
        return result;
    };

    auto production = run(Mode::production);
    auto forced = run(Mode::forced);
    auto oracle = run(Mode::disabled);
    CHECK(production.has_value());
    CHECK(forced.has_value());
    CHECK(oracle.has_value());
    CHECK(production->state_slots == forced->state_slots);
    CHECK(production->state_slots == oracle->state_slots);
    CHECK(oracle->general_colored_slots == 0u);
    CHECK(production->outputs == oracle->outputs);
    CHECK(forced->outputs == oracle->outputs);
    if constexpr (Width == 1u) {
        CHECK(production->direct_control_flow);
        CHECK(forced->general_colored_slots == 0u);
        CHECK(production->assembly == oracle->assembly);
        CHECK(forced->assembly == oracle->assembly);
    } else {
        CHECK(!production->direct_control_flow);
        CHECK(production->state_slots >= 32u);
        CHECK(forced->general_colored_slots != 0u);
        CHECK(forced->coalesced_slots ==
              oracle->coalesced_slots +
                  forced->general_colored_slots);
        CHECK(forced->assembly != oracle->assembly);
        if constexpr (Width == 16u) {
            CHECK(production->general_colored_slots != 0u);
            CHECK(production->coalesced_slots ==
                  forced->coalesced_slots);
            CHECK(production->assembly == forced->assembly);
        } else {
            CHECK(production->general_colored_slots == 0u);
            CHECK(production->coalesced_slots ==
                  oracle->coalesced_slots);
            CHECK(production->assembly == oracle->assembly);
        }
    }
    for (auto active_lanes = uint32_t{0u};
         active_lanes <= Width; active_lanes++) {
        for (auto lane = uint32_t{0u}; lane < Width; lane++) {
            auto expected = 16u * lane +
                            ((lane & 2u) == 0u ? 3336u : 4936u);
            CHECK(production->outputs[active_lanes][lane] ==
                  (lane < active_lanes ? expected : 0xdeadbeefu));
        }
    }
    return true;
}

[[nodiscard]] bool run_single_general_state_coloring_candidate() {
    static constexpr auto width = size_t{16u};
    auto schedule_function =
        make_single_general_state_coloring_candidate(width);
    CHECK(schedule_function.has_value());
    enum class Mode : uint8_t { production,
                                forced,
                                disabled };
    struct RunResult {
        size_t state_slots{0u};
        size_t coalesced_slots{0u};
        size_t general_colored_slots{0u};
        std::string assembly{};
        std::array<std::array<uint32_t, width>, width + 1u> outputs{};
    };
    auto run = [&](Mode mode) -> std::optional<RunResult> {
        ScopedEnvironmentVariable master{
            "LUISA_SIMD_DISABLE_STATE_PHI_COALESCING", "0"};
        ScopedEnvironmentVariable force{
            "LUISA_SIMD_FORCE_GENERAL_STATE_COLORING",
            mode == Mode::forced ? "1" : "0"};
        ScopedEnvironmentVariable disable{
            "LUISA_SIMD_DISABLE_GENERAL_STATE_COLORING",
            mode == Mode::disabled ? "1" : "0"};
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            "single-general-state-coloring", *context);
        auto name = std::string{"single_general_state_coloring"};
        auto codegen = lower_schedule_to_llvm(
            *module, *schedule_function, width, name);
        if (!codegen.succeeded() ||
            ::llvm::verifyModule(*module, &::llvm::errs())) {
            if (!codegen.error.empty()) {
                std::cerr << codegen.error << '\n';
            }
            return std::nullopt;
        }
        LLVMJIT jit;
        if (!jit.succeeded()) { return std::nullopt; }
        auto assembly = jit.emit_assembly_copy(*module);
        if (assembly.empty() ||
            !jit.add_module(std::move(module), std::move(context))) {
            return std::nullopt;
        }
        using Entry = void(
            const void *, uint32_t *,
            const SIMDPacketLaunchConfig *, uint32_t);
        auto *entry = reinterpret_cast<Entry *>(jit.lookup(name));
        if (entry == nullptr) { return std::nullopt; }
        RunResult result{
            .state_slots = codegen.state_slot_count,
            .coalesced_slots = codegen.coalesced_state_slot_count,
            .general_colored_slots =
                codegen.general_colored_state_slot_count,
            .assembly = std::move(assembly),
        };
        for (auto active_lanes = uint32_t{0u};
             active_lanes <= width; active_lanes++) {
            auto &output = result.outputs[active_lanes];
            output.fill(0xdeadbeefu);
            auto config = launch_1d(active_lanes, width);
            entry(nullptr, output.data(), &config, active_lanes);
        }
        return result;
    };

    auto production = run(Mode::production);
    auto forced = run(Mode::forced);
    auto oracle = run(Mode::disabled);
    CHECK(production.has_value());
    CHECK(forced.has_value());
    CHECK(oracle.has_value());
    CHECK(production->state_slots >= 32u);
    CHECK(production->state_slots == forced->state_slots);
    CHECK(production->state_slots == oracle->state_slots);
    CHECK(forced->general_colored_slots == 1u);
    CHECK(production->general_colored_slots == 0u);
    CHECK(oracle->general_colored_slots == 0u);
    CHECK(production->coalesced_slots == oracle->coalesced_slots);
    CHECK(forced->coalesced_slots == oracle->coalesced_slots + 1u);
    CHECK(production->assembly == oracle->assembly);
    CHECK(production->outputs == forced->outputs);
    CHECK(production->outputs == oracle->outputs);
    for (auto active_lanes = uint32_t{0u};
         active_lanes <= width; active_lanes++) {
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            auto expected = 31u * lane +
                            ((lane & 2u) == 0u ? 3596u : 6696u);
            CHECK(production->outputs[active_lanes][lane] ==
                  (lane < active_lanes ? expected : 0xdeadbeefu));
        }
    }
    return true;
}

[[nodiscard]] bool run_general_state_coloring_codegen() {
    return run_general_state_coloring_width<1u>() &&
           run_general_state_coloring_width<2u>() &&
           run_general_state_coloring_width<4u>() &&
           run_general_state_coloring_width<8u>() &&
           run_general_state_coloring_width<16u>() &&
           run_single_general_state_coloring_candidate();
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
    CHECK(codegen.coherent_mask_reuse_count == 4u);
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

[[nodiscard]] bool run_runtime_coherent_control_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        for (auto use_switch : {false, true}) {
            auto schedule_function = use_switch ?
                                         make_varying_switch(width, true) :
                                         make_runtime_coherent_branch(width);
            CHECK(schedule_function.has_value());
            for (auto disable_reuse : {false, true}) {
                ScopedEnvironmentVariable disable{
                    "LUISA_SIMD_DISABLE_COHERENT_MASK_REUSE",
                    disable_reuse ? "1" : nullptr};
                auto context = std::make_unique<::llvm::LLVMContext>();
                auto module = std::make_unique<::llvm::Module>(
                    "simd-runtime-coherent-control", *context);
                auto name = std::string{
                                use_switch ?
                                    "simd_runtime_coherent_switch_w" :
                                    "simd_runtime_coherent_branch_w"} +
                            std::to_string(width) + (disable_reuse ? "_oracle" : "_reuse");
                auto codegen = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
                if (!codegen.succeeded()) {
                    std::cerr << codegen.error << '\n';
                    return false;
                }
                CHECK(codegen.coherent_mask_reuse_count ==
                      (disable_reuse ? 0u :
                       use_switch    ? 4u :
                                       2u));
                CHECK(!codegen.direct_control_flow);
                CHECK(!::llvm::verifyModule(
                    *module, &::llvm::errs()));
                LLVMJIT jit;
                CHECK(jit.succeeded());
                CHECK(jit.add_module(
                    std::move(module), std::move(context)));
                using Entry = void(
                    const void *, uint32_t *,
                    const SIMDPacketLaunchConfig *, uint32_t);
                auto function = reinterpret_cast<Entry *>(
                    jit.lookup(name));
                CHECK(function != nullptr);
                auto selector_count = use_switch ? 4u : 2u;
                std::array<uint32_t, 4u> selectors{0u, 2u, 5u, 3u};
                for (auto selector_index = uint32_t{0u};
                     selector_index < selector_count;
                     selector_index++) {
                    auto selector = selectors[selector_index];
                    for (auto active_lanes :
                         {width, std::max(1u, width - 2u), 1u}) {
                        std::vector<uint32_t> output(
                            width, 0xdeadbeefu);
                        auto config = launch_1d(
                            active_lanes, width);
                        config.block_id[1u] = selector;
                        config.dispatch_size[1u] = 6u;
                        function(
                            nullptr, output.data(),
                            &config, active_lanes);
                        auto addend = use_switch ?
                                          selector == 0u ? 10u :
                                          selector == 2u ? 20u :
                                          selector == 5u ? 30u :
                                                           40u :
                                      selector < 2u ? 10u :
                                                      20u;
                        for (auto lane = uint32_t{0u};
                             lane < width; lane++) {
                            auto expected = lane < active_lanes ?
                                                lane + addend :
                                                0xdeadbeefu;
                            CHECK(output[lane] == expected);
                        }
                    }
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool run_coherent_all_on_region_codegen() {
    for (auto width : {2u, 4u, 8u, 16u}) {
        for (auto divergent_entry : {false, true}) {
            auto schedule_function = make_coherent_all_on_region(
                width, divergent_entry);
            CHECK(schedule_function.has_value());
            for (auto disable_versioning : {false, true}) {
                ScopedEnvironmentVariable disable{
                    "LUISA_SIMD_DISABLE_ALL_ON_REGION_VERSIONING",
                    disable_versioning ? "1" : nullptr};
                auto context = std::make_unique<::llvm::LLVMContext>();
                auto module = std::make_unique<::llvm::Module>(
                    "simd-coherent-all-on-region", *context);
                auto name = std::string{"simd_all_on_region_w"} +
                            std::to_string(width) +
                            (divergent_entry ? "_divergent" : "_coherent") +
                            (disable_versioning ? "_oracle" : "_candidate");
                auto codegen = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
                if (!codegen.succeeded()) {
                    std::cerr << codegen.error << '\n';
                    return false;
                }
                auto enabled = width == 2u || width == 8u;
                CHECK(codegen.all_on_region_version_count ==
                      (disable_versioning || !enabled ? 0u : 1u));
                CHECK(codegen.all_on_region_block_count ==
                      (disable_versioning || !enabled ? 0u : 3u));
                CHECK(codegen.all_on_region_instruction_count ==
                      (disable_versioning || !enabled ? 0u : 3u));
                CHECK(!codegen.direct_control_flow);
                CHECK(!::llvm::verifyModule(
                    *module, &::llvm::errs()));
                std::string ir;
                ::llvm::raw_string_ostream stream{ir};
                module->print(stream, nullptr);
                stream.flush();
                CHECK((ir.find("all.on.region.version") !=
                       std::string::npos) ==
                      (!disable_versioning && enabled));

                LLVMJIT jit;
                CHECK(jit.succeeded());
                CHECK(jit.add_module(
                    std::move(module), std::move(context)));
                using Entry = void(
                    const void *, uint32_t *,
                    const SIMDPacketLaunchConfig *, uint32_t);
                auto function = reinterpret_cast<Entry *>(
                    jit.lookup(name));
                CHECK(function != nullptr);
                auto half = std::max(1u, width / 2u);
                for (auto selector : {0u, width}) {
                    for (auto active_lanes :
                         {width, std::max(1u, width - 1u), 1u}) {
                        std::vector<uint32_t> output(
                            width, 0xdeadbeefu);
                        auto config = launch_1d(
                            active_lanes, width);
                        config.block_id[1u] = selector;
                        config.dispatch_size[1u] = width + 1u;
                        function(
                            nullptr, output.data(),
                            &config, active_lanes);
                        for (auto lane = uint32_t{0u};
                             lane < width; lane++) {
                            if (lane >= active_lanes) {
                                CHECK(output[lane] == 0xdeadbeefu);
                                continue;
                            }
                            auto first_true = divergent_entry ?
                                                  lane < half :
                                                  selector < half;
                            auto base = first_true ?
                                            lane / 2u :
                                            lane + 10u;
                            auto expected = base + 1u +
                                            (lane < half ? 1000u :
                                                           2000u);
                            CHECK(output[lane] == expected);
                        }
                    }
                }
            }
        }
    }

    // The fixed all-on test and code clone do not amortize over a two-block
    // W8 region. W2 retains that case because its paired Voxel measurements
    // are profitable; W8 must fail closed to the scheduler oracle.
    for (auto width : {2u, 8u}) {
        auto schedule_function = make_coherent_all_on_region(
            width, false, true);
        CHECK(schedule_function.has_value());
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            "simd-short-coherent-all-on-region", *context);
        auto codegen = lower_schedule_to_llvm(
            *module, *schedule_function, width,
            "simd_short_all_on_region_w" +
                std::to_string(width));
        CHECK(codegen.succeeded());
        CHECK(codegen.all_on_region_version_count ==
              (width == 2u ? 1u : 0u));
        CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
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
    for (auto width : {2u, 4u, 8u, 16u}) {
        auto schedule_function =
            make_return_convergence_cascade(width);
        CHECK(schedule_function.has_value());
        auto return_count = std::count_if(
            schedule_function->blocks().begin(),
            schedule_function->blocks().end(),
            [](const schedule::BasicBlock &block) noexcept {
                return std::holds_alternative<
                    schedule::ReturnTerminator>(block.terminator);
            });
        CHECK(return_count != 0u);
        for (auto disable_return_guard : {false, true}) {
            ScopedEnvironmentVariable disable_return{
                "LUISA_SIMD_DISABLE_RETURN_FRAME_GUARD",
                disable_return_guard ? "1" : nullptr};
            for (auto disable_guard : {false, true}) {
                ScopedEnvironmentVariable disable{
                    "LUISA_SIMD_DISABLE_CONVERGENCE_TOKEN_GUARD",
                    disable_guard ? "1" : nullptr};
                auto context = std::make_unique<::llvm::LLVMContext>();
                auto module = std::make_unique<::llvm::Module>(
                    "simd-return-convergence-cascade", *context);
                auto name =
                    std::string{"simd_return_convergence_cascade_w"} +
                    std::to_string(width) +
                    (disable_guard ? "_cascade_oracle" : "_cascade_guarded") +
                    (disable_return_guard ?
                         "_return_oracle" :
                         "_return_guarded");
                auto codegen = lower_schedule_to_llvm(
                    *module, *schedule_function, width, name);
                if (!codegen.succeeded()) {
                    std::cerr << codegen.error << '\n';
                    return false;
                }
                CHECK(codegen.convergence_token_guard_count ==
                      (disable_guard ? 0u : 1u));
                CHECK(codegen.return_frame_guard_count ==
                      (disable_return_guard ? 0u : return_count));
                CHECK(!::llvm::verifyModule(*module, &::llvm::errs()));
                std::string ir;
                ::llvm::raw_string_ostream stream{ir};
                module->print(stream, nullptr);
                stream.flush();
                CHECK((ir.find("convergence.token.present") !=
                       std::string::npos) == !disable_guard);
                CHECK((ir.find("convergence.parent.present") !=
                       std::string::npos) == !disable_guard);
                CHECK((ir.find("convergence.cascade.result") !=
                       std::string::npos) == !disable_guard);
                CHECK((ir.find("return.frames.present") !=
                       std::string::npos) == !disable_return_guard);
                CHECK((ir.find("return.frame.cleanup") !=
                       std::string::npos) == !disable_return_guard);
                LLVMJIT jit;
                CHECK(jit.succeeded());
                CHECK(jit.add_module(
                    std::move(module), std::move(context)));
                using Entry = void(
                    const void *, uint32_t *,
                    const SIMDPacketLaunchConfig *, uint32_t);
                auto function = reinterpret_cast<Entry *>(
                    jit.lookup(name));
                CHECK(function != nullptr);
                for (auto active_lanes = uint32_t{1u};
                     active_lanes <= width; active_lanes++) {
                    std::vector<uint32_t> output(
                        width, 0xdeadbeefu);
                    auto config = launch_1d(active_lanes, width);
                    function(
                        nullptr, output.data(),
                        &config, active_lanes);
                    auto early_count =
                        std::min(active_lanes, 4u) -
                        std::min(active_lanes, 2u);
                    auto expected_live = active_lanes - early_count;
                    for (auto lane = uint32_t{0u};
                         lane < width; lane++) {
                        auto expected =
                            lane >= active_lanes    ? 0xdeadbeefu :
                            lane >= 2u && lane < 4u ? lane + 100u :
                                                      expected_live;
                        CHECK(output[lane] == expected);
                    }
                }
            }
        }
    }
    return true;
}

[[nodiscard]] bool run_scalar_frame_metadata_codegen() {
    for (auto width : {1u, 2u, 4u, 8u, 16u}) {
        auto schedule_function =
            make_return_convergence_cascade(width);
        CHECK(schedule_function.has_value());
        struct RunResult {
            bool scalar_frame_metadata{false};
            std::string ir;
            std::string assembly;
            std::vector<std::vector<uint32_t>> outputs;
        };
        auto run = [&](bool disable) -> std::optional<RunResult> {
            ScopedEnvironmentVariable setting{
                "LUISA_SIMD_DISABLE_SCALAR_FRAME_METADATA",
                disable ? "1" : nullptr};
            auto context = std::make_unique<::llvm::LLVMContext>();
            auto module = std::make_unique<::llvm::Module>(
                "simd-scalar-frame-metadata", *context);
            auto name =
                std::string{"simd_scalar_frame_metadata_w"} +
                std::to_string(width);
            auto codegen = lower_schedule_to_llvm(
                *module, *schedule_function, width, name);
            if (!codegen.succeeded() ||
                ::llvm::verifyModule(*module, &::llvm::errs())) {
                if (!codegen.error.empty()) {
                    std::cerr << codegen.error << '\n';
                }
                return std::nullopt;
            }
            RunResult result{
                .scalar_frame_metadata =
                    codegen.scalar_frame_metadata,
            };
            ::llvm::raw_string_ostream stream{result.ir};
            module->print(stream, nullptr);
            stream.flush();
            LLVMJIT jit;
            if (!jit.succeeded()) { return std::nullopt; }
            result.assembly = jit.emit_assembly_copy(*module);
            if (result.assembly.empty() ||
                !jit.add_module(
                    std::move(module), std::move(context))) {
                return std::nullopt;
            }
            using Entry = void(
                const void *, uint32_t *,
                const SIMDPacketLaunchConfig *, uint32_t);
            auto *function = reinterpret_cast<Entry *>(
                jit.lookup(name));
            if (function == nullptr) { return std::nullopt; }
            result.outputs.reserve(width + 1u);
            for (auto active_lanes = uint32_t{0u};
                 active_lanes <= width; active_lanes++) {
                auto &output = result.outputs.emplace_back(
                    width, 0xdeadbeefu);
                auto config = launch_1d(active_lanes, width);
                function(
                    nullptr, output.data(),
                    &config, active_lanes);
            }
            return result;
        };

        auto candidate = run(false);
        auto oracle = run(true);
        CHECK(candidate.has_value());
        CHECK(oracle.has_value());
        CHECK(!oracle->scalar_frame_metadata);
        CHECK(candidate->scalar_frame_metadata == (width == 16u));
        CHECK(candidate->outputs == oracle->outputs);
        if (width == 16u) {
            CHECK(candidate->ir != oracle->ir);
            CHECK(candidate->assembly != oracle->assembly);
            CHECK(candidate->ir.find(
                      "frame.static.id = alloca [16 x i32]") !=
                  std::string::npos);
            CHECK(candidate->ir.find(
                      "frame.parent.token = alloca [16 x i32]") !=
                  std::string::npos);
            CHECK(oracle->ir.find(
                      "frame.static.id = alloca <16 x i32>") !=
                  std::string::npos);
            CHECK(oracle->ir.find(
                      "frame.parent.token = alloca <16 x i32>") !=
                  std::string::npos);
        } else {
            CHECK(candidate->ir == oracle->ir);
            CHECK(candidate->assembly == oracle->assembly);
        }
        for (auto active_lanes = uint32_t{0u};
             active_lanes <= width; active_lanes++) {
            auto early_count =
                std::min(active_lanes, 4u) -
                std::min(active_lanes, 2u);
            auto expected_live = active_lanes - early_count;
            for (auto lane = uint32_t{0u}; lane < width; lane++) {
                auto expected =
                    lane >= active_lanes    ? 0xdeadbeefu :
                    lane >= 2u && lane < 4u ? lane + 100u :
                                              expected_live;
                CHECK(candidate->outputs[active_lanes][lane] ==
                      expected);
            }
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

[[nodiscard]] bool run_buffer_vector_codegen_width(uint32_t width) {
    static constexpr auto count = 13u;
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

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    CHECK(lowered.succeeded());
    auto make_ir = [&](bool enable_lane_affine)
        -> std::optional<std::string> {
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto llvm_module = std::make_unique<::llvm::Module>(
            enable_lane_affine ? "lane-value-transpose" :
                                 "lane-value-transpose-oracle",
            *context);
        auto codegen = lower_schedule_to_llvm(
            *llvm_module, *lowered.function, width,
            enable_lane_affine ? "simd_lane_value_transpose" :
                                 "simd_lane_value_transpose_oracle",
            false, {64u, 1u, 1u}, true,
            enable_lane_affine);
        auto expect_transpose = enable_lane_affine && width >= 2u;
        if (!codegen.succeeded() ||
            codegen.transposed_buffer_read_count !=
                (expect_transpose ? 2u : 0u) ||
            codegen.transposed_buffer_write_count !=
                (expect_transpose ? 1u : 0u) ||
            ::llvm::verifyModule(*llvm_module, &::llvm::errs())) {
            return std::nullopt;
        }
        std::string ir;
        ::llvm::raw_string_ostream stream{ir};
        llvm_module->print(stream, nullptr);
        stream.flush();
        return std::optional<std::string>{std::move(ir)};
    };
    auto candidate_ir = make_ir(true);
    auto oracle_ir = make_ir(false);
    CHECK(candidate_ir.has_value());
    CHECK(oracle_ir.has_value());
    if (width >= 2u) {
        CHECK(count_occurrences(
                  *candidate_ir,
                  "call <" + std::to_string(width * 4u) +
                      " x i32> @llvm.masked.load") == 2u);
        CHECK(count_occurrences(
                  *candidate_ir,
                  "call void @llvm.masked.store") == 1u);
        CHECK(candidate_ir->find("@llvm.masked.gather") ==
              std::string::npos);
        CHECK(candidate_ir->find("@llvm.masked.scatter") ==
              std::string::npos);
    }
    CHECK(count_occurrences(
              *oracle_ir, "call <" + std::to_string(width) +
                              " x i32> @llvm.masked.gather") == 8u);
    CHECK(count_occurrences(
              *oracle_ir, "call void @llvm.masked.scatter") == 4u);

    auto compiled = compile_simd_kernel(
        kernel, width, "simd_buffer_vector_add",
        false, true, true, true);
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.argument_buffer_size ==
          3u * sizeof(SIMDHostBufferView));
    CHECK(compiled.uniform_buffer_broadcast_count == 0u);
    auto expect_transpose = width >= 2u;
    CHECK(compiled.transposed_buffer_read_count ==
          (expect_transpose ? 2u : 0u));
    CHECK(compiled.transposed_buffer_write_count ==
          (expect_transpose ? 1u : 0u));
    CHECK(compiled.paired_leaf_gather_count == 0u);
    if (expect_transpose) {
        CHECK(compiled.assembly.find("gather") == std::string::npos);
        CHECK(compiled.assembly.find("scatter") == std::string::npos);
    }

    ScopedEnvironmentVariable disable_paired{
        "LUISA_SIMD_DISABLE_PAIRED_LEAF_GATHER", "1"};
    auto oracle = compile_simd_kernel(
        kernel, width, "simd_buffer_vector_add_oracle",
        false, true, false, true);
    CHECK(oracle.succeeded());
    CHECK(oracle.transposed_buffer_read_count == 0u);
    CHECK(oracle.transposed_buffer_write_count == 0u);
    CHECK(oracle.paired_leaf_gather_count == 0u);

    std::array<luisa::uint4, count> lhs_data{};
    std::array<luisa::uint4, count> rhs_data{};
    std::array<luisa::uint4, count> output_data{};
    std::array<luisa::uint4, count> oracle_output{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        lhs_data[i] = luisa::make_uint4(i, i + 1u, i + 2u, i + 3u);
        rhs_data[i] = luisa::make_uint4(10u, 20u, 30u, 40u);
        output_data[i] = luisa::make_uint4(0xdeadbeefu);
        oracle_output[i] = luisa::make_uint4(0xdeadbeefu);
    }
    alignas(16) std::array<SIMDHostBufferView, 3u> arguments{
        SIMDHostBufferView{lhs_data.data(), sizeof(lhs_data)},
        SIMDHostBufferView{rhs_data.data(), sizeof(rhs_data)},
        SIMDHostBufferView{output_data.data(), sizeof(output_data)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    auto oracle_entry = reinterpret_cast<Entry *>(oracle.entry);
    CHECK(entry != nullptr);
    CHECK(oracle_entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(arguments.data(), nullptr, &config, width);
    }
    arguments[2u] = SIMDHostBufferView{
        oracle_output.data(), sizeof(oracle_output)};
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        oracle_entry(arguments.data(), nullptr, &config, width);
    }
    CHECK(std::memcmp(
              output_data.data(), oracle_output.data(),
              sizeof(output_data)) == 0);
    for (auto i = uint32_t{0u}; i < count; i++) {
        CHECK(output_data[i].x == lhs_data[i].x + rhs_data[i].x);
        CHECK(output_data[i].y == lhs_data[i].y + rhs_data[i].y);
        CHECK(output_data[i].z == lhs_data[i].z + rhs_data[i].z);
        CHECK(output_data[i].w == lhs_data[i].w + rhs_data[i].w);
    }
    return true;
}

[[nodiscard]] bool run_buffer_vector_codegen() {
    for (auto width : {1u, 2u, 4u, 8u, 16u}) {
        if (!run_buffer_vector_codegen_width(width)) { return false; }
    }
    return true;
}

[[nodiscard]] bool run_sparse_lane_value_transpose_codegen() {
    static constexpr auto count = 13u;
    Kernel1D kernel = [](
                          BufferFloat2 input2, BufferFloat3 input3,
                          BufferFloat4 input4, BufferFloat2 output2,
                          BufferFloat3 output3,
                          BufferFloat4 output4) noexcept {
        auto index = dispatch_id().x;
        $if ((index & 1u) == 0u) {
            output2.write(index, input2.read(index));
            output3.write(index, input3.read(index));
            output4.write(index, input4.read(index));
        };
    };
    auto compile = [&](uint32_t width, bool disable,
                       std::string_view name) {
        ScopedEnvironmentVariable setting{
            "LUISA_SIMD_DISABLE_LANE_AFFINE_BUFFER",
            disable ? "1" : "0"};
        return compile_simd_kernel(
            kernel.function()->function(), width, name,
            false, true);
    };
    std::array<luisa::float2, count> input2{};
    std::array<luisa::float3, count> input3{};
    std::array<luisa::float4, count> input4{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        auto value = static_cast<float>(i);
        input2[i] = luisa::make_float2(
            value + 0.125f, value + 0.25f);
        input3[i] = luisa::make_float3(
            value + 1.0f, value + 2.0f, value + 3.0f);
        input4[i] = luisa::make_float4(
            value + 4.0f, value + 5.0f,
            value + 6.0f, value + 7.0f);
    }
    struct Outputs {
        std::array<luisa::float2, count> output2{};
        std::array<luisa::float3, count> output3{};
        std::array<luisa::float4, count> output4{};
    };
    static constexpr auto padding_sentinel = uint32_t{0x5aa55aa5u};
    auto execute = [&](const SIMDCompiledKernel &compiled,
                       uint32_t width, Outputs &outputs) {
        outputs.output2.fill(luisa::make_float2(-999.0f));
        outputs.output3.fill(luisa::make_float3(-999.0f));
        outputs.output4.fill(luisa::make_float4(-999.0f));
        if constexpr (sizeof(luisa::float3) == 4u * sizeof(float)) {
            for (auto &value : outputs.output3) {
                std::memcpy(
                    reinterpret_cast<std::byte *>(&value) +
                        3u * sizeof(float),
                    &padding_sentinel, sizeof(padding_sentinel));
            }
        }
        alignas(16) std::array<SIMDHostBufferView, 6u> arguments{
            SIMDHostBufferView{input2.data(), sizeof(input2)},
            SIMDHostBufferView{input3.data(), sizeof(input3)},
            SIMDHostBufferView{input4.data(), sizeof(input4)},
            SIMDHostBufferView{outputs.output2.data(),
                               sizeof(outputs.output2)},
            SIMDHostBufferView{outputs.output3.data(),
                               sizeof(outputs.output3)},
            SIMDHostBufferView{outputs.output4.data(),
                               sizeof(outputs.output4)},
        };
        using Entry = void(
            const void *, void *, const SIMDPacketLaunchConfig *,
            uint32_t);
        auto entry = reinterpret_cast<Entry *>(compiled.entry);
        CHECK(entry != nullptr);
        auto config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += width) {
            config.thread_index = first;
            entry(arguments.data(), nullptr, &config, width);
        }
        return true;
    };
    for (auto width : {1u, 2u, 4u, 8u, 16u}) {
        auto suffix = "_w" + std::to_string(width);
        auto candidate = compile(
            width, false,
            "simd_sparse_lane_value_transpose" + suffix);
        auto oracle = compile(
            width, true,
            "simd_sparse_lane_value_transpose_oracle" + suffix);
        CHECK(candidate.succeeded());
        CHECK(oracle.succeeded());
        auto expect_transpose = width >= 2u;
        CHECK(candidate.transposed_buffer_read_count ==
              (expect_transpose ? 3u : 0u));
        CHECK(candidate.transposed_buffer_write_count ==
              (expect_transpose ? 3u : 0u));
        CHECK(oracle.transposed_buffer_read_count == 0u);
        CHECK(oracle.transposed_buffer_write_count == 0u);
        if (expect_transpose) {
            CHECK(candidate.assembly.find("gather") ==
                  std::string::npos);
            CHECK(candidate.assembly.find("scatter") ==
                  std::string::npos);
        }
        Outputs candidate_outputs;
        Outputs oracle_outputs;
        CHECK(execute(candidate, width, candidate_outputs));
        CHECK(execute(oracle, width, oracle_outputs));
        for (auto i = uint32_t{0u}; i < count; i++) {
            auto even = (i & 1u) == 0u;
            auto expected2 = even ? input2[i] :
                                    luisa::make_float2(-999.0f);
            auto expected3 = even ? input3[i] :
                                    luisa::make_float3(-999.0f);
            auto expected4 = even ? input4[i] :
                                    luisa::make_float4(-999.0f);
            CHECK(luisa::all(
                candidate_outputs.output2[i] == expected2));
            CHECK(luisa::all(
                candidate_outputs.output3[i] == expected3));
            CHECK(luisa::all(
                candidate_outputs.output4[i] == expected4));
            CHECK(luisa::all(
                candidate_outputs.output2[i] ==
                oracle_outputs.output2[i]));
            CHECK(luisa::all(
                candidate_outputs.output3[i] ==
                oracle_outputs.output3[i]));
            CHECK(luisa::all(
                candidate_outputs.output4[i] ==
                oracle_outputs.output4[i]));
            if constexpr (sizeof(luisa::float3) ==
                          4u * sizeof(float)) {
                auto read_padding = [](const luisa::float3 &value) {
                    auto padding = uint32_t{0u};
                    std::memcpy(
                        &padding,
                        reinterpret_cast<const std::byte *>(&value) +
                            3u * sizeof(float),
                        sizeof(padding));
                    return padding;
                };
                CHECK(read_padding(candidate_outputs.output3[i]) ==
                      padding_sentinel);
                CHECK(read_padding(oracle_outputs.output3[i]) ==
                      padding_sentinel);
            }
        }
    }
    return true;
}

[[nodiscard]] bool run_paired_leaf_gather_ir() {
    static constexpr auto width = 8u;
    xir::Module xir_module;
    auto *kernel = xir_module.create_kernel();
    kernel->set_block_size(luisa::make_uint3(64u, 1u, 1u));
    auto *buffer_type = Type::buffer(Type::of<luisa::uint3>());
    auto *input = kernel->create_resource_argument(buffer_type);
    auto *entry = kernel->create_body_block();
    auto *dispatch_id = xir_module.create_dispatch_id();
    auto *zero = xir_module.create_constant_zero(Type::of<uint32_t>());
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    static_cast<void>(builder.call(
        Type::of<luisa::uint3>(), xir::ResourceReadOp::BUFFER_READ,
        {input, index}));
    builder.return_void();
    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    CHECK(lowered.succeeded());

    auto make_ir = [&](bool enabled)
        -> std::optional<std::string> {
        auto context = std::make_unique<::llvm::LLVMContext>();
        auto module = std::make_unique<::llvm::Module>(
            enabled ? "paired-leaf-gather" :
                      "scalar-leaf-gather",
            *context);
        auto codegen = lower_schedule_to_llvm(
            *module, *lowered.function, width,
            enabled ? "simd_paired_leaf_gather" :
                      "simd_scalar_leaf_gather",
            false, {64u, 1u, 1u}, true, false, enabled);
        if (!codegen.succeeded() ||
            codegen.paired_leaf_gather_count !=
                (enabled ? 1u : 0u) ||
            ::llvm::verifyModule(*module, &::llvm::errs())) {
            return std::nullopt;
        }
        std::string text;
        ::llvm::raw_string_ostream stream{text};
        module->print(stream, nullptr);
        stream.flush();
        return std::optional<std::string>{std::move(text)};
    };
    auto paired = make_ir(true);
    auto ordinary = make_ir(false);
    CHECK(paired.has_value());
    CHECK(ordinary.has_value());
    CHECK(count_occurrences(
              *paired, "call <8 x i64> @llvm.masked.gather") == 1u);
    CHECK(count_occurrences(
              *paired, "call <8 x i32> @llvm.masked.gather") == 1u);
    CHECK(count_occurrences(
              *ordinary, "call <8 x i64> @llvm.masked.gather") == 0u);
    CHECK(count_occurrences(
              *ordinary, "call <8 x i32> @llvm.masked.gather") == 3u);
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
    auto two_value = uint32_t{2u};
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *true_value = module.create_constant_one(Type::of<bool>());
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
    auto *varying_opacity = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {instance_id, zero});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY,
        {accel, instance_id, varying_opacity});
    builder.call(
        xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY,
        {accel, two, true_value});
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
    CHECK(ir.find("accel.instance.opacity.byte") != std::string::npos);
    CHECK(ir.find("call void %") == std::string::npos);

    LLVMJIT jit{true};
    CHECK(jit.succeeded());
    auto assembly = jit.emit_assembly_copy(*llvm_module);
    CHECK(!assembly.empty());
    CHECK(assembly.find("accel_set_instance_opacity") ==
          std::string::npos);
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    CHECK(!jit.object().empty());

    std::array<SIMDHostAccelInstance, 3u> instances{};
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
    instances[0u].opaque = 0u;
    instances[1u].user_id = 22u;
    instances[1u].mask = 0x2u;
    instances[1u].opaque = 0u;
    instances[2u].opaque = 0u;
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
    CHECK(instances[2u].dirty == 1u);
    CHECK(instances[0u].opaque == 1u);
    CHECK(instances[1u].opaque == 0u);
    CHECK(instances[2u].opaque == 1u);
    return true;
}

struct AccelDirectPacketProbe {
    uint32_t closest_calls{0u};
    uint32_t any_calls{0u};
    uint32_t failure_code{0u};
    uint32_t expected_lane_count{0u};
    uint32_t expected_active_mask{0u};
    bool valid{true};
};

void accel_direct_closest_probe(
    void *accel, uint32_t lane_count,
    void *packet_storage) {
    auto *probe = static_cast<AccelDirectPacketProbe *>(accel);
    auto *words = static_cast<uint32_t *>(packet_storage);
    auto *floats = static_cast<float *>(packet_storage);
    auto *valid = reinterpret_cast<const int *>(
        words + simd_host_accel_ray_id_field * lane_count);
    probe->closest_calls++;
    auto verify = [&](bool condition, uint32_t code) {
        if (!condition && probe->failure_code == 0u) {
            probe->failure_code = code;
        }
        probe->valid &= condition;
    };
    verify(lane_count == probe->expected_lane_count, 1u);
    verify(valid != nullptr, 2u);
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        auto active =
            (probe->expected_active_mask & (1u << lane)) != 0u;
        verify(valid[lane] == (active ? -1 : 0), 3u + lane);
        verify(
            words[simd_host_accel_ray_id_field * lane_count + lane] ==
                (active ? ~0u : 0u),
            10u + lane);
        verify(words[11u * lane_count + lane] == 0u, 20u + lane);
        verify(
            words[9u * lane_count + lane] == (active ? 0x5au : 0u),
            30u + lane);
        for (auto component = uint32_t{0u}; component < 7u;
             component++) {
            static constexpr std::array expected_components{
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
            auto expected = active ?
                                expected_components[component] :
                            component == 6u ? 1.0f :
                                              0.0f;
            verify(
                floats[component * lane_count + lane] == expected,
                100u + component * lane_count + lane);
        }
        verify(floats[7u * lane_count + lane] == 0.0f, 200u + lane);
        verify(
            floats[simd_host_accel_ray_tfar_field * lane_count + lane] ==
                (active ? 8.0f : 0.0f),
            210u + lane);
        if (!active) { continue; }
        words[simd_host_accel_hit_inst_field * lane_count + lane] =
            10u + lane;
        words[simd_host_accel_hit_prim_field * lane_count + lane] =
            20u + lane;
        floats[simd_host_accel_hit_u_field * lane_count + lane] =
            0.125f + static_cast<float>(lane);
        floats[simd_host_accel_hit_v_field * lane_count + lane] =
            0.25f + static_cast<float>(lane);
        floats[simd_host_accel_ray_tfar_field * lane_count + lane] =
            30.0f + static_cast<float>(lane);
    }
}

void accel_direct_any_probe(
    void *accel, uint32_t lane_count,
    void *packet_storage) {
    auto *probe = static_cast<AccelDirectPacketProbe *>(accel);
    auto *floats = static_cast<float *>(packet_storage);
    auto *valid = reinterpret_cast<const int *>(
        static_cast<uint32_t *>(packet_storage) +
        simd_host_accel_ray_id_field * lane_count);
    probe->any_calls++;
    probe->valid &= lane_count == probe->expected_lane_count &&
                    valid != nullptr;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        probe->valid &=
            valid[lane] ==
            ((probe->expected_active_mask & (1u << lane)) != 0u ?
                 -1 :
                 0);
    }
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if ((probe->expected_active_mask & (1u << lane)) == 0u) {
            continue;
        }
        floats[simd_host_accel_ray_tfar_field * lane_count + lane] =
            (lane & 1u) == 0u ? -1.0f : 1.0f;
    }
}

[[nodiscard]] bool check_accel_direct_packet_ir_shape() {
    static constexpr auto width = 8u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("accel_direct_packet_shape");
    auto *accel = kernel->create_resource_argument(Type::of<Accel>());
    auto *ray = kernel->create_value_argument(Type::of<Ray>());
    auto *visibility = kernel->create_value_argument(Type::of<uint32_t>());
    auto *entry = kernel->create_body_block();
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    static_cast<void>(builder.call(
        Type::of<SurfaceHit>(),
        xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST,
        {accel, ray, visibility}));
    builder.return_void();
    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-accel-direct-packet-shape", *context);
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width,
        "simd_accel_direct_packet_shape");
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(ir.find("accel.rayhit.packet") != std::string::npos);
    CHECK(ir.find("accel.valid") == std::string::npos);
    CHECK(ir.find("accel.closest.ids") == std::string::npos);
    CHECK(ir.find("accel.closest.values") == std::string::npos);
    CHECK(count_occurrences(ir, "call void %") == 1u);
    return true;
}

[[nodiscard]] bool run_accel_direct_packet_codegen_case(
    uint32_t width, uint32_t active_lanes) {
    Kernel1D kernel = [](
                          AccelVar accel,
                          BufferVar<Ray> rays,
                          BufferUInt visibility,
                          BufferVar<SurfaceHit> hits,
                          BufferUInt any_hits) noexcept {
        auto index = dispatch_x();
        auto ray = rays.read(index);
        auto options = AccelTraceOptions{
            .visibility_mask = visibility.read(index)};
        hits.write(index, accel.intersect(ray, options));
        any_hits.write(
            index, cast<uint>(accel.intersect_any(ray, options)));
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_accel_direct_packet_w" + std::to_string(width));
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        SIMDHostBufferView rays;
        SIMDHostBufferView visibility;
        SIMDHostBufferView hits;
        SIMDHostBufferView any_hits;
    };
    CHECK(compiled.argument_buffer_size == sizeof(Arguments));
    std::array<Ray, 16u> rays{};
    std::array<uint32_t, 16u> visibility{};
    std::array<SurfaceHit, 16u> hits{};
    std::array<uint32_t, 16u> any_hits{};
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        rays[lane] = Ray{
            .compressed_origin = {1.0f, 2.0f, 3.0f},
            .compressed_t_min = 4.0f,
            .compressed_direction = {5.0f, 6.0f, 7.0f},
            .compressed_t_max = 8.0f,
        };
        visibility[lane] = 0x5au;
    }
    AccelDirectPacketProbe probe{
        .expected_lane_count = width,
        .expected_active_mask =
            (1u << active_lanes) - 1u,
    };
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .trace_closest = accel_direct_closest_probe,
            .trace_any = accel_direct_any_probe,
        },
        .rays = {rays.data(), sizeof(rays)},
        .visibility = {visibility.data(), sizeof(visibility)},
        .hits = {hits.data(), sizeof(hits)},
        .any_hits = {any_hits.data(), sizeof(any_hits)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    auto config = launch_1d(active_lanes, width);
    entry(&arguments, nullptr, &config, width);
    if (!probe.valid) {
        std::cerr << "direct packet failure code: "
                  << probe.failure_code << '\n';
    }
    CHECK(probe.valid);
    CHECK(probe.closest_calls == 1u);
    CHECK(probe.any_calls == 1u);
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        CHECK(hits[lane].inst == 10u + lane);
        CHECK(hits[lane].prim == 20u + lane);
        CHECK(hits[lane].bary.x ==
              0.125f + static_cast<float>(lane));
        CHECK(hits[lane].bary.y ==
              0.25f + static_cast<float>(lane));
        CHECK(hits[lane].committed_ray_t ==
              30.0f + static_cast<float>(lane));
        CHECK(any_hits[lane] == ((lane & 1u) == 0u ? 1u : 0u));
    }
    return true;
}

[[nodiscard]] bool run_accel_direct_sparse_packet_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_mask = 0x55u;
    Kernel1D kernel = [](
                          AccelVar accel,
                          BufferVar<Ray> rays,
                          BufferUInt visibility,
                          BufferVar<SurfaceHit> hits,
                          BufferUInt any_hits) noexcept {
        auto index = dispatch_x();
        if_(index % 2u == 0u, [&] {
            auto ray = rays.read(index);
            auto options = AccelTraceOptions{
                .visibility_mask = visibility.read(index)};
            hits.write(index, accel.intersect(ray, options));
            any_hits.write(
                index, cast<uint>(accel.intersect_any(ray, options)));
        });
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_accel_direct_sparse_packet_w8");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    struct alignas(16) Arguments {
        SIMDHostAccelView accel;
        SIMDHostBufferView rays;
        SIMDHostBufferView visibility;
        SIMDHostBufferView hits;
        SIMDHostBufferView any_hits;
    };
    CHECK(compiled.argument_buffer_size == sizeof(Arguments));
    std::array<Ray, width> rays{};
    std::array<uint32_t, width> visibility{};
    std::array<SurfaceHit, width> hits{};
    std::array<uint32_t, width> any_hits{};
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        rays[lane] = Ray{
            .compressed_origin = {1.0f, 2.0f, 3.0f},
            .compressed_t_min = 4.0f,
            .compressed_direction = {5.0f, 6.0f, 7.0f},
            .compressed_t_max = 8.0f,
        };
        visibility[lane] = 0x5au;
        hits[lane] = SurfaceHit{
            .inst = 0xdeadbeefu,
            .prim = 0xcafebabeu,
            .bary = {-1.0f, -2.0f},
            .committed_ray_t = -3.0f,
        };
        any_hits[lane] = 0xabcdef01u;
    }
    AccelDirectPacketProbe probe{
        .expected_lane_count = width,
        .expected_active_mask = active_mask,
    };
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .trace_closest = accel_direct_closest_probe,
            .trace_any = accel_direct_any_probe,
        },
        .rays = {rays.data(), sizeof(rays)},
        .visibility = {visibility.data(), sizeof(visibility)},
        .hits = {hits.data(), sizeof(hits)},
        .any_hits = {any_hits.data(), sizeof(any_hits)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    auto config = launch_1d(width, width);
    entry(&arguments, nullptr, &config, width);
    if (!probe.valid) {
        std::cerr << "sparse direct packet failure code: "
                  << probe.failure_code << '\n';
    }
    CHECK(probe.valid);
    CHECK(probe.closest_calls == 1u);
    CHECK(probe.any_calls == 1u);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        if ((active_mask & (1u << lane)) != 0u) {
            CHECK(hits[lane].inst == 10u + lane);
            CHECK(hits[lane].prim == 20u + lane);
            CHECK(hits[lane].bary.x ==
                  0.125f + static_cast<float>(lane));
            CHECK(hits[lane].bary.y ==
                  0.25f + static_cast<float>(lane));
            CHECK(hits[lane].committed_ray_t ==
                  30.0f + static_cast<float>(lane));
            CHECK(any_hits[lane] == 1u);
        } else {
            CHECK(hits[lane].inst == 0xdeadbeefu);
            CHECK(hits[lane].prim == 0xcafebabeu);
            CHECK(hits[lane].bary.x == -1.0f);
            CHECK(hits[lane].bary.y == -2.0f);
            CHECK(hits[lane].committed_ray_t == -3.0f);
            CHECK(any_hits[lane] == 0xabcdef01u);
        }
    }
    return true;
}

[[nodiscard]] bool run_accel_direct_packet_codegen() {
    return check_accel_direct_packet_ir_shape() &&
           run_accel_direct_packet_codegen_case(1u, 1u) &&
           run_accel_direct_packet_codegen_case(2u, 1u) &&
           run_accel_direct_packet_codegen_case(4u, 3u) &&
           run_accel_direct_packet_codegen_case(8u, 5u) &&
           run_accel_direct_packet_codegen_case(16u, 3u) &&
           run_accel_direct_sparse_packet_codegen();
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
                        state->procedural_cursor_valid == 0u &&
                        state->candidate_batch_initialized == 0u &&
                        state->procedural_batch_initialized == 0u &&
                        state->committed.inst == ~0u &&
                        state->committed.prim == ~0u &&
                        std::bit_cast<uint32_t>(
                            state->committed.bary[0u]) == 0u &&
                        std::bit_cast<uint32_t>(
                            state->committed.bary[1u]) == 0u &&
                        state->committed.kind == 0u &&
                        std::bit_cast<uint32_t>(
                            state->committed.t) == 0u;
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
    bool expect_wide, bool disable_lazy_batch_init = false,
    bool disable_packed_init = false) {
    ScopedEnvironmentVariable lazy_batch_init{
        "LUISA_SIMD_DISABLE_RAY_QUERY_LAZY_BATCH_INIT",
        disable_lazy_batch_init ? "1" : nullptr};
    ScopedEnvironmentVariable packed_init{
        "LUISA_SIMD_DISABLE_RAY_QUERY_PACKED_INIT",
        disable_packed_init ? "1" : nullptr};
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
        *llvm_module, *lowered.function, width, name,
        false, {}, true, false);
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
    auto scatter_calls =
        count_occurrences(ir, "call void @llvm.masked.scatter");
    auto expect_packed = width >= 4u && !disable_packed_init;
    CHECK(scatter_calls ==
          (width == 2u || disable_lazy_batch_init ? 37u : 31u) -
              (expect_packed ? 5u : 0u));
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
    return run_ray_query_packet_codegen_case(1u, 1u, false) &&
           run_ray_query_packet_codegen_case(2u, 2u, false) &&
           run_ray_query_packet_codegen_case(4u, 3u, false) &&
           run_ray_query_packet_codegen_case(8u, 5u, true) &&
           run_ray_query_packet_codegen_case(8u, 5u, true, true) &&
           run_ray_query_packet_codegen_case(
               8u, 5u, true, false, true) &&
           run_ray_query_packet_codegen_case(16u, 3u, true);
}

struct RayQueryStatusProbe {
    uint32_t calls{0u};
    uint64_t mask{0u};
    uint32_t expected_lane_count{0u};
    SIMDHostAccelRayQueryProceed *expected_proceed{nullptr};
    bool valid{true};
};

[[nodiscard]] uint64_t ray_query_status_reference(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept {
    auto lane_mask = (uint64_t{1u} << lane_count) - 1u;
    auto active = active_mask_bits & lane_mask;
    auto status = uint64_t{0u};
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        auto bit = uint64_t{1u} << lane;
        if ((active & bit) == 0u) { continue; }
        auto *state = states[lane];
        if (state->terminated != 0u) {
            status |= bit << simd_host_ray_query_terminated_status_shift;
        }
        switch (static_cast<SIMDHostRayQueryCandidateKind>(
            state->candidate_kind)) {
            case SIMDHostRayQueryCandidateKind::surface:
                status |= bit << simd_host_ray_query_surface_status_shift;
                break;
            case SIMDHostRayQueryCandidateKind::procedural:
                status |= bit << simd_host_ray_query_procedural_status_shift;
                break;
            default: break;
        }
    }
    return status;
}

[[nodiscard]] bool run_ray_query_status_pack() {
    constexpr std::array widths{1u, 2u, 4u, 8u, 16u};
    for (auto width : widths) {
        std::array<SIMDHostRayQueryState, 16u> storage{};
        std::array<SIMDHostRayQueryState *, 16u> states{};
        for (auto lane = uint32_t{0u}; lane < width; lane++) {
            auto &state = storage[lane];
            state.terminated = (lane % 3u) == 0u ? 1u : 0u;
            state.candidate_kind = lane % 4u;
            states[lane] = &state;
        }
        auto lane_mask = (uint64_t{1u} << width) - 1u;
        constexpr std::array sparse_patterns{
            uint64_t{0u}, uint64_t{1u}, uint64_t{0xaaaau},
            uint64_t{0x8421u}, ~uint64_t{0u}};
        for (auto pattern : sparse_patterns) {
            auto mask = pattern & lane_mask;
            auto sparse_states = states;
            for (auto lane = uint32_t{0u}; lane < width; lane++) {
                if ((mask & (uint64_t{1u} << lane)) == 0u) {
                    sparse_states[lane] = nullptr;
                }
            }
            auto input_mask = mask | (uint64_t{1u} << 63u);
            CHECK(simd_host_ray_query_pack_status(
                      width, input_mask, sparse_states.data()) ==
                  ray_query_status_reference(
                      width, input_mask, sparse_states.data()));
        }
        CHECK(simd_host_ray_query_pack_status(
                  width, lane_mask, states.data()) ==
              ray_query_status_reference(
                  width, lane_mask, states.data()));
        if (width == 16u) {
            for (auto pattern : sparse_patterns) {
                auto mask = pattern & lane_mask;
                auto sparse_states = states;
                for (auto lane = uint32_t{0u}; lane < width; lane++) {
                    if ((mask & (uint64_t{1u} << lane)) == 0u) {
                        sparse_states[lane] = nullptr;
                    }
                }
                auto input_mask = mask | (uint64_t{1u} << 63u);
                CHECK(simd_host_ray_query_pack_procedural_wide_status(
                          width, input_mask, sparse_states.data()) ==
                      ray_query_status_reference(
                          width, input_mask, sparse_states.data()));
            }
        }
    }
    CHECK(!simd_host_ray_query_use_procedural_wide_status(
        8u, true, true));
    CHECK(!simd_host_ray_query_use_procedural_wide_status(
        16u, false, false));
    CHECK(!simd_host_ray_query_use_procedural_wide_status(
        16u, false, true));
    CHECK(!simd_host_ray_query_use_procedural_wide_status(
        16u, true, false));
    CHECK(simd_host_ray_query_use_procedural_wide_status(
        16u, true, true));
    return true;
}

[[nodiscard]] uint64_t ray_query_status_probe_impl(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    SIMDHostRayQueryState *first_state = nullptr;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if (states[lane] != nullptr) {
            first_state = states[lane];
            break;
        }
    }
    if (first_state == nullptr) { return 0u; }
    auto *probe = static_cast<RayQueryStatusProbe *>(
        first_state->accel);
    probe->calls++;
    probe->mask |= active_mask_bits;
    probe->valid &= lane_count == probe->expected_lane_count;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        auto active =
            (active_mask_bits & (uint64_t{1u} << lane)) != 0u;
        if (!active) {
            probe->valid &= states[lane] == nullptr;
            continue;
        }
        auto *state = states[lane];
        auto valid_state =
            state != nullptr && state->accel == probe &&
            state->proceed == probe->expected_proceed;
        probe->valid &= valid_state;
        if (!valid_state) { std::abort(); }
        state->terminated = 0u;
        state->candidate_kind =
            lane % 2u == 0u ?
                static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::surface) :
                static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::procedural);
    }
    return simd_host_ray_query_pack_status(
        lane_count, active_mask_bits, states);
}

void ray_query_status_plain_probe(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    (void)ray_query_status_probe_impl(
        lane_count, active_mask_bits, states);
}

[[nodiscard]] uint64_t ray_query_status_probe(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    return ray_query_status_probe_impl(
        lane_count, active_mask_bits, states);
}

void ray_query_status_mismatched_plain_probe(
    uint32_t, uint64_t,
    SIMDHostRayQueryState *const *) noexcept {}

[[nodiscard]] bool run_ray_query_status_pairing_fail_closed() noexcept {
#if defined(__unix__) || defined(__APPLE__)
    auto child = fork();
    if (child < 0) { return false; }
    if (child == 0) {
        const rlimit no_core{0u, 0u};
        static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
        RayQueryStatusProbe probe{
            .expected_lane_count = 2u,
            .expected_proceed = ray_query_status_plain_probe,
        };
        std::array<SIMDHostRayQueryState, 2u> state_storage{};
        state_storage[0u].accel = &probe;
        state_storage[0u].proceed = ray_query_status_plain_probe;
        state_storage[1u].accel = &probe;
        state_storage[1u].proceed =
            ray_query_status_mismatched_plain_probe;
        std::array<SIMDHostRayQueryState *, 2u> states{
            &state_storage[0u], &state_storage[1u]};
        static_cast<void>(ray_query_status_probe(
            2u, 0b11u, states.data()));
        _exit(EXIT_SUCCESS);
    }
    auto status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) { return false; }
    }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
#else
    return true;
#endif
}

[[nodiscard]] bool run_ray_query_status_cache_codegen_case(
    uint32_t width, uint32_t active_lanes,
    bool disable_cache, bool disable_coloring = false,
    bool disable_state_handles = false,
    bool disable_status_pairing = false) {
    ScopedEnvironmentVariable cache{
        "LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CACHE",
        disable_cache ? "1" : nullptr};
    ScopedEnvironmentVariable coloring{
        "LUISA_SIMD_DISABLE_RAY_QUERY_SCRATCH_COLORING",
        disable_coloring ? "1" : nullptr};
    ScopedEnvironmentVariable state_handles{
        "LUISA_SIMD_DISABLE_RAY_QUERY_STATE_HANDLE_CACHE",
        disable_state_handles ? "1" : nullptr};
    ScopedEnvironmentVariable status_pairing{
        "LUISA_SIMD_DISABLE_RAY_QUERY_STATUS_CALLBACK_PAIRING",
        disable_status_pairing ? "1" : nullptr};
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("ray_query_status_cache");
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
    auto visibility_value = uint32_t{0xffu};
    auto split_value = std::max(active_lanes / 2u, 1u);
    auto *visibility = module.create_constant(
        Type::of<uint32_t>(), &visibility_value);
    auto *split = module.create_constant(
        Type::of<uint32_t>(), &split_value);
    auto *query_type = Type::custom("LC_RayQueryAll");
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero});
    auto *query_value = builder.call(
        query_type, xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL,
        {accel, ray, visibility});
    auto *query = builder.alloca_local(query_type);
    builder.store(query, query_value);
    auto *condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_LESS,
        {x, split});
    builder.cond_br(condition, left, right);
    builder.set_insertion_point(left);
    builder.call(
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
        {query});
    builder.call(
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE,
        {query});
    builder.br(merge);
    builder.set_insertion_point(right);
    builder.call(
        xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED,
        {query});
    builder.br(merge);
    builder.set_insertion_point(merge);
    auto *terminated = builder.call(
        Type::of<bool>(),
        xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
        {query});
    auto *surface = builder.call(
        Type::of<bool>(),
        xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE,
        {query});
    auto *procedural = builder.call(
        Type::of<bool>(),
        xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE,
        {query});
    auto *terminated_u32 = builder.cast_(
        Type::of<uint32_t>(), xir::CastOp::STATIC_CAST, terminated);
    auto *surface_u32 = builder.cast_(
        Type::of<uint32_t>(), xir::CastOp::STATIC_CAST, surface);
    auto *procedural_u32 = builder.cast_(
        Type::of<uint32_t>(), xir::CastOp::STATIC_CAST, procedural);
    auto *result = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::AGGREGATE,
        {terminated_u32, surface_u32, procedural_u32, x});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, x, result});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-ray-query-status-cache", *context);
    auto name = std::string{"simd_ray_query_status_cache_"} +
                std::to_string(width) +
                (disable_cache ? "_disabled" : "_enabled") +
                (disable_coloring ? "_uncolored" : "_colored") +
                (disable_state_handles ? "_handles_disabled" :
                                         "_handles_enabled") +
                (disable_status_pairing ? "_pairing_disabled" :
                                          "_pairing_enabled");
    auto codegen = lower_schedule_to_llvm(
        *llvm_module, *lowered.function, width, name);
    if (!codegen.succeeded()) {
        std::cerr << codegen.error << '\n';
        return false;
    }
    CHECK(codegen.ray_query_count == 1u);
    CHECK(codegen.ray_query_scratch_slot_count == 1u);
    auto expect_cache =
        !disable_cache && !disable_coloring && width >= 4u;
    CHECK(codegen.ray_query_status_slot_count ==
          (expect_cache ? 1u : 0u));
    auto expect_state_handles = expect_cache && !disable_state_handles;
    CHECK(codegen.ray_query_state_handle_slot_count ==
          (expect_state_handles ? 1u : 0u));
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK((ir.find("ray.query.status.slot.0") != std::string::npos) ==
          expect_cache);
    CHECK((ir.find("ray.query.state.handles.slot.0") != std::string::npos) ==
          expect_state_handles);
    if (expect_state_handles) {
        auto state_alloca = line_containing(
            ir, "ray.query.state.handles.slot.0 = alloca");
        auto state_load = line_containing(
            ir, "ray.query.cached.state.handles = load");
        auto callback_load = line_containing(
            ir, "ray.query.status.callbacks = load");
        CHECK(state_alloca.find("align 8") != std::string_view::npos);
        CHECK(state_load.find("align 8") != std::string_view::npos);
        CHECK(callback_load.find("align 8") != std::string_view::npos);
    }
    auto gather_count = count_occurrences(ir, "llvm.masked.gather");
    auto expect_status_pairing =
        expect_cache && !disable_status_pairing;
    CHECK((ir.find("ray.query.proceed.callback.mismatch") ==
           std::string::npos) == expect_status_pairing);
    CHECK((ir.find("ray.query.proceed.status.callback.mismatch") !=
           std::string::npos) == expect_cache);
    auto expected_gather_count = expect_cache ?
                                     (expect_state_handles ? 3u : 6u) :
                                     13u;
    if (expect_status_pairing) {
        expected_gather_count -= expect_state_handles ? 3u : 2u;
    }
    if (gather_count != expected_gather_count) {
        std::cerr << "ray-query gather count W" << width
                  << " cache=" << expect_cache
                  << " handles=" << expect_state_handles
                  << ": " << gather_count << " expected "
                  << expected_gather_count << '\n';
    }
    CHECK(gather_count == expected_gather_count);
    CHECK(count_occurrences(ir, expect_cache ? "call i64 %" : "call void %") == 2u);

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
    RayQueryStatusProbe probe{
        .expected_lane_count = width,
        .expected_proceed = ray_query_status_plain_probe,
    };
    SIMDHostAccelInstanceTable instance_table{
        .ray_query_proceed_status = ray_query_status_probe,
        .ray_query_proceed_wide_status = ray_query_status_probe};
    luisa::vector<luisa::uint4> values(width);
    std::fill(
        values.begin(), values.end(),
        luisa::make_uint4(0xdeadbeefu));
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .instances = &instance_table,
            .ray_query_proceed = ray_query_status_plain_probe,
            .ray_query_proceed_wide = ray_query_status_plain_probe,
        },
        .ray = {
            .compressed_origin = {1.0f, 2.0f, 3.0f},
            .compressed_t_min = 0.25f,
            .compressed_direction = {4.0f, 5.0f, 6.0f},
            .compressed_t_max = 7.0f,
        },
        .output = {values.data(), sizeof(luisa::uint4) * values.size()},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.calls == (active_lanes > 1u ? 2u : 1u));
    CHECK(probe.mask == (uint64_t{1u} << active_lanes) - 1u);
    for (auto lane = uint32_t{0u}; lane < width; lane++) {
        auto expected = lane < active_lanes ?
                            luisa::make_uint4(
                                lane < split_value,
                                lane % 2u == 0u,
                                lane % 2u == 1u, lane) :
                            luisa::make_uint4(0xdeadbeefu);
        CHECK(luisa::all(values[lane] == expected));
    }
    return true;
}

[[nodiscard]] bool run_ray_query_status_cache_codegen() {
    if (!run_ray_query_status_pairing_fail_closed()) { return false; }
    for (auto disable_cache : {false, true}) {
        if (!run_ray_query_status_cache_codegen_case(
                1u, 1u, disable_cache) ||
            !run_ray_query_status_cache_codegen_case(
                2u, 2u, disable_cache) ||
            !run_ray_query_status_cache_codegen_case(
                4u, 3u, disable_cache) ||
            !run_ray_query_status_cache_codegen_case(
                8u, 5u, disable_cache) ||
            !run_ray_query_status_cache_codegen_case(
                16u, 3u, disable_cache)) {
            return false;
        }
    }
    return run_ray_query_status_cache_codegen_case(
               8u, 5u, false, true) &&
           run_ray_query_status_cache_codegen_case(
               4u, 3u, false, false, true) &&
           run_ray_query_status_cache_codegen_case(
               8u, 5u, false, false, true) &&
           run_ray_query_status_cache_codegen_case(
               16u, 3u, false, false, true) &&
           run_ray_query_status_cache_codegen_case(
               4u, 3u, false, false, false, true) &&
           run_ray_query_status_cache_codegen_case(
               8u, 5u, false, false, false, true) &&
           run_ray_query_status_cache_codegen_case(
               16u, 3u, false, false, false, true);
}

struct RayQueryScratchProbe {
    uint32_t calls{0u};
    uint64_t mask{0u};
    SIMDHostAccelRayQueryProceed *expected_proceed{nullptr};
    bool valid{true};
};

[[nodiscard]] uint64_t ray_query_scratch_probe_impl(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    SIMDHostRayQueryState *first_state = nullptr;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if (states[lane] != nullptr) {
            first_state = states[lane];
            break;
        }
    }
    if (first_state == nullptr) { return 0u; }
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
                        state->proceed == probe->expected_proceed &&
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
        state->terminated = state->visibility_mask == 0x31u ? 1u : 0u;
        state->candidate_kind =
            state->visibility_mask == 0x72u ?
                static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::surface) :
                static_cast<uint32_t>(
                    SIMDHostRayQueryCandidateKind::none);
    }
    return simd_host_ray_query_pack_status(
        lane_count, active_mask_bits, states);
}

void ray_query_scratch_plain_probe(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    (void)ray_query_scratch_probe_impl(
        lane_count, active_mask_bits, states);
}

[[nodiscard]] uint64_t ray_query_scratch_probe(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) {
    return ray_query_scratch_probe_impl(
        lane_count, active_mask_bits, states);
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
    CHECK(codegen.ray_query_status_slot_count == 0u);
    CHECK(codegen.ray_query_state_handle_slot_count == 0u);
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
    RayQueryScratchProbe probe{
        .expected_proceed = ray_query_scratch_plain_probe};
    SIMDHostAccelInstanceTable instance_table{
        .ray_query_proceed_status = ray_query_scratch_probe,
        .ray_query_proceed_wide_status = ray_query_scratch_probe};
    std::array<luisa::uint4, width> values{};
    values.fill(luisa::make_uint4(0xdeadbeefu));
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .instances = &instance_table,
            .ray_query_proceed = ray_query_scratch_plain_probe,
            .ray_query_proceed_wide = ray_query_scratch_plain_probe,
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
        auto *terminated = builder.call(
            Type::of<bool>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query});
        auto *surface = builder.call(
            Type::of<bool>(),
            xir::RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE,
            {query});
        auto *terminated_u32 = builder.cast_(
            Type::of<uint32_t>(), xir::CastOp::STATIC_CAST,
            terminated);
        auto *surface_u32 = builder.cast_(
            Type::of<uint32_t>(), xir::CastOp::STATIC_CAST,
            surface);
        builder.br(merge);
        return std::array<xir::Value *, 3u>{
            inst, terminated_u32, surface_u32};
    };
    auto result1 = emit_query(left, visibility1);
    auto result2 = emit_query(right, visibility2);
    builder.set_insertion_point(merge);
    auto *inst = builder.phi(
        Type::of<uint32_t>(), {{result1[0u], left}, {result2[0u], right}});
    auto *terminated = builder.phi(
        Type::of<uint32_t>(),
        {{result1[1u], left}, {result2[1u], right}});
    auto *surface = builder.phi(
        Type::of<uint32_t>(),
        {{result1[2u], left}, {result2[2u], right}});
    auto *metadata = builder.call(
        Type::of<luisa::uint4>(), xir::ArithmeticOp::AGGREGATE,
        {inst, x, terminated, surface});
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
    CHECK(codegen.ray_query_status_slot_count == 1u);
    CHECK(codegen.ray_query_state_handle_slot_count == 1u);
    CHECK(!::llvm::verifyModule(*llvm_module, &::llvm::errs()));
    std::string ir;
    ::llvm::raw_string_ostream stream{ir};
    llvm_module->print(stream, nullptr);
    stream.flush();
    CHECK(count_occurrences(ir, "alloca [9728 x i8]") == 1u);
    CHECK(count_occurrences(ir, "ray.query.status.slot.0") >= 1u);
    CHECK(count_occurrences(
              ir, "ray.query.state.handles.slot.0") >= 1u);
    CHECK(count_occurrences(ir, "call i64 %") == 2u);

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
    RayQueryScratchProbe probe{
        .expected_proceed = ray_query_scratch_plain_probe};
    SIMDHostAccelInstanceTable instance_table{
        .ray_query_proceed_status = ray_query_scratch_probe,
        .ray_query_proceed_wide_status = ray_query_scratch_probe};
    std::array<luisa::uint4, width> values{};
    values.fill(luisa::make_uint4(0xdeadbeefu));
    Arguments arguments{
        .accel = {
            .accel = &probe,
            .instances = &instance_table,
            .ray_query_proceed = ray_query_scratch_plain_probe,
            .ray_query_proceed_wide = ray_query_scratch_plain_probe,
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
                                lane,
                                lane < two_value ? 1u : 0u,
                                lane < two_value ? 0u : 1u) :
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
    std::array<uint64_t, 4u> masks{};
    std::array<std::array<uint32_t, 8u>, 4u> slots{};
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
    auto gradient_call = call == 1u;
    probe->valid &= call < probe->masks.size() && slot_count == 2u &&
                    dimension == 2u && lane_count == 8u &&
                    u != nullptr && v != nullptr && w != nullptr;
    if (gradient_call) {
        probe->valid &= sampler_codes != nullptr && levels != nullptr;
        for (auto lane = uint32_t{5u}; lane < lane_count; lane++) {
            probe->valid &= sampler_codes[lane] == 0u &&
                            levels[lane] == 0.0f;
        }
    } else {
        probe->valid &= sampler_codes == nullptr && levels == nullptr &&
                        u != nullptr;
    }
    if (call >= probe->masks.size()) { return; }
    probe->masks[call] = active_mask_bits;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        if ((active_mask_bits & (uint64_t{1u} << lane)) == 0u) {
            continue;
        }
        probe->slots[call][lane] = slot_indices[lane];
        if (gradient_call) {
            probe->valid &= sampler_codes[lane] ==
                                Sampler::linear_linear_edge().code() &&
                            levels[lane] == 1.25f;
        }
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
    auto ddx_value = luisa::make_float2(0.5f, 0.0f);
    auto ddy_value = luisa::make_float2(0.0f, 0.5f);
    auto minimum_level_value = 1.25f;
    auto filter_value = static_cast<uint32_t>(
        Sampler::Filter::LINEAR_LINEAR);
    auto address_value = static_cast<uint32_t>(
        Sampler::Address::EDGE);
    auto *ddx = module.create_constant(
        Type::of<luisa::float2>(), &ddx_value);
    auto *ddy = module.create_constant(
        Type::of<luisa::float2>(), &ddy_value);
    auto *minimum_level = module.create_constant(
        Type::of<float>(), &minimum_level_value);
    auto *filter = module.create_constant(
        Type::of<uint32_t>(), &filter_value);
    auto *address = module.create_constant(
        Type::of<uint32_t>(), &address_value);
    auto *gradient_pixel = builder.call(
        Type::of<luisa::float4>(),
        xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER,
        {bindless, slot, uv, ddx, ddy, minimum_level, filter, address});
    auto gradient_output_offset_value = uint32_t{16u};
    auto *gradient_output_offset = module.create_constant(
        Type::of<uint32_t>(), &gradient_output_offset_value);
    auto *gradient_output_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {x, gradient_output_offset});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, gradient_output_index, gradient_pixel});
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
    auto *x_float = builder.cast_(
        Type::of<float>(), xir::CastOp::STATIC_CAST, x);
    auto *varying_uv = builder.call(
        Type::of<luisa::float2>(), xir::ArithmeticOp::AGGREGATE,
        {x_float, x_float});
    auto *direct_pixel = builder.call(
        Type::of<luisa::float4>(),
        xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE,
        {bindless, zero, varying_uv});
    auto direct_output_offset_value = uint32_t{24u};
    auto *direct_output_offset = module.create_constant(
        Type::of<uint32_t>(), &direct_output_offset_value);
    auto *direct_output_index = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::BINARY_ADD,
        {x, direct_output_offset});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, direct_output_index, direct_pixel});
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
    CHECK(ir.find("bindless.texture.sample.safe.gradient") != std::string::npos);
    CHECK(ir.find("llvm.masked.gather") != std::string::npos);
    CHECK(ir.find("__luisa_cpu_native_log2_f32_v8_u35") !=
          std::string::npos);
    CHECK(ir.find("bindless.texture.sample.gradient.levels") ==
          std::string::npos);
    CHECK(ir.find("bindless.texture.sample.lane") == std::string::npos);
    CHECK(ir.find("bindless.uniform.callback.mask") != std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.direct") != std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.sample.mask") !=
          std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.reduced.valid") !=
          std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.wide.eligible") !=
          std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.narrow") != std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.v00") != std::string::npos);
    CHECK(ir.find("bindless.texture.byte1.callback") != std::string::npos);

    LLVMJIT jit{true};
    CHECK(jit.succeeded());
    auto assembly = jit.emit_assembly_copy(*llvm_module);
    std::transform(
        assembly.begin(), assembly.end(), assembly.begin(),
        [](unsigned char c) noexcept {
            return static_cast<char>(std::tolower(c));
        });
    CHECK(assembly.find("log2f") == std::string::npos);
    CHECK(assembly.find("_zgv") == std::string::npos);
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);
    CHECK(!jit.object().empty());

    BindlessTexturePacketProbe probe;
    std::array<SIMDHostBindlessSlot, 2u> slots{};
    for (auto &slot_descriptor : slots) {
        slot_descriptor.texture2d.texture = &probe;
        slot_descriptor.texture2d.metadata =
            simd_bindless_texture_metadata(
                Sampler::point_edge().code(), 4u, 4u, 1u);
    }
    std::array<luisa::float4, 32u> output_values{};
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
    CHECK(probe.calls == 4u);
    CHECK(probe.masks[0u] == 0x1fu);
    CHECK(probe.masks[1u] == 0x1fu);
    CHECK(probe.masks[2u] == 0x01u);
    CHECK(probe.masks[3u] == 0x1fu);
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
                             200.0f, 210.0f, 220.0f, 230.0f)));
        CHECK(luisa::all(output_values[16u + lane] ==
                         luisa::make_float4(
                             static_cast<float>(100u + expected_slot),
                             static_cast<float>(110u + expected_slot),
                             static_cast<float>(120u + expected_slot),
                             static_cast<float>(130u + expected_slot))));
        CHECK(luisa::all(output_values[24u + lane] ==
                         luisa::make_float4(
                             300.0f, 310.0f, 320.0f, 330.0f)));
    }
    return true;
}

struct BindlessUniformGradientProbe {
    bool valid{true};
    uint32_t calls{0u};
};

void bindless_uniform_gradient_probe(
    const SIMDHostBindlessSlot *slots, size_t slot_count,
    uint32_t dimension, uint32_t lane_count,
    uint64_t active_mask_bits, const uint32_t *slot_indices,
    const uint32_t *sampler_codes,
    const float *u, const float *v, const float *w,
    const float *levels, float *values) {
    auto *probe = static_cast<BindlessUniformGradientProbe *>(
        slots[0u].texture2d.texture);
    probe->calls++;
    probe->valid &= slot_count == 1u && dimension == 2u &&
                    lane_count == 8u && active_mask_bits == 0x1fu &&
                    slot_indices != nullptr && sampler_codes == nullptr &&
                    u != nullptr && v != nullptr && w != nullptr &&
                    levels != nullptr && values != nullptr;
    for (auto lane = uint32_t{0u}; lane < lane_count; lane++) {
        auto active = lane < 5u;
        probe->valid &= slot_indices[lane] == 0u;
        probe->valid &= levels[lane] == (active ? 1.25f : 0.0f);
        if (!active) { continue; }
        for (auto component = uint32_t{0u}; component < 4u;
             component++) {
            values[component * lane_count + lane] =
                static_cast<float>(100u + component * 10u + lane);
        }
    }
}

[[nodiscard]] bool run_bindless_uniform_gradient_lod_codegen() {
    static constexpr auto width = 8u;
    static constexpr auto active_lanes = 5u;
    xir::Module module;
    auto *kernel = module.create_kernel();
    kernel->set_name("bindless_uniform_gradient_lod");
    auto *bindless = kernel->create_resource_argument(
        Type::of<BindlessArray>());
    auto *output = kernel->create_resource_argument(
        Type::buffer(Type::of<luisa::float4>()));
    auto *entry_block = kernel->create_body_block();
    auto *dispatch_id = module.create_dispatch_id();
    auto *zero_u32 = module.create_constant_zero(Type::of<uint32_t>());
    auto zero_f32_value = 0.0f;
    auto *zero_f32 = module.create_constant(
        Type::of<float>(), &zero_f32_value);
    auto ddx_value = luisa::make_float2(0.5f, 0.0f);
    auto ddy_value = luisa::make_float2(0.0f, 0.5f);
    auto minimum_level_value = 1.25f;
    auto *ddx = module.create_constant(
        Type::of<luisa::float2>(), &ddx_value);
    auto *ddy = module.create_constant(
        Type::of<luisa::float2>(), &ddy_value);
    auto *minimum_level = module.create_constant(
        Type::of<float>(), &minimum_level_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto *x = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::EXTRACT,
        {dispatch_id, zero_u32});
    auto *x_f32 = builder.cast_(
        Type::of<float>(), xir::CastOp::STATIC_CAST, x);
    auto *uv = builder.call(
        Type::of<luisa::float2>(), xir::ArithmeticOp::AGGREGATE,
        {x_f32, zero_f32});
    auto *pixel = builder.call(
        Type::of<luisa::float4>(),
        xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL,
        {bindless, zero_u32, uv, ddx, ddy, minimum_level});
    builder.call(
        xir::ResourceWriteOp::BUFFER_WRITE,
        {output, x, pixel});
    builder.return_void();

    auto lowered = schedule::lower_xir_to_schedule(
        kernel, {.logical_warp_width = width});
    if (!lowered.succeeded()) {
        std::cerr << diagnostics_text(lowered);
        return false;
    }
    auto context = std::make_unique<::llvm::LLVMContext>();
    auto llvm_module = std::make_unique<::llvm::Module>(
        "simd-bindless-uniform-gradient-lod", *context);
    auto name = std::string{"simd_bindless_uniform_gradient_lod"};
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
    CHECK(count_occurrences(ir, "call float @llvm.log2.f32") == 1u);
    CHECK(ir.find("__luisa_cpu_native_log2") == std::string::npos);
    CHECK(ir.find("bindless.texture.sample.uniform.metadata") !=
          std::string::npos);
    CHECK(ir.find(
              "bindless.texture.sample.uniform.gradient.minimum.splat") !=
          std::string::npos);
    CHECK(ir.find("llvm.masked.gather") == std::string::npos);

    LLVMJIT jit;
    CHECK(jit.succeeded());
    CHECK(jit.add_module(std::move(llvm_module), std::move(context)));
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto function = reinterpret_cast<Entry *>(jit.lookup(name));
    CHECK(function != nullptr);

    BindlessUniformGradientProbe probe;
    std::array<SIMDHostBindlessSlot, 1u> slots{};
    slots[0u].texture2d.texture = &probe;
    slots[0u].texture2d.metadata = simd_bindless_texture_metadata(
        Sampler::point_edge().code(), 4u, 4u, 1u);
    std::array<luisa::float4, width> output_values{};
    Arguments arguments{
        .bindless = {
            .slots = slots.data(),
            .size = slots.size(),
            .sample_texture = bindless_uniform_gradient_probe,
        },
        .output = {output_values.data(), sizeof(output_values)},
    };
    auto config = launch_1d(active_lanes, width);
    function(&arguments, nullptr, &config, active_lanes);
    CHECK(probe.valid);
    CHECK(probe.calls == 1u);
    for (auto lane = uint32_t{0u}; lane < active_lanes; lane++) {
        CHECK(luisa::all(
            output_values[lane] ==
            luisa::make_float4(
                static_cast<float>(100u + lane),
                static_cast<float>(110u + lane),
                static_cast<float>(120u + lane),
                static_cast<float>(130u + lane))));
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

[[nodiscard]] bool run_ast_packet_batch_entry() {
    static constexpr auto width = 8u;
    static constexpr auto count = 22u;
    Kernel1D kernel = [](BufferUInt output) noexcept {
        set_block_size(32u, 1u, 1u);
        auto index = dispatch_id().x;
        output.write(index, index * 7u + 3u);
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_packet_batch_entry", false, false, 1u, true);
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.entry == nullptr);
    CHECK(compiled.packet_batch_entry != nullptr);
    std::array<uint32_t, count> output{};
    output.fill(0xdeadbeefu);
    alignas(16) SIMDHostBufferView argument{
        output.data(), sizeof(output)};
    using PacketBatchEntry = void(
        const void *, void *, SIMDPacketLaunchConfig *, uint32_t);
    auto packet_batch_entry = reinterpret_cast<PacketBatchEntry *>(
        compiled.packet_batch_entry);
    auto config = launch_1d(count, 32u);
    config.thread_index = 3u;
    packet_batch_entry(&argument, nullptr, &config, 4u);
    CHECK(config.thread_index == 3u + 3u * width);
    for (auto index = uint32_t{0u}; index < count; index++) {
        auto expected = index < 3u ? 0xdeadbeefu : index * 7u + 3u;
        CHECK(output[index] == expected);
    }

    return true;
}

[[nodiscard]] bool run_ast_aggregate_promotion() {
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferFloat4 output) noexcept {
        auto index = dispatch_id().x;
        auto index_f32 = cast<float>(index);
        Var<SIMDAggregatePromotionProbe> state =
            def<SIMDAggregatePromotionProbe>(
                index_f32 + 0.25f, index_f32 * 2.0f + 0.5f,
                1.0f, index + 100u);
        $if ((index & 1u) == 0u) {
            state.x += 10.0f;
            state.tag += 3u;
        }
        $else {
            state.y -= 4.0f;
            state.tag += 5u;
        };
        $for (iteration, 3u) {
            state.z += (cast<float>(iteration) + 0.5f) *
                       (index_f32 + 1.0f);
        };
        output.write(
            index,
            make_float4(state.x, state.y, state.z,
                        cast<float>(state.tag)));
    };

    auto compile = [&](uint32_t width, bool disable,
                       std::string_view name) {
        ScopedEnvironmentVariable setting{
            "LUISA_SIMD_DISABLE_AGGREGATE_PROMOTION",
            disable ? "1" : "0"};
        return compile_simd_kernel(
            kernel.function()->function(), width, name);
    };
    auto execute = [&](const SIMDCompiledKernel &compiled,
                       uint32_t width,
                       std::array<luisa::float4, count> &output) {
        output.fill(luisa::make_float4(-999.0f));
        alignas(16) SIMDHostBufferView argument{
            output.data(), sizeof(output)};
        using Entry = void(
            const void *, void *, const SIMDPacketLaunchConfig *,
            uint32_t);
        auto entry = reinterpret_cast<Entry *>(compiled.entry);
        auto config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += width) {
            config.thread_index = first;
            entry(&argument, nullptr, &config, width);
        }
    };
    for (auto width : {1u, 2u, 4u, 8u, 16u}) {
        auto suffix = "_w" + std::to_string(width);
        auto promoted = compile(
            width, false, "simd_ast_aggregate_promoted" + suffix);
        auto baseline = compile(
            width, true, "simd_ast_aggregate_baseline" + suffix);
        CHECK(promoted.succeeded());
        CHECK(baseline.succeeded());
        CHECK(promoted.decomposed_aggregate_alloca_count != 0u);
        CHECK(promoted.inserted_aggregate_leaf_alloca_count >
              promoted.decomposed_aggregate_alloca_count);
        CHECK(baseline.decomposed_aggregate_alloca_count == 0u);
        CHECK(baseline.inserted_aggregate_leaf_alloca_count == 0u);

        std::array<luisa::float4, count> promoted_output{};
        std::array<luisa::float4, count> baseline_output{};
        execute(promoted, width, promoted_output);
        execute(baseline, width, baseline_output);
        for (auto i = uint32_t{0u}; i < count; i++) {
            auto i_f32 = static_cast<float>(i);
            auto even = (i & 1u) == 0u;
            auto expected = luisa::make_float4(
                i_f32 + 0.25f + (even ? 10.0f : 0.0f),
                i_f32 * 2.0f + 0.5f -
                    (even ? 0.0f : 4.0f),
                1.0f + 4.5f * (i_f32 + 1.0f),
                static_cast<float>(
                    i + 100u + (even ? 3u : 5u)));
            CHECK(luisa::all(promoted_output[i] == expected));
            CHECK(luisa::all(baseline_output[i] == expected));
            CHECK(luisa::all(
                promoted_output[i] == baseline_output[i]));
        }
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

[[nodiscard]] bool run_ast_predicated_select_ladder() {
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        auto selector = index & 7u;
        $uint value = 71u;
        $if (selector == 1u) {
            value = 11u;
        }
        $elif (selector == 2u) {
            value = 22u;
        }
        $elif (selector == 3u) {
            value = 33u;
        }
        $elif (selector == 4u) {
            value = 44u;
        }
        $elif (selector == 5u) {
            value = 55u;
        };
        output.write(index, value);
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    constexpr std::array widths{2u, 4u, 8u, 16u};
    for (auto width : widths) {
        auto compiled = compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_predicated_select_ladder");
        if (!compiled.succeeded()) {
            for (auto &&diagnostic : compiled.diagnostics) {
                std::cerr << diagnostic << '\n';
            }
            return false;
        }
        auto refinement_enabled = width == 4u || width == 8u;
        CHECK((compiled.predicated_forwarding_block_count != 0u) ==
              refinement_enabled);
        CHECK((compiled.predicated_forwarded_phi_count != 0u) ==
              refinement_enabled);

        std::array<uint32_t, count> output{};
        output.fill(0xdeadbeefu);
        alignas(16) SIMDHostBufferView argument{
            output.data(), sizeof(output)};
        auto *entry = reinterpret_cast<Entry *>(compiled.entry);
        CHECK(entry != nullptr);
        auto config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += width) {
            config.thread_index = first;
            entry(&argument, nullptr, &config, width);
        }
        for (auto index = uint32_t{0u}; index < count; index++) {
            auto expected = 71u;
            switch (index & 7u) {
                case 1u: expected = 11u; break;
                case 2u: expected = 22u; break;
                case 3u: expected = 33u; break;
                case 4u: expected = 44u; break;
                case 5u: expected = 55u; break;
                default: break;
            }
            CHECK(output[index] == expected);
        }

        if (refinement_enabled) {
            ScopedEnvironmentVariable disable_refinement{
                "LUISA_SIMD_DISABLE_PREDICATED_IF_REFINEMENT", "1"};
            auto oracle = compile_simd_kernel(
                kernel.function()->function(), width,
                "simd_ast_scheduled_select_ladder");
            CHECK(oracle.succeeded());
            CHECK(oracle.predicated_forwarding_block_count == 0u);
            CHECK(oracle.predicated_forwarded_phi_count == 0u);
            std::array<uint32_t, count> oracle_output{};
            oracle_output.fill(0xdeadbeefu);
            alignas(16) SIMDHostBufferView oracle_argument{
                oracle_output.data(), sizeof(oracle_output)};
            auto *oracle_entry =
                reinterpret_cast<Entry *>(oracle.entry);
            CHECK(oracle_entry != nullptr);
            for (auto first = uint32_t{0u}; first < 16u;
                 first += width) {
                config.thread_index = first;
                oracle_entry(
                    &oracle_argument, nullptr, &config, width);
            }
            CHECK(output == oracle_output);
        }
    }
    return true;
}

[[nodiscard]] bool run_ast_deep_predicated_select_ladder() {
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferFloat4 output) noexcept {
        auto index = dispatch_id().x;
        auto selector = index & 7u;
        Var<float3> value = make_float3(71.0f, 72.0f, 73.0f);
        $if (selector == 1u) {
            value = make_float3(11.0f, 12.0f, 13.0f);
        }
        $elif (selector == 2u) {
            value = make_float3(21.0f, 22.0f, 23.0f);
        }
        $elif (selector == 3u) {
            value = make_float3(31.0f, 32.0f, 33.0f);
        }
        $elif (selector == 4u) {
            value = make_float3(41.0f, 42.0f, 43.0f);
        }
        $elif (selector == 5u) {
            value = make_float3(51.0f, 52.0f, 53.0f);
        };
        output.write(index, make_float4(value, 1.0f));
    };
    auto compile = [&](uint32_t width, bool disable_deep,
                       bool disable_wide) {
        ScopedEnvironmentVariable deep_setting{
            "LUISA_SIMD_DISABLE_DEEP_PREDICATED_IF_REFINEMENT",
            disable_deep ? "1" : "0"};
        ScopedEnvironmentVariable wide_setting{
            "LUISA_SIMD_DISABLE_WIDE_PREDICATED_IF_REFINEMENT",
            disable_wide ? "1" : "0"};
        return compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_deep_select_ladder", false, true);
    };
    auto execute = [&](const SIMDCompiledKernel &compiled,
                       uint32_t width,
                       std::array<luisa::float4, count> &output) {
        output.fill(luisa::make_float4(-999.0f));
        alignas(16) SIMDHostBufferView argument{
            output.data(), sizeof(output)};
        using Entry = void(
            const void *, void *, const SIMDPacketLaunchConfig *,
            uint32_t);
        auto *entry = reinterpret_cast<Entry *>(compiled.entry);
        CHECK(entry != nullptr);
        auto config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += width) {
            config.thread_index = first;
            entry(&argument, nullptr, &config, width);
        }
        return true;
    };

    for (auto width : {2u, 4u, 8u, 16u}) {
        auto candidate = compile(width, false, true);
        auto oracle = compile(width, true, true);
        CHECK(candidate.succeeded());
        CHECK(oracle.succeeded());
        if (width == 8u) {
            CHECK(candidate.predicated_diamond_count ==
                  oracle.predicated_diamond_count + 1u);
            CHECK(candidate.predicated_refinement_round_count ==
                  oracle.predicated_refinement_round_count + 1u);
            CHECK(candidate.predicated_forwarded_phi_count ==
                  oracle.predicated_forwarded_phi_count + 1u);
            CHECK(candidate.predicated_forwarding_block_count ==
                  oracle.predicated_forwarding_block_count + 1u);
            CHECK(candidate.schedule_block_count <
                  oracle.schedule_block_count);
            CHECK(candidate.convergence_point_count <
                  oracle.convergence_point_count);
            CHECK(candidate.state_slot_count < oracle.state_slot_count);
            if (candidate.target_triple.starts_with("x86_64")) {
                CHECK(candidate.assembly.size() < oracle.assembly.size());
            }
        } else {
            CHECK(candidate.predicated_diamond_count ==
                  oracle.predicated_diamond_count);
            CHECK(candidate.predicated_refinement_round_count ==
                  oracle.predicated_refinement_round_count);
            CHECK(candidate.predicated_forwarded_phi_count ==
                  oracle.predicated_forwarded_phi_count);
            CHECK(candidate.predicated_forwarding_block_count ==
                  oracle.predicated_forwarding_block_count);
            CHECK(candidate.schedule_block_count ==
                  oracle.schedule_block_count);
            CHECK(candidate.convergence_point_count ==
                  oracle.convergence_point_count);
            CHECK(candidate.state_slot_count == oracle.state_slot_count);
            CHECK(candidate.assembly == oracle.assembly);
        }

        std::array<luisa::float4, count> output{};
        std::array<luisa::float4, count> oracle_output{};
        CHECK(execute(candidate, width, output));
        CHECK(execute(oracle, width, oracle_output));
        CHECK(std::memcmp(
                  output.data(), oracle_output.data(),
                  sizeof(output)) == 0);
        auto wide_candidate = compile(width, false, false);
        CHECK(wide_candidate.succeeded());
        if (width == 8u) {
            CHECK(wide_candidate.predicated_wide_select_ladder_diamond_count ==
                  1u);
            CHECK(wide_candidate.predicated_diamond_count ==
                  candidate.predicated_diamond_count + 1u);
            CHECK(wide_candidate.predicated_instruction_count ==
                  candidate.predicated_instruction_count + 6u);
            CHECK(wide_candidate.predicated_phi_count ==
                  candidate.predicated_phi_count + 1u);
            CHECK(wide_candidate.predicated_refinement_round_count ==
                  candidate.predicated_refinement_round_count + 1u);
            CHECK(wide_candidate.predicated_forwarded_phi_count ==
                  candidate.predicated_forwarded_phi_count + 1u);
            CHECK(wide_candidate.predicated_forwarding_block_count ==
                  candidate.predicated_forwarding_block_count + 1u);
            CHECK(wide_candidate.schedule_block_count <
                  candidate.schedule_block_count);
            CHECK(wide_candidate.convergence_point_count <
                  candidate.convergence_point_count);
            CHECK(wide_candidate.state_slot_count <
                  candidate.state_slot_count);
            if (wide_candidate.target_triple.starts_with("x86_64")) {
                CHECK(wide_candidate.assembly.size() <
                      candidate.assembly.size());
            }
        } else {
            CHECK(wide_candidate.predicated_wide_select_ladder_diamond_count ==
                  0u);
            CHECK(wide_candidate.predicated_diamond_count ==
                  candidate.predicated_diamond_count);
            CHECK(wide_candidate.predicated_instruction_count ==
                  candidate.predicated_instruction_count);
            CHECK(wide_candidate.predicated_phi_count ==
                  candidate.predicated_phi_count);
            CHECK(wide_candidate.predicated_refinement_round_count ==
                  candidate.predicated_refinement_round_count);
            CHECK(wide_candidate.predicated_forwarded_phi_count ==
                  candidate.predicated_forwarded_phi_count);
            CHECK(wide_candidate.predicated_forwarding_block_count ==
                  candidate.predicated_forwarding_block_count);
            CHECK(wide_candidate.schedule_block_count ==
                  candidate.schedule_block_count);
            CHECK(wide_candidate.convergence_point_count ==
                  candidate.convergence_point_count);
            CHECK(wide_candidate.state_slot_count ==
                  candidate.state_slot_count);
            CHECK(wide_candidate.assembly == candidate.assembly);
        }
        std::array<luisa::float4, count> wide_output{};
        CHECK(execute(wide_candidate, width, wide_output));
        CHECK(std::memcmp(
                  output.data(), wide_output.data(),
                  sizeof(output)) == 0);
        for (auto index = uint32_t{0u}; index < count; index++) {
            auto base = 71.0f;
            switch (index & 7u) {
                case 1u: base = 11.0f; break;
                case 2u: base = 21.0f; break;
                case 3u: base = 31.0f; break;
                case 4u: base = 41.0f; break;
                case 5u: base = 51.0f; break;
                default: break;
            }
            CHECK(output[index].x == base);
            CHECK(output[index].y == base + 1.0f);
            CHECK(output[index].z == base + 2.0f);
            CHECK(output[index].w == 1.0f);
        }
    }
    return true;
}

[[nodiscard]] bool run_ast_widened_predicated_update() {
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferFloat4 output,
                         BufferUInt material_output) noexcept {
        auto index = dispatch_id().x;
        auto discriminant = cast<float>(index) - 4.0f;
        Float hit = 1000.0f;
        Float3 normal = make_float3(0.0f);
        UInt material = 0xffffffffu;
        $if (discriminant > 0.0f) {
            auto root = sqrt(discriminant);
            $if (root < 2.0f) {
                auto scaled = make_float3(root) *
                              make_float3(2.0f, 3.0f, 4.0f);
                auto shifted = scaled -
                               make_float3(0.5f, 0.25f, 0.125f);
                auto denominator = 2.0f - root;
                normal = shifted / make_float3(denominator);
                hit = root;
                material = 7u;
            };
        };
        output.write(index, make_float4(normal, hit));
        material_output.write(index, material);
    };
    auto compile = [&](uint32_t width, bool disable) {
        ScopedEnvironmentVariable setting{
            "LUISA_SIMD_DISABLE_WIDENED_PREDICATED_UPDATE",
            disable ? "1" : "0"};
        return compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_widened_update",
            false, true);
    };
    auto execute = [&](const SIMDCompiledKernel &compiled,
                       uint32_t width,
                       std::array<luisa::float4, count> &output,
                       std::array<uint32_t, count> &materials) {
        output.fill(luisa::make_float4(-999.0f));
        materials.fill(0xdeadbeefu);
        std::array<SIMDHostBufferView, 2u> arguments{
            SIMDHostBufferView{output.data(), sizeof(output)},
            SIMDHostBufferView{materials.data(), sizeof(materials)}};
        using Entry = void(
            const void *, void *, const SIMDPacketLaunchConfig *,
            uint32_t);
        auto *entry = reinterpret_cast<Entry *>(compiled.entry);
        CHECK(entry != nullptr);
        auto config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += width) {
            config.thread_index = first;
            entry(arguments.data(), nullptr, &config, width);
        }
        return true;
    };

    for (auto width : {1u, 2u, 4u, 8u, 16u}) {
        auto candidate = compile(width, false);
        auto oracle = compile(width, true);
        CHECK(candidate.succeeded());
        CHECK(oracle.succeeded());
        if (width == 1u) {
            CHECK(candidate.predicated_widened_update_diamond_count ==
                  0u);
            CHECK(oracle.predicated_widened_update_diamond_count == 0u);
            CHECK(candidate.predicated_diamond_count ==
                  oracle.predicated_diamond_count);
            CHECK(candidate.schedule_block_count ==
                  oracle.schedule_block_count);
            CHECK(candidate.convergence_point_count ==
                  oracle.convergence_point_count);
            CHECK(candidate.state_slot_count == oracle.state_slot_count);
            CHECK(candidate.assembly == oracle.assembly);
        } else {
            CHECK(candidate.predicated_widened_update_diamond_count ==
                  1u);
            CHECK(oracle.predicated_widened_update_diamond_count == 0u);
            CHECK(candidate.predicated_diamond_count ==
                  oracle.predicated_diamond_count + 1u);
            CHECK(candidate.schedule_block_count + 2u ==
                  oracle.schedule_block_count);
            CHECK(candidate.convergence_point_count + 1u ==
                  oracle.convergence_point_count);
            CHECK(candidate.state_slot_count < oracle.state_slot_count);
            if (candidate.target_triple.starts_with("x86_64")) {
                CHECK(candidate.assembly.size() < oracle.assembly.size());
            }
        }

        std::array<luisa::float4, count> output{};
        std::array<luisa::float4, count> oracle_output{};
        std::array<uint32_t, count> materials{};
        std::array<uint32_t, count> oracle_materials{};
        CHECK(execute(candidate, width, output, materials));
        CHECK(execute(
            oracle, width, oracle_output, oracle_materials));
        CHECK(std::memcmp(
                  output.data(), oracle_output.data(),
                  sizeof(output)) == 0);
        CHECK(materials == oracle_materials);
        for (auto index = uint32_t{0u}; index < count; index++) {
            auto updated = index > 4u && index < 8u;
            CHECK(materials[index] ==
                  (updated ? 7u : 0xffffffffu));
            CHECK(std::isfinite(output[index].x));
            CHECK(std::isfinite(output[index].y));
            CHECK(std::isfinite(output[index].z));
            if (updated) {
                CHECK(std::abs(
                          output[index].w -
                          std::sqrt(static_cast<float>(index) - 4.0f)) <
                      1.0e-6f);
            } else {
                CHECK(luisa::all(
                    output[index] == luisa::make_float4(
                                         0.0f, 0.0f, 0.0f,
                                         1000.0f)));
            }
        }
    }
    return true;
}

[[nodiscard]] bool run_predicated_select_forwarding_metadata() {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *outer_true = kernel->create_basic_block();
    auto *outer_false = kernel->create_basic_block();
    auto *inner_true = kernel->create_basic_block();
    auto *inner_false = kernel->create_basic_block();
    auto *inner_merge = kernel->create_basic_block();
    auto *outer_merge = kernel->create_basic_block();
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto two_value = uint32_t{2u};
    auto three_value = uint32_t{3u};
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *outer_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {lane, zero});
    builder.cond_br(
        outer_condition, outer_true, outer_false);
    builder.set_insertion_point(outer_true);
    builder.br(outer_merge);
    builder.set_insertion_point(outer_false);
    auto *inner_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {lane, one});
    builder.cond_br(
        inner_condition, inner_true, inner_false);
    builder.set_insertion_point(inner_true);
    builder.br(inner_merge);
    builder.set_insertion_point(inner_false);
    builder.br(inner_merge);
    builder.set_insertion_point(inner_merge);
    auto *inner_phi = builder.phi(
        Type::of<uint32_t>(),
        {{two, inner_true}, {three, inner_false}});
    inner_phi->add_comment(
        "select forwarding must preserve this owner");
    builder.br(outer_merge);
    builder.set_insertion_point(outer_merge);
    static_cast<void>(builder.phi(
        Type::of<uint32_t>(),
        {{one, outer_true}, {inner_phi, inner_merge}}));
    builder.return_void();
    CHECK(xir::xir_verify_module(&module).succeeded());

    auto predication =
        schedule::predicate_small_varying_diamonds(kernel);
    CHECK(predication.if_conversion.converted_diamond_count == 1u);
    CHECK(predication.refinement_round_count == 0u);
    CHECK(predication.forwarded_phi_count == 0u);
    CHECK(predication.removed_forwarding_block_count == 0u);
    CHECK(entry->terminator()->isa<xir::ConditionalBranchInst>());
    CHECK(inner_phi->is_linked());
    CHECK(!inner_phi->metadata_list().empty());
    CHECK(xir::xir_verify_module(&module).succeeded());
    return true;
}

[[nodiscard]] bool run_predicated_select_forwarding_provenance() {
    xir::Module module;
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *outer_true = kernel->create_basic_block();
    auto *outer_false = kernel->create_basic_block();
    auto *inner_true = kernel->create_basic_block();
    auto *inner_false = kernel->create_basic_block();
    auto *inner_merge = kernel->create_basic_block();
    auto *outer_merge = kernel->create_basic_block();
    auto *preexisting_select_block = kernel->create_basic_block();
    auto *preexisting_forwarder = kernel->create_basic_block();
    auto *exit = kernel->create_basic_block();
    auto *lane = module.create_warp_lane_id();
    auto *zero = module.create_constant_zero(Type::of<uint32_t>());
    auto *one = module.create_constant_one(Type::of<uint32_t>());
    auto two_value = uint32_t{2u};
    auto three_value = uint32_t{3u};
    auto *two = module.create_constant(
        Type::of<uint32_t>(), &two_value);
    auto *three = module.create_constant(
        Type::of<uint32_t>(), &three_value);
    xir::XIRBuilder builder;
    builder.set_insertion_point(entry);
    auto *outer_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {lane, zero});
    builder.cond_br(
        outer_condition, outer_true, outer_false);
    builder.set_insertion_point(outer_true);
    builder.br(outer_merge);
    builder.set_insertion_point(outer_false);
    auto *inner_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {lane, one});
    builder.cond_br(
        inner_condition, inner_true, inner_false);
    builder.set_insertion_point(inner_true);
    builder.br(inner_merge);
    builder.set_insertion_point(inner_false);
    builder.br(inner_merge);
    builder.set_insertion_point(inner_merge);
    auto *inner_phi = builder.phi(
        Type::of<uint32_t>(),
        {{two, inner_true}, {three, inner_false}});
    inner_phi->set_name("generated_inner_value");
    builder.br(outer_merge);
    builder.set_insertion_point(outer_merge);
    auto *outer_phi = builder.phi(
        Type::of<uint32_t>(),
        {{one, outer_true}, {inner_phi, inner_merge}});
    outer_phi->set_name("generated_outer_value");
    builder.br(preexisting_select_block);
    builder.set_insertion_point(preexisting_select_block);
    auto *preexisting_condition = builder.call(
        Type::of<bool>(), xir::ArithmeticOp::BINARY_EQUAL,
        {lane, three});
    auto *preexisting_select = builder.call(
        Type::of<uint32_t>(), xir::ArithmeticOp::SELECT,
        {two, three, preexisting_condition});
    builder.br(preexisting_forwarder);
    builder.set_insertion_point(preexisting_forwarder);
    auto *preexisting_phi = builder.phi(
        Type::of<uint32_t>(),
        {{preexisting_select, preexisting_select_block}});
    preexisting_phi->set_name("preexisting_value");
    builder.br(exit);
    builder.set_insertion_point(exit);
    builder.return_void();
    CHECK(xir::xir_verify_module(&module).succeeded());

    auto predication =
        schedule::predicate_small_varying_diamonds(kernel);
    CHECK(predication.if_conversion.converted_diamond_count == 2u);
    CHECK(predication.refinement_round_count == 1u);
    CHECK(predication.removed_forwarding_block_count == 1u);
    CHECK(outer_merge->is_linked());
    CHECK(preexisting_forwarder->is_linked());
    CHECK(preexisting_phi->is_linked());
    CHECK(preexisting_phi->incoming(0u).value == preexisting_select);
    CHECK(xir::xir_verify_module(&module).succeeded());
    return true;
}

[[nodiscard]] bool run_ast_predicated_memory_diamond() {
    static constexpr auto width = 8u;
    static constexpr auto count = 13u;
    Kernel1D kernel = [](BufferUInt input, BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0x12345678u;
        $if (index != 0u) {
            // Lane zero forms UINT_MAX here. The address is legal to form but
            // must be masked before the gather touches memory.
            value = input.read(index - 1u) + 7u;
        };
        output.write(index, value);
    };
    auto compiled = compile_simd_kernel(
        kernel.function()->function(), width,
        "simd_ast_predicated_memory_diamond");
    if (!compiled.succeeded()) {
        for (auto &&diagnostic : compiled.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(compiled.predicated_memory_diamond_count == 1u);
    CHECK(compiled.predicated_memory_instruction_count == 3u);
    CHECK(compiled.direct_control_flow);
    // One value crosses the ordinary direct-CFG block boundary. LLVM's
    // mem2reg removes this logical spill; the stackless shape below audits
    // the optimized machine code separately.
    CHECK(compiled.spilled_instruction_count == 1u);
    CHECK(compiled.contiguous_buffer_read_count == 1u);

    Kernel1D stackless_kernel = [](
                                    BufferUInt input, BufferUInt mask,
                                    BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0u;
        $if (mask.read(index) != 0u) {
            value = input.read(index) + 7u;
        };
        output.write(index, value);
    };
    auto stackless = compile_simd_kernel(
        stackless_kernel.function()->function(), width,
        "simd_ast_stackless_predicated_memory_diamond", false, true);
    CHECK(stackless.succeeded());
    CHECK(stackless.predicated_memory_diamond_count == 1u);
    CHECK(stackless.spilled_instruction_count == 1u);
    CHECK(stackless.contiguous_buffer_read_count == 2u);
    CHECK(!stackless.assembly.empty());
    if (stackless.target_triple.starts_with("x86_64")) {
        CHECK(stackless.assembly.find("%rsp") == std::string::npos);
        CHECK(stackless.assembly.find("%rbp") == std::string::npos);
    }

    // Exercise a completely empty read arm with an invalid input pointer.
    // Masked lowering must sanitize the dynamic seed before extraction and
    // must not touch the input address in any lane.
    std::array<uint32_t, count> zero_mask{};
    std::array<uint32_t, count> zero_output{};
    zero_output.fill(0xdeadbeefu);
    alignas(16) std::array<SIMDHostBufferView, 3u> stackless_arguments{
        SIMDHostBufferView{nullptr, 0u},
        SIMDHostBufferView{zero_mask.data(), sizeof(zero_mask)},
        SIMDHostBufferView{zero_output.data(), sizeof(zero_output)},
    };
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto stackless_entry = reinterpret_cast<Entry *>(stackless.entry);
    CHECK(stackless_entry != nullptr);
    auto stackless_config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        stackless_config.thread_index = first;
        stackless_entry(
            stackless_arguments.data(), nullptr,
            &stackless_config, width);
    }
    CHECK(std::all_of(
        zero_output.begin(), zero_output.end(),
        [](uint32_t value) noexcept { return value == 0u; }));

    auto scalar = compile_simd_kernel(
        kernel.function()->function(), 1u,
        "simd_ast_scalar_memory_diamond");
    CHECK(scalar.succeeded());
    CHECK(scalar.predicated_memory_diamond_count == 0u);
    CHECK(scalar.direct_control_flow);

    {
        ScopedEnvironmentVariable disable_predication{
            "LUISA_SIMD_DISABLE_PREDICATED_IF", "1"};
        auto scheduled = compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_scheduled_memory_diamond");
        CHECK(scheduled.succeeded());
        CHECK(scheduled.predicated_memory_diamond_count == 0u);
        CHECK(scheduled.predicated_memory_instruction_count == 0u);
        CHECK(!scheduled.direct_control_flow);
    }

    Kernel1D volatile_kernel = [](BufferUInt input,
                                  BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0u;
        $if (index != 0u) {
            value = input.volatile_read(index - 1u);
        };
        output.write(index, value);
    };
    auto volatile_read = compile_simd_kernel(
        volatile_kernel.function()->function(), width,
        "simd_ast_volatile_memory_diamond");
    CHECK(volatile_read.succeeded());
    CHECK(volatile_read.predicated_memory_diamond_count == 0u);
    CHECK(!volatile_read.direct_control_flow);

    Kernel1D unsafe_arithmetic_kernel = [](
                                            BufferUInt input,
                                            BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0u;
        $if (index != 0u) {
            value = input.read(index - 1u) / index;
        };
        output.write(index, value);
    };
    auto unsafe_arithmetic = compile_simd_kernel(
        unsafe_arithmetic_kernel.function()->function(), width,
        "simd_ast_unsafe_arithmetic_memory_diamond");
    CHECK(unsafe_arithmetic.succeeded());
    CHECK(unsafe_arithmetic.predicated_memory_diamond_count == 0u);
    CHECK(!unsafe_arithmetic.direct_control_flow);

    std::array<uint32_t, count> input{};
    std::array<uint32_t, count> output{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        input[i] = i * 11u + 5u;
        output[i] = 0xdeadbeefu;
    }
    alignas(16) std::array<SIMDHostBufferView, 2u> arguments{
        SIMDHostBufferView{input.data(), sizeof(input)},
        SIMDHostBufferView{output.data(), sizeof(output)},
    };
    auto entry = reinterpret_cast<Entry *>(compiled.entry);
    CHECK(entry != nullptr);
    auto config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        config.thread_index = first;
        entry(arguments.data(), nullptr, &config, width);
    }
    CHECK(output[0u] == 0x12345678u);
    for (auto i = uint32_t{1u}; i < count; i++) {
        CHECK(output[i] == input[i - 1u] + 7u);
    }

    for (auto test_width : {2u, 4u, 16u}) {
        auto candidate = compile_simd_kernel(
            kernel.function()->function(), test_width,
            "simd_ast_predicated_memory_diamond_w" +
                std::to_string(test_width));
        CHECK(candidate.succeeded());
        CHECK(candidate.predicated_memory_diamond_count == 1u);
        CHECK(candidate.direct_control_flow);
        CHECK(candidate.contiguous_buffer_read_count ==
              (test_width >= 4u ? 1u : 0u));
        output.fill(0xdeadbeefu);
        auto candidate_entry = reinterpret_cast<Entry *>(candidate.entry);
        CHECK(candidate_entry != nullptr);
        auto candidate_config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += test_width) {
            candidate_config.thread_index = first;
            candidate_entry(
                arguments.data(), nullptr,
                &candidate_config, test_width);
        }
        CHECK(output[0u] == 0x12345678u);
        for (auto i = uint32_t{1u}; i < count; i++) {
            CHECK(output[i] == input[i - 1u] + 7u);
        }
    }

    Kernel1D nested_kernel = [](BufferUInt input,
                                BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = 0x11111111u;
        $if ((index & 1u) == 0u) {
            $if (index != 0u) {
                value = input.read(index - 1u) + 9u;
            };
        }
        $else {
            value = 0x22222222u;
        };
        output.write(index, value);
    };
    auto nested = compile_simd_kernel(
        nested_kernel.function()->function(), width,
        "simd_ast_nested_predicated_memory_diamond");
    CHECK(nested.succeeded());
    CHECK(nested.predicated_memory_diamond_count == 1u);
    CHECK(!nested.direct_control_flow);
    output.fill(0xdeadbeefu);
    auto nested_entry = reinterpret_cast<Entry *>(nested.entry);
    CHECK(nested_entry != nullptr);
    auto nested_config = launch_1d(count, 16u);
    for (auto first = uint32_t{0u}; first < 16u; first += width) {
        nested_config.thread_index = first;
        nested_entry(
            arguments.data(), nullptr,
            &nested_config, width);
    }
    CHECK(output[0u] == 0x11111111u);
    for (auto i = uint32_t{1u}; i < count; i++) {
        auto expected = (i & 1u) == 0u ? input[i - 1u] + 9u :
                                         0x22222222u;
        CHECK(output[i] == expected);
    }
    return true;
}

[[nodiscard]] bool run_ast_predicated_loop_batch() {
    static constexpr auto width = 16u;
    static constexpr auto count = 13u;
    static constexpr auto max_iterations = 32u;
    Kernel1D kernel = [](BufferUInt limits, BufferUInt input,
                         BufferFloat cast_input,
                         BufferUInt output) noexcept {
        set_block_size(32u, 1u, 1u);
        auto index = dispatch_id().x;
        auto limit = limits.read(index);
        auto cast_source = cast_input.read(index);
        auto cast_lane = (index & 1u) == 0u;
        $uint value = index * 0x9e3779b9u + 17u;
        $for (iteration, 32u) {
            $if (iteration >= limit) { $break; };
            auto address = index * 32u + iteration;
            auto sample = input.read(address);
            $if ((sample & 0xffu) == 0x7fu) {
                value = value + sample + iteration;
                $break;
            };
            $if (cast_lane) {
                value = value + cast<uint>(cast_source);
            }
            $else {
                value = value ^ 0x51ed270bu;
            };
            $if ((sample & 1u) == 0u) {
                value = value * 3u + sample + iteration;
            }
            $else {
                value = (value ^ sample) + iteration + 5u;
            };
        };
        auto sum = warp_active_sum(value);
        output.write(index, value ^ sum);
    };

    SIMDCompiledKernel candidate;
    {
        ScopedEnvironmentVariable force{
            "LUISA_SIMD_FORCE_PREDICATED_LOOP", "1"};
        candidate = compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_predicated_loop_batch");
    }
    if (!candidate.succeeded()) {
        for (auto &&diagnostic : candidate.diagnostics) {
            std::cerr << diagnostic << '\n';
        }
        return false;
    }
    CHECK(candidate.predicated_loop_count == 1u);
    CHECK(candidate.predicated_loop_block_count >= 6u);
    CHECK(candidate.predicated_loop_instruction_count != 0u);
    CHECK(candidate.predicated_loop_batch_iteration_count ==
          max_iterations + 1u);

    SIMDCompiledKernel oracle;
    {
        ScopedEnvironmentVariable disable{
            "LUISA_SIMD_DISABLE_PREDICATED_LOOP", "1"};
        oracle = compile_simd_kernel(
            kernel.function()->function(), width,
            "simd_ast_predicated_loop_batch_oracle");
    }
    CHECK(oracle.succeeded());
    CHECK(oracle.predicated_loop_count == 0u);
    CHECK(oracle.predicated_loop_block_count == 0u);
    CHECK(oracle.predicated_loop_instruction_count == 0u);
    CHECK(oracle.predicated_loop_batch_iteration_count == 0u);

    auto narrow = compile_simd_kernel(
        kernel.function()->function(), 8u,
        "simd_ast_predicated_loop_batch_w8");
    CHECK(narrow.succeeded());
    CHECK(narrow.predicated_loop_count == 0u);
    auto w8_parallel = compile_simd_kernel(
        kernel.function()->function(), 8u,
        "simd_ast_predicated_loop_batch_w8_parallel",
        false, false, 24u);
    CHECK(w8_parallel.succeeded());
    CHECK(w8_parallel.predicated_loop_count ==
          static_cast<size_t>(w8_parallel.native_predicated_loop));
    auto w8_below_crossover = compile_simd_kernel(
        kernel.function()->function(), 8u,
        "simd_ast_predicated_loop_batch_w8_below_crossover",
        false, false, 23u);
    CHECK(w8_below_crossover.succeeded());
    CHECK(w8_below_crossover.predicated_loop_count == 0u);

    // Side effects and volatile reads must keep the ordinary scheduler. These
    // kernels retain the same finite, multi-block loop shape as the accepted
    // case so a zero counter exercises the instruction whitelist rather than
    // merely missing the structural profitability threshold.
    Kernel1D writing_kernel = [](BufferUInt input,
                                 BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = index + 1u;
        $for (iteration, 32u) {
            auto sample = input.read(index * 32u + iteration);
            $if ((sample & 0xffu) == 0x7fu) { $break; };
            $if ((sample & 1u) == 0u) {
                value = value * 3u + sample;
            }
            $else {
                value = value ^ sample;
            };
            output.write(index, value);
        };
        output.write(index, value);
    };
    auto writing = compile_simd_kernel(
        writing_kernel.function()->function(), width,
        "simd_ast_predicated_loop_writing_rejection");
    CHECK(writing.succeeded());
    CHECK(writing.predicated_loop_count == 0u);

    Kernel1D volatile_kernel = [](BufferUInt input,
                                  BufferUInt output) noexcept {
        auto index = dispatch_id().x;
        $uint value = index + 1u;
        $for (iteration, 32u) {
            auto sample = input.volatile_read(
                index * 32u + iteration);
            $if ((sample & 0xffu) == 0x7fu) { $break; };
            $if ((sample & 1u) == 0u) {
                value = value * 3u + sample;
            }
            $else {
                value = value ^ sample;
            };
        };
        output.write(index, value);
    };
    auto volatile_read = compile_simd_kernel(
        volatile_kernel.function()->function(), width,
        "simd_ast_predicated_loop_volatile_rejection");
    CHECK(volatile_read.succeeded());
    CHECK(volatile_read.predicated_loop_count == 0u);

    std::array<uint32_t, count> limits{};
    std::array<uint32_t, count * max_iterations> input{};
    std::array<float, count> cast_input{};
    std::array<uint32_t, count> values{};
    std::array<uint32_t, count> expected{};
    std::array<uint32_t, count> candidate_output{};
    std::array<uint32_t, count> oracle_output{};
    for (auto index = uint32_t{0u}; index < count; index++) {
        limits[index] = (index * 11u) % (max_iterations + 1u);
        cast_input[index] = (index & 1u) == 0u ?
                                3.75f :
                                std::bit_cast<float>(0x7fc00000u);
        for (auto iteration = uint32_t{0u};
             iteration < max_iterations; iteration++) {
            input[index * max_iterations + iteration] =
                (index + 3u) * 131u + iteration * 17u + 2u;
        }
        if (index % 3u == 1u && limits[index] > 2u) {
            auto sentinel_iteration = limits[index] / 2u;
            input[index * max_iterations + sentinel_iteration] =
                0x127fu;
        }
        auto value = index * 0x9e3779b9u + 17u;
        for (auto iteration = uint32_t{0u};
             iteration < max_iterations &&
             iteration < limits[index];
             iteration++) {
            auto sample =
                input[index * max_iterations + iteration];
            if ((sample & 0xffu) == 0x7fu) {
                value = value + sample + iteration;
                break;
            }
            value = (index & 1u) == 0u ?
                        value + static_cast<uint32_t>(cast_input[index]) :
                        value ^ 0x51ed270bu;
            value = (sample & 1u) == 0u ?
                        value * 3u + sample + iteration :
                        (value ^ sample) + iteration + 5u;
        }
        values[index] = value;
    }
    auto sum = uint32_t{0u};
    for (auto value : values) { sum += value; }
    for (auto index = uint32_t{0u}; index < count; index++) {
        expected[index] = values[index] ^ sum;
    }
    candidate_output.fill(0xdeadbeefu);
    oracle_output.fill(0xdeadbeefu);
    using Entry = void(
        const void *, void *, const SIMDPacketLaunchConfig *, uint32_t);
    auto run = [&](SIMDCompiledKernel &compiled,
                   std::array<uint32_t, count> &output) noexcept {
        alignas(16) std::array<SIMDHostBufferView, 4u> arguments{
            SIMDHostBufferView{limits.data(), sizeof(limits)},
            SIMDHostBufferView{input.data(), sizeof(input)},
            SIMDHostBufferView{cast_input.data(), sizeof(cast_input)},
            SIMDHostBufferView{output.data(), sizeof(output)},
        };
        auto entry = reinterpret_cast<Entry *>(compiled.entry);
        CHECK(entry != nullptr);
        auto config = launch_1d(count, width);
        entry(arguments.data(), nullptr, &config, width);
        return true;
    };
    CHECK(run(candidate, candidate_output));
    CHECK(run(oracle, oracle_output));
    CHECK(candidate_output == expected);
    CHECK(oracle_output == expected);
    CHECK(candidate_output == oracle_output);
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

    // Unknown per-lane trip counts use an entry guard: zero-trip lanes leave
    // before the invariant varying selector is evaluated, while entering
    // cohorts choose one specialized innermost-loop version. Exercise all
    // packet widths and a partial W8 tail against the scheduled oracle.
    Kernel1D dynamic_kernel = [](BufferUInt trip_counts,
                                 BufferUInt dynamic_output) noexcept {
        auto index = dispatch_id().x;
        auto trip_count = trip_counts.read(index);
        auto choose_left = (index & 1u) == 0u;
        $uint value = 1u;
        $uint iteration = 0u;
        $while (iteration < trip_count) {
            $if (choose_left) {
                value = value * 3u + index;
                value = value ^ (iteration + 9u);
            }
            $else {
                value = value * 5u + index;
                value = value ^ (iteration + 17u);
            };
            iteration += 1u;
        };
        dynamic_output.write(index, value);
    };
    std::array<uint32_t, count> trip_counts{};
    for (auto i = uint32_t{0u}; i < count; i++) {
        trip_counts[i] = (i * 5u) % 9u;
    }
    auto run_dynamic = [&](uint32_t test_width,
                           bool disable_guarded,
                           std::array<uint32_t, count> &dynamic_output) noexcept {
        ScopedEnvironmentVariable disable{
            "LUISA_SIMD_DISABLE_GUARDED_LOOP_UNSWITCH",
            disable_guarded ? "1" : nullptr};
        auto compiled = compile_simd_kernel(
            dynamic_kernel.function()->function(), test_width,
            "simd_ast_dynamic_loop_unswitch_w" +
                std::to_string(test_width) +
                (disable_guarded ? "_oracle" : "_candidate"));
        CHECK(compiled.succeeded());
        CHECK(compiled.unswitched_loop_count ==
              (disable_guarded ? 0u : 1u));
        CHECK(compiled.guarded_unswitched_loop_count ==
              (disable_guarded ? 0u : 1u));
        dynamic_output.fill(0xdeadbeefu);
        alignas(16) std::array<SIMDHostBufferView, 2u> arguments{
            SIMDHostBufferView{trip_counts.data(), sizeof(trip_counts)},
            SIMDHostBufferView{dynamic_output.data(),
                               sizeof(dynamic_output)},
        };
        auto dynamic_entry = reinterpret_cast<Entry *>(compiled.entry);
        CHECK(dynamic_entry != nullptr);
        auto dynamic_config = launch_1d(count, 16u);
        for (auto first = uint32_t{0u}; first < 16u;
             first += test_width) {
            dynamic_config.thread_index = first;
            dynamic_entry(
                arguments.data(), nullptr,
                &dynamic_config, test_width);
        }
        for (auto index = uint32_t{0u}; index < count; index++) {
            auto expected = uint32_t{1u};
            auto choose_left = (index & 1u) == 0u;
            for (auto iteration = uint32_t{0u};
                 iteration < trip_counts[index]; iteration++) {
                if (choose_left) {
                    expected = expected * 3u + index;
                    expected ^= iteration + 9u;
                } else {
                    expected = expected * 5u + index;
                    expected ^= iteration + 17u;
                }
            }
            CHECK(dynamic_output[index] == expected);
        }
        return true;
    };
    for (auto test_width : {2u, 4u, 8u, 16u}) {
        std::array<uint32_t, count> candidate{};
        CHECK(run_dynamic(test_width, false, candidate));
        if (test_width == width) {
            std::array<uint32_t, count> oracle{};
            CHECK(run_dynamic(test_width, true, oracle));
            CHECK(candidate == oracle);
        }
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
        {"direct divergent child", &run_direct_divergent_child_codegen},
        {"static power-of-two block size",
         &run_static_block_size_codegen},
        {"Schedule IR loop warp1", &run_loop_codegen<1u>},
        {"Schedule IR loop warp2", &run_loop_codegen<2u>},
        {"Schedule IR loop warp4", &run_loop_codegen<4u>},
        {"Schedule IR loop warp8", &run_loop_codegen<8u>},
        {"innermost-loop local predicated regions",
         &run_local_predicated_region_codegen},
        {"innermost-loop local predicated terminal bridge",
         &run_local_predicated_terminal_bridge_codegen},
        {"innermost-loop two-sided local predication",
         &run_two_sided_local_predicated_codegen},
        {"nested innermost-loop local predicated regions",
         &run_nested_local_predicated_region_codegen},
        {"varying loop exit collective",
         &run_varying_loop_collective_codegen},
        {"multiple-exit loop collective",
         &run_multiple_exit_loop_collective_codegen},
        {"Schedule IR nested convergence", &run_nested_codegen},
        {"Schedule IR nested convergence W2", &run_nested_w2_codegen},
        {"Schedule IR 96-block CFG", &run_large_cfg_codegen},
        {"scheduler state residency", &run_state_residency_codegen},
        {"scheduler state PHI coalescing",
         &run_state_phi_coalescing_codegen},
        {"scheduler general state coloring",
         &run_general_state_coloring_codegen},
        {"scalar uniform values", &run_uniform_value_codegen},
        {"scalar uniform switch", &run_uniform_switch_codegen},
        {"varying switch convergence", &run_varying_switch_codegen},
        {"runtime coherent varying control",
         &run_runtime_coherent_control_codegen},
        {"coherent all-on region versioning",
         &run_coherent_all_on_region_codegen},
        {"scalar switch loop exits",
         &run_switch_loop_exits_codegen<1u>},
        {"switch loop exits", &run_switch_loop_exits_codegen<8u>},
        {"multiple loop backedges", &run_multiple_backedge_loop_codegen},
        {"dynamic non-dominating convergence",
         &run_non_dominating_convergence_codegen},
        {"return convergence cascade",
         &run_return_convergence_cascade_codegen},
        {"W16 scalar frame metadata",
         &run_scalar_frame_metadata_codegen},
        {"XIR compiler facade", &run_compiler_facade},
        {"lane/value transposed direct buffer",
         &run_buffer_vector_codegen},
        {"sparse lane/value transposed direct buffer",
         &run_sparse_lane_value_transpose_codegen},
        {"paired direct-buffer leaf gather",
         &run_paired_leaf_gather_ir},
        {"XIR faceforward fixed-vector arithmetic",
         &run_faceforward_codegen},
        {"lane-affine scalar buffer load/store",
         &run_lane_affine_buffer_codegen},
        {"uniform buffer read broadcast",
         &run_uniform_buffer_broadcast_codegen},
        {"XIR texture packet callback", &run_texture_packet_codegen},
        {"XIR accel instance metadata",
         &run_accel_instance_metadata_codegen},
        {"XIR accel direct packet ABI",
         &run_accel_direct_packet_codegen},
        {"XIR ray-query packet callback",
         &run_ray_query_packet_codegen},
        {"ray-query status packing", &run_ray_query_status_pack},
        {"XIR ray-query status cache",
         &run_ray_query_status_cache_codegen},
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
        {"XIR bindless uniform gradient LOD",
         &run_bindless_uniform_gradient_lod_codegen},
        {"AST buffer dispatch", &run_ast_buffer_codegen},
        {"AST packet-batch runtime entry", &run_ast_packet_batch_entry},
        {"AST aggregate local promotion",
         &run_ast_aggregate_promotion},
        {"AST uniform-loop buffer broadcast",
         &run_ast_uniform_loop_buffer_broadcast},
        {"AST coherent-loop direct control",
         &run_ast_coherent_loop_direct_control},
        {"AST select operand order", &run_ast_select_codegen},
        {"AST fast radix pow canonicalization",
         &run_ast_fast_math_canonicalization},
        {"AST predicated varying diamond",
         &run_ast_predicated_diamond},
        {"AST predicated select ladder",
         &run_ast_predicated_select_ladder},
        {"AST deep predicated select ladder",
         &run_ast_deep_predicated_select_ladder},
        {"AST widened predicated update",
         &run_ast_widened_predicated_update},
        {"predicated select forwarding metadata",
         &run_predicated_select_forwarding_metadata},
        {"predicated select forwarding provenance",
         &run_predicated_select_forwarding_provenance},
        {"AST predicated memory diamond",
         &run_ast_predicated_memory_diamond},
        {"AST bounded predicated loop batch",
         &run_ast_predicated_loop_batch},
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
