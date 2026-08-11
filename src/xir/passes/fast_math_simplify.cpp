#include <luisa/xir/passes/fast_math_simplify.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_f32_or_f32_vector(
    const Type *type) noexcept {
    return type != nullptr &&
           (type->is_float32() ||
            (type->is_vector() &&
             type->element()->is_float32()));
}

template<typename Predicate>
[[nodiscard]] static bool is_uniform_f32_constant(
    const Value *value, Predicate &&predicate) noexcept {
    if (value == nullptr || !value->isa<Constant>() ||
        !is_f32_or_f32_vector(value->type())) {
        return false;
    }
    auto *type = value->type();
    auto lane_count = type->is_vector() ? type->dimension() : 1u;
    auto *bytes = static_cast<const std::byte *>(
        static_cast<const Constant *>(value)->data());
    for (auto lane = 0u; lane < lane_count; lane++) {
        uint32_t bits = 0u;
        std::memcpy(&bits, bytes + lane * sizeof(float), sizeof(bits));
        if (!predicate(bits)) { return false; }
    }
    return true;
}

[[nodiscard]] static bool is_uniform_f32_bits(
    const Value *value, uint32_t expected) noexcept {
    return is_uniform_f32_constant(
        value, [expected](uint32_t bits) noexcept {
            return bits == expected;
        });
}

[[nodiscard]] static bool is_uniform_f32_zero(
    const Value *value) noexcept {
    return is_uniform_f32_constant(
        value, [](uint32_t bits) noexcept {
            return (bits & 0x7fffffffu) == 0u;
        });
}

static void simplify_function(
    Function *function, FastMathSimplifyInfo &info,
    FastMathSimplifyOptions options) noexcept {
    if (!options.enable_fast_math || function == nullptr) { return; }
    auto *definition = function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return;
    }
    auto *module = function->parent_module();
    luisa::vector<ArithmeticInst *> candidates;
    definition->traverse_instructions([&](Instruction *instruction) noexcept {
        if (instruction->isa<ArithmeticInst>()) {
            auto *arithmetic = static_cast<ArithmeticInst *>(instruction);
            if (arithmetic->op() == ArithmeticOp::POW) {
                candidates.emplace_back(arithmetic);
            }
        }
    });

    XIRBuilder builder;
    for (auto *power : candidates) {
        if (power->operand_count() != 2u ||
            !power->metadata_list().empty() ||
            !is_f32_or_f32_vector(power->type())) {
            continue;
        }
        auto *base = power->operand(0u);
        auto *exponent = power->operand(1u);
        if (base == nullptr || exponent == nullptr ||
            base->type() != power->type() ||
            exponent->type() != power->type()) {
            continue;
        }
        Value *replacement = nullptr;
        auto identity = false;
        if (is_uniform_f32_zero(exponent) ||
            is_uniform_f32_bits(
                base, std::bit_cast<uint32_t>(1.0f))) {
            replacement = module->create_constant_one(power->type());
            identity = true;
        } else if (is_uniform_f32_bits(
                       base,
                       std::bit_cast<uint32_t>(2.0f))) {
            builder.set_insertion_point(power);
            replacement = builder.call(
                power->type(), ArithmeticOp::EXP2, {exponent});
        } else if (is_uniform_f32_bits(
                       base,
                       std::bit_cast<uint32_t>(10.0f))) {
            builder.set_insertion_point(power);
            replacement = builder.call(
                power->type(), ArithmeticOp::EXP10, {exponent});
        }
        if (replacement == nullptr) { continue; }
        power->replace_all_uses_with(replacement);
        power->remove_self();
        if (identity) {
            info.identity_count++;
        } else {
            info.radix_pow_count++;
        }
    }
}

}// namespace detail

FastMathSimplifyInfo fast_math_simplify_pass_run_on_function(
    Function *function, FastMathSimplifyOptions options) noexcept {
    FastMathSimplifyInfo info;
    detail::simplify_function(function, info, options);
    return info;
}

FastMathSimplifyInfo fast_math_simplify_pass_run_on_module(
    Module *module, FastMathSimplifyOptions options,
    PassReport *report) noexcept {
    FastMathSimplifyInfo info;
    if (module != nullptr) {
        for (auto *function : module->function_list()) {
            detail::simplify_function(function, info, options);
        }
    }
    if (report != nullptr) {
        report->set("identity", info.identity_count);
        report->set("radix-pow", info.radix_pow_count);
    }
    return info;
}

}// namespace luisa::compute::xir
