#include <luisa/xir/passes/fast_math_simplify.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/unordered_map.h>

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

struct DifferenceKey {
    const Value *lhs;
    const Value *rhs;

    bool operator==(const DifferenceKey &) const noexcept = default;
};

struct DifferenceKeyHash {
    [[nodiscard]] size_t operator()(const DifferenceKey &key) const noexcept {
        return luisa::hash_combine({reinterpret_cast<uint64_t>(key.lhs),
                                    reinterpret_cast<uint64_t>(key.rhs)});
    }
};

// Only these direct uses hide the sign of a difference. In particular, a
// normalize, extract, store, PHI, or callable argument observes its direction.
[[nodiscard]] static bool has_only_even_uses(const ArithmeticInst *difference) noexcept {
    if (difference->use_list().empty()) { return false; }
    for (auto *use : difference->use_list()) {
        auto *user = use->user();
        if (!user->isa<ArithmeticInst>()) { return false; }
        auto *arithmetic = static_cast<const ArithmeticInst *>(user);
        switch (arithmetic->op()) {
            case ArithmeticOp::LENGTH:
            case ArithmeticOp::LENGTH_SQUARED:
                if (arithmetic->operand_count() != 1u ||
                    arithmetic->operand(0u) != difference) { return false; }
                break;
            case ArithmeticOp::DOT:
            case ArithmeticOp::BINARY_MUL:
                if (arithmetic->operand_count() != 2u ||
                    arithmetic->operand(0u) != difference ||
                    arithmetic->operand(1u) != difference) { return false; }
                break;
            default: return false;
        }
    }
    return true;
}

static void share_opposite_differences(
    FunctionDefinition *definition, FastMathSimplifyInfo &info) noexcept {
    luisa::unordered_map<DifferenceKey, ArithmeticInst *, DifferenceKeyHash> seen;
    luisa::unordered_set<const ArithmeticInst *> known_non_even;
    luisa::vector<ArithmeticInst *> to_remove;
    auto uses_are_even = [&](const ArithmeticInst *difference) noexcept {
        if (known_non_even.contains(difference)) { return false; }
        if (has_only_even_uses(difference)) { return true; }
        known_non_even.emplace(difference);
        return false;
    };
    definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
        // An earlier instruction in the same block dominates every replacement.
        // Never extend this proof across PHIs, branches, or loop boundaries.
        seen.clear();
        // Cache only rejections. RAUW may change use lists, so a cached success
        // could become unsafe. A stale rejection only misses a later opportunity;
        // it cannot authorize a rewrite. Reset at each block and pass invocation.
        known_non_even.clear();
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<ArithmeticInst>()) { continue; }
            auto *difference = static_cast<ArithmeticInst *>(instruction);
            if (difference->op() != ArithmeticOp::BINARY_SUB ||
                difference->operand_count() != 2u ||
                !difference->metadata_list().empty() ||
                !is_f32_or_f32_vector(difference->type())) { continue; }
            auto *lhs = difference->operand(0u);
            auto *rhs = difference->operand(1u);
            if (lhs == nullptr || rhs == nullptr || lhs == rhs ||
                lhs->type() != difference->type() ||
                rhs->type() != difference->type()) { continue; }
            DifferenceKey key{lhs, rhs};
            auto opposite = seen.find(DifferenceKey{rhs, lhs});
            if (opposite != seen.end()) {
                auto *leader = opposite->second;
                // RAUW of another difference may have changed a saved key's
                // operands. Recheck the exact SSA identities before reusing it.
                if (leader->operand(0u) == rhs && leader->operand(1u) == lhs) {
                    auto merge = uses_are_even(difference);
                    if (!merge && uses_are_even(leader)) {
                        // Keep the earlier definition, but use the direction
                        // needed by the later sign-sensitive users. The leader's
                        // existing uses only observe its square or norm.
                        leader->set_operand(0u, lhs);
                        leader->set_operand(1u, rhs);
                        seen.erase(opposite);
                        seen.insert_or_assign(key, leader);
                        merge = true;
                    }
                    if (merge) {
                        difference->replace_all_uses_with(leader);
                        to_remove.emplace_back(difference);
                        ++info.opposite_sub_count;
                        continue;
                    }
                }
            }
            seen.insert_or_assign(key, difference);
        }
    });
    for (auto *difference : to_remove) { difference->remove_self(); }
}

static void simplify_function(
    Function *function, FastMathSimplifyInfo &info,
    FastMathSimplifyOptions options) noexcept {
    if (!options.enable_fast_math || function == nullptr) { return; }
    auto *definition = function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return;
    }
    share_opposite_differences(definition, info);
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
                base, luisa::bit_cast<uint32_t>(1.0f))) {
            replacement = module->create_constant_one(power->type());
            identity = true;
        } else if (is_uniform_f32_bits(
                       base,
                       luisa::bit_cast<uint32_t>(2.0f))) {
            builder.set_insertion_point(power);
            replacement = builder.call(
                power->type(), ArithmeticOp::EXP2, {exponent});
        } else if (is_uniform_f32_bits(
                       base,
                       luisa::bit_cast<uint32_t>(10.0f))) {
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
        report->set("opposite-sub", info.opposite_sub_count);
    }
    return info;
}

}// namespace luisa::compute::xir
