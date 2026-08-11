#include "coro_predicate_analysis.h"

#include <algorithm>

#include <luisa/ast/type.h>
#include <luisa/core/stl/hash.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>

#include "coro_semantic_graph.h"
#include "helpers.h"

namespace luisa::compute::xir::detail {

namespace {

[[nodiscard]] bool is_exactly_commutative(
    ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
            return true;
        default: return false;
    }
}

[[nodiscard]] luisa::optional<bool>
decode_bool_constant(Value *value) noexcept {
    if (value != nullptr && value->isa<Constant>() &&
        value->type() == Type::of<bool>()) {
        return static_cast<Constant *>(value)->as<bool>();
    }
    return luisa::nullopt;
}

[[nodiscard]] uint64_t hash_arithmetic_term(
    const Type *type, ArithmeticOp op,
    luisa::span<const size_t> operands) noexcept {
    constexpr auto tag = uint8_t{1u};
    auto hash = luisa::hash64(
        &tag, sizeof(tag), hash64_default_seed);
    auto type_hash = type == nullptr ? uint64_t{0u} : type->hash();
    hash = luisa::hash64(&type_hash, sizeof(type_hash), hash);
    hash = luisa::hash64(&op, sizeof(op), hash);
    return luisa::hash64(
        operands.data(), operands.size_bytes(), hash);
}

}// namespace

luisa::optional<size_t>
CoroPredicateAnalysis::_term_for_value(Value *value) noexcept {
    if (value == nullptr || value->type() == nullptr) {
        return luisa::nullopt;
    }
    if (auto iter = _leaf_terms.find(value);
        iter != _leaf_terms.end()) {
        return iter->second;
    }

    if (value->isa<ArithmeticInst>()) {
        auto *arithmetic = static_cast<ArithmeticInst *>(value);
        if (is_arithmetic_op_safe_to_speculate(arithmetic->op())) {
            Term term{
                .kind = TermKind::arithmetic,
                .type = arithmetic->type(),
                .op = arithmetic->op()};
            term.operands.reserve(arithmetic->operand_count());
            for (size_t i = 0u;
                 i < arithmetic->operand_count(); ++i) {
                auto operand = _term_for_value(
                    arithmetic->operand(i));
                if (!operand) { return luisa::nullopt; }
                term.operands.emplace_back(*operand);
            }
            if (term.operands.size() == 2u &&
                is_exactly_commutative(term.op) &&
                term.operands[1] < term.operands[0]) {
                std::swap(term.operands[0], term.operands[1]);
            }
            for (auto operand : term.operands) {
                for (auto *dependency :
                     _terms[operand].dynamic_dependencies) {
                    if (std::find(
                            term.dynamic_dependencies.begin(),
                            term.dynamic_dependencies.end(),
                            dependency) ==
                        term.dynamic_dependencies.end()) {
                        term.dynamic_dependencies.emplace_back(
                            dependency);
                    }
                }
            }
            auto hash = hash_arithmetic_term(
                term.type, term.op, term.operands);
            auto &bucket = _term_buckets[hash];
            for (auto candidate : bucket) {
                auto &&existing = _terms[candidate];
                if (existing.kind == term.kind &&
                    existing.type == term.type &&
                    existing.op == term.op &&
                    existing.operands == term.operands) {
                    return candidate;
                }
            }
            auto id = _terms.size();
            _terms.emplace_back(std::move(term));
            bucket.emplace_back(id);
            return id;
        }
    }

    switch (value->derived_value_tag()) {
        case DerivedValueTag::ARGUMENT:
        case DerivedValueTag::CONSTANT:
        case DerivedValueTag::INSTRUCTION: break;
        // Special registers may denote continuation-local state (for example
        // the current coroutine token), so their value is not assumed stable
        // across a suspend edge without a dedicated semantic contract.
        case DerivedValueTag::SPECIAL_REGISTER: [[fallthrough]];
        case DerivedValueTag::UNDEFINED: [[fallthrough]];
        case DerivedValueTag::FUNCTION: [[fallthrough]];
        case DerivedValueTag::BASIC_BLOCK:
            return luisa::nullopt;
    }

    Term term{
        .kind = TermKind::leaf,
        .leaf = value,
        .type = value->type()};
    if (value->isa<Instruction>()) {
        term.dynamic_dependencies.emplace_back(
            static_cast<Instruction *>(value));
    }
    auto id = _terms.size();
    _terms.emplace_back(std::move(term));
    _leaf_terms.emplace(value, id);
    return id;
}

luisa::optional<CoroPredicateLiteral>
CoroPredicateAnalysis::_literal_for_condition(
    Value *condition) noexcept {
    if (condition == nullptr ||
        condition->type() != Type::of<bool>()) {
        return luisa::nullopt;
    }
    auto negated = false;
    for (;;) {
        if (!condition->isa<ArithmeticInst>()) { break; }
        auto *arithmetic = static_cast<ArithmeticInst *>(condition);
        if (arithmetic->op() == ArithmeticOp::UNARY_BIT_NOT &&
            arithmetic->operand_count() == 1u &&
            arithmetic->operand(0)->type() == Type::of<bool>()) {
            condition = arithmetic->operand(0);
            negated = !negated;
            continue;
        }
        if ((arithmetic->op() == ArithmeticOp::BINARY_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_NOT_EQUAL ||
             arithmetic->op() == ArithmeticOp::BINARY_BIT_XOR) &&
            arithmetic->operand_count() == 2u) {
            Value *variable = nullptr;
            luisa::optional<bool> constant;
            if (auto decoded = decode_bool_constant(
                    arithmetic->operand(1))) {
                variable = arithmetic->operand(0);
                constant = decoded;
            } else if (auto decoded = decode_bool_constant(
                           arithmetic->operand(0))) {
                variable = arithmetic->operand(1);
                constant = decoded;
            }
            if (variable != nullptr && constant &&
                variable->type() == Type::of<bool>()) {
                auto invert = false;
                switch (arithmetic->op()) {
                    case ArithmeticOp::BINARY_EQUAL:
                        invert = !*constant;
                        break;
                    case ArithmeticOp::BINARY_NOT_EQUAL:
                        invert = *constant;
                        break;
                    case ArithmeticOp::BINARY_BIT_XOR:
                        invert = *constant;
                        break;
                    default: break;
                }
                condition = variable;
                negated ^= invert;
                continue;
            }
        }
        break;
    }
    auto term = _term_for_value(condition);
    if (!term) { return luisa::nullopt; }
    return CoroPredicateLiteral{
        .predicate = *term,
        .value = !negated};
}

void CoroPredicateAnalysis::_register_predicate(
    size_t predicate) noexcept {
    if (!_registered_predicates.emplace(predicate).second) {
        return;
    }
    for (auto *dependency :
         _terms[predicate].dynamic_dependencies) {
        _predicate_kills[dependency].emplace_back(predicate);
    }
}

CoroPredicateAnalysis::CoroPredicateAnalysis(
    const CoroSemanticGraph &graph) noexcept {
    for (size_t block_id = 0u;
         block_id < graph.block_count(); ++block_id) {
        auto *block = graph.block(block_id);
        auto *terminator = block == nullptr ?
                               nullptr :
                               block->terminator();
        if (terminator == nullptr ||
            !terminator->isa<ConditionalBranchInst>()) {
            continue;
        }
        auto *branch = static_cast<
            ConditionalBranchTerminatorInstruction *>(terminator);
        if (branch->true_block() == nullptr ||
            branch->false_block() == nullptr ||
            branch->true_block() == branch->false_block()) {
            continue;
        }
        if (auto literal =
                _literal_for_condition(branch->condition())) {
            _condition_literals.emplace(terminator, *literal);
            _register_predicate(literal->predicate);
        }
    }
    for (auto &[_, predicates] : _predicate_kills) {
        std::sort(predicates.begin(), predicates.end());
        predicates.erase(
            std::unique(predicates.begin(), predicates.end()),
            predicates.end());
    }
}

luisa::optional<CoroPredicateLiteral>
CoroPredicateAnalysis::literal_on_edge(
    BasicBlock *predecessor, BasicBlock *successor) const noexcept {
    auto *terminator = predecessor == nullptr ?
                           nullptr :
                           predecessor->terminator();
    if (terminator == nullptr || successor == nullptr) {
        return luisa::nullopt;
    }
    auto iter = _condition_literals.find(terminator);
    if (iter == _condition_literals.end()) {
        return luisa::nullopt;
    }
    auto *branch = static_cast<
        ConditionalBranchTerminatorInstruction *>(terminator);
    auto literal = iter->second;
    if (branch->true_block() == successor &&
        branch->false_block() != successor) {
        return literal;
    }
    if (branch->false_block() == successor &&
        branch->true_block() != successor) {
        literal.value = !literal.value;
        return literal;
    }
    return luisa::nullopt;
}

luisa::span<const size_t>
CoroPredicateAnalysis::killed_predicates(
    Instruction *instruction) const noexcept {
    if (auto iter = _predicate_kills.find(instruction);
        iter != _predicate_kills.end()) {
        return iter->second;
    }
    return {};
}

}// namespace luisa::compute::xir::detail
