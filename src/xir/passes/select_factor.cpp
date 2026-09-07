#include <luisa/xir/passes/select_factor.h>

#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/passes/pass_pipeline.h>

#include "helpers.h"

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_component_wise(
    ArithmeticOp op) noexcept {
    switch (op) {
        case ArithmeticOp::UNARY_MINUS:
        case ArithmeticOp::UNARY_BIT_NOT:
        case ArithmeticOp::BINARY_ADD:
        case ArithmeticOp::BINARY_SUB:
        case ArithmeticOp::BINARY_MUL:
        case ArithmeticOp::BINARY_BIT_AND:
        case ArithmeticOp::BINARY_BIT_OR:
        case ArithmeticOp::BINARY_BIT_XOR:
        case ArithmeticOp::BINARY_ROTATE_LEFT:
        case ArithmeticOp::BINARY_ROTATE_RIGHT:
        case ArithmeticOp::BINARY_LESS:
        case ArithmeticOp::BINARY_GREATER:
        case ArithmeticOp::BINARY_LESS_EQUAL:
        case ArithmeticOp::BINARY_GREATER_EQUAL:
        case ArithmeticOp::BINARY_EQUAL:
        case ArithmeticOp::BINARY_NOT_EQUAL:
        case ArithmeticOp::CLAMP:
        case ArithmeticOp::SATURATE:
        case ArithmeticOp::LERP:
        case ArithmeticOp::SMOOTHSTEP:
        case ArithmeticOp::STEP:
        case ArithmeticOp::ABS:
        case ArithmeticOp::MIN:
        case ArithmeticOp::MAX:
        case ArithmeticOp::CLZ:
        case ArithmeticOp::CTZ:
        case ArithmeticOp::POPCOUNT:
        case ArithmeticOp::REVERSE:
        case ArithmeticOp::ISINF:
        case ArithmeticOp::ISNAN:
        case ArithmeticOp::ACOS:
        case ArithmeticOp::ACOSH:
        case ArithmeticOp::ASIN:
        case ArithmeticOp::ASINH:
        case ArithmeticOp::ATAN:
        case ArithmeticOp::ATAN2:
        case ArithmeticOp::ATANH:
        case ArithmeticOp::COS:
        case ArithmeticOp::COSH:
        case ArithmeticOp::SIN:
        case ArithmeticOp::SINH:
        case ArithmeticOp::TAN:
        case ArithmeticOp::TANH:
        case ArithmeticOp::EXP:
        case ArithmeticOp::EXP2:
        case ArithmeticOp::EXP10:
        case ArithmeticOp::LOG:
        case ArithmeticOp::LOG2:
        case ArithmeticOp::LOG10:
        case ArithmeticOp::POW:
        case ArithmeticOp::POW_INT:
        case ArithmeticOp::SQRT:
        case ArithmeticOp::RSQRT:
        case ArithmeticOp::CEIL:
        case ArithmeticOp::FLOOR:
        case ArithmeticOp::FRACT:
        case ArithmeticOp::TRUNC:
        case ArithmeticOp::ROUND:
        case ArithmeticOp::RINT:
        case ArithmeticOp::FMA:
        case ArithmeticOp::COPYSIGN: return true;
        default: return false;
    }
}

[[nodiscard]] static bool select_condition_supports(
    const Type *value_type, const Type *condition_type) noexcept {
    if (value_type == nullptr || condition_type == nullptr) {
        return false;
    }
    if (condition_type->is_bool()) { return true; }
    return value_type->is_vector() &&
           condition_type->is_bool_vector() &&
           condition_type->dimension() == value_type->dimension();
}

[[nodiscard]] static bool is_only_used_by(
    const Value *value, const User *user) noexcept {
    return value != nullptr && value->use_list().count_size() == 1u &&
           value->use_list().front()->user() == user;
}

[[nodiscard]] static ArithmeticInst *as_factorable_producer(
    Value *value, ArithmeticInst *select) noexcept {
    if (value == nullptr || !value->isa<ArithmeticInst>()) {
        return nullptr;
    }
    auto *producer = static_cast<ArithmeticInst *>(value);
    if (!is_only_used_by(producer, select) ||
        !producer->metadata_list().empty() ||
        producer->op() == ArithmeticOp::SELECT ||
        !is_arithmetic_op_safe_to_speculate(producer->op())) {
        return nullptr;
    }
    return producer;
}

[[nodiscard]] static ArithmeticInst *try_factor(
    ArithmeticInst *select, SelectFactorInfo &info,
    luisa::vector<ArithmeticInst *> &worklist) noexcept {
    if (select == nullptr || select->op() != ArithmeticOp::SELECT ||
        select->operand_count() != 3u ||
        !select->metadata_list().empty()) {
        return nullptr;
    }
    auto *condition = select->operand(2u);
    auto *false_producer = as_factorable_producer(
        select->operand(0u), select);
    auto *true_producer = as_factorable_producer(
        select->operand(1u), select);
    if (false_producer == nullptr || true_producer == nullptr ||
        false_producer == true_producer ||
        false_producer->op() != true_producer->op() ||
        false_producer->type() != select->type() ||
        true_producer->type() != select->type() ||
        false_producer->operand_count() !=
            true_producer->operand_count()) {
        return nullptr;
    }
    auto *condition_type = condition == nullptr ?
                               nullptr :
                               condition->type();
    if (condition_type == nullptr ||
        (!condition_type->is_bool() &&
         !is_component_wise(false_producer->op()))) {
        return nullptr;
    }
    auto differing_index = size_t{0u};
    auto differing_count = size_t{0u};
    for (auto i = size_t{0u};
         i < false_producer->operand_count(); i++) {
        auto *false_operand = false_producer->operand(i);
        auto *true_operand = true_producer->operand(i);
        if (false_operand != true_operand) {
            differing_index = i;
            differing_count++;
            if (differing_count > 1u ||
                false_operand == nullptr || true_operand == nullptr ||
                false_operand->type() != true_operand->type() ||
                !select_condition_supports(
                    false_operand->type(), condition_type)) {
                return nullptr;
            }
        }
    }
    if (differing_count != 1u) { return nullptr; }

    XIRBuilder builder;
    builder.set_insertion_point(select);
    luisa::vector<Value *> operands;
    operands.reserve(false_producer->operand_count());
    for (auto i = size_t{0u};
         i < false_producer->operand_count(); i++) {
        if (i == differing_index) {
            auto *operand_select = builder.call(
                false_producer->operand(i)->type(),
                ArithmeticOp::SELECT,
                {false_producer->operand(i),
                 true_producer->operand(i), condition});
            operands.emplace_back(operand_select);
            worklist.emplace_back(operand_select);
        } else {
            operands.emplace_back(false_producer->operand(i));
        }
    }
    auto *factored = builder.call(
        select->type(), false_producer->op(), operands);
    select->replace_all_uses_with(factored);
    select->remove_self();
    false_producer->remove_self();
    true_producer->remove_self();
    info.factored_select_count++;
    info.removed_arithmetic_count += 2u;
    return factored;
}

static void factor_function(
    Function *function, SelectFactorInfo &info) noexcept {
    if (function == nullptr || !function->is_definition()) { return; }
    auto *definition = function->definition();
    if (definition == nullptr || definition->body_block() == nullptr) {
        return;
    }
    luisa::vector<ArithmeticInst *> worklist;
    definition->traverse_instructions([&](Instruction *instruction) noexcept {
        if (instruction->isa<ArithmeticInst>()) {
            auto *arithmetic = static_cast<ArithmeticInst *>(instruction);
            if (arithmetic->op() == ArithmeticOp::SELECT) {
                worklist.emplace_back(arithmetic);
            }
        }
    });
    for (auto i = size_t{0u}; i < worklist.size(); i++) {
        static_cast<void>(try_factor(worklist[i], info, worklist));
    }
}

}// namespace detail

SelectFactorInfo select_factor_pass_run_on_function(
    Function *function) noexcept {
    SelectFactorInfo info;
    detail::factor_function(function, info);
    return info;
}

SelectFactorInfo select_factor_pass_run_on_module(
    Module *module, PassReport *report) noexcept {
    SelectFactorInfo info;
    if (module != nullptr) {
        for (auto *function : module->function_list()) {
            detail::factor_function(function, info);
        }
    }
    if (report != nullptr) {
        report->set("factored-select", info.factored_select_count);
        report->set(
            "removed-arithmetic", info.removed_arithmetic_count);
    }
    return info;
}

}// namespace luisa::compute::xir
