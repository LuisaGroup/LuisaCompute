#include "control_flow_optimization.h"

#include <cstring>
#include <utility>

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>

#include "structural_closure.h"

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;
namespace xir = luisa::compute::xir;

namespace {

[[nodiscard]] bool is_integral_constant(const xir::Value *value,
                                        int64_t &result) noexcept {
    if (value == nullptr || !value->isa<xir::Constant>()) { return false; }
    auto *constant = static_cast<const xir::Constant *>(value);
    auto *type = constant->type();
    if (type == nullptr || (!type->is_int() && !type->is_uint())) {
        return false;
    }
    switch (type->size()) {
        case 1u: {
            if (type->is_int()) {
                int8_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            } else {
                uint8_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            }
            return true;
        }
        case 2u: {
            if (type->is_int()) {
                int16_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            } else {
                uint16_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            }
            return true;
        }
        case 4u: {
            if (type->is_int()) {
                int32_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            } else {
                uint32_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            }
            return true;
        }
        case 8u: {
            if (type->is_int()) {
                int64_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                result = v;
            } else {
                uint64_t v{};
                std::memcpy(&v, constant->data(), sizeof(v));
                if (v > static_cast<uint64_t>(INT64_MAX)) { return false; }
                result = static_cast<int64_t>(v);
            }
            return true;
        }
        default: return false;
    }
}

[[nodiscard]] const xir::ArithmeticInst *as_arithmetic(
    const xir::Value *value) noexcept {
    return value != nullptr && value->isa<xir::ArithmeticInst>() ?
               static_cast<const xir::ArithmeticInst *>(value) :
               nullptr;
}

// The instruction must be provably safe to duplicate and speculative: no
// memory, no synchronization, no resource access, no convergence-sensitive
// operation and no call. Anything else keeps the loop at MaskNone.
[[nodiscard]] bool payload_is_duplicable(const xir::Instruction *instruction) noexcept {
    if (instruction == nullptr) { return false; }
    switch (instruction->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::ARITHMETIC:
        case xir::DerivedInstructionTag::CAST:
        case xir::DerivedInstructionTag::PHI:
        case xir::DerivedInstructionTag::BRANCH:
            return true;
        default:
            // ALLOCA, LOAD, STORE, GEP, ATOMIC, THREAD_GROUP (barrier,
            // subgroup, split barrier), RESOURCE_QUERY/READ/WRITE, RAY_QUERY*,
            // CALL, PRINT, CLOCK, ASSERT, and every nested control-flow
            // construct stay rejected.
            return false;
    }
}

[[nodiscard]] bool block_payload_is_duplicable(
    const xir::BasicBlock *block,
    const xir::Instruction *expected_terminator) noexcept {
    if (block == nullptr) { return false; }
    for (auto *instruction : block->instructions()) {
        if (instruction == expected_terminator) { continue; }
        if (!payload_is_duplicable(instruction)) { return false; }
    }
    return true;
}

[[nodiscard]] size_t block_instruction_count(const xir::BasicBlock *block) noexcept {
    if (block == nullptr) { return 0u; }
    auto count = size_t{0u};
    for (auto *instruction : block->instructions()) {
        static_cast<void>(instruction);
        count++;
    }
    return count;
}

[[nodiscard]] SpirvLoopControlDecision reject(luisa::string reason) noexcept {
    SpirvLoopControlDecision decision;
    decision.reason = std::move(reason);
    return decision;
}

}// namespace

ControlFlowOptimizationPlan::Options
ControlFlowOptimizationPlan::Options::from_environment() noexcept {
    Options options;
    // A/B switch for measurement: `LUISA_SPIRV_LOOP_HINTS=0` keeps every loop at
    // MaskNone. The value is part of the shader cache identity (see
    // xir_spirv_environment_hash in the Vulkan backend).
    if (auto value = luisa::get_environment_variable("LUISA_SPIRV_LOOP_HINTS")) {
        auto text = luisa::string_view{*value};
        if (text == "0" || text == "false" || text == "FALSE") {
            options.enable_loop_unroll = false;
        }
    }
    return options;
}

SpirvLoopControlDecision plan_spirv_loop_control(
    const xir::LoopInst *loop,
    ControlFlowOptimizationPlan::Options options) noexcept {
    if (loop == nullptr) { return reject("missing loop"); }
    if (!options.enable_loop_unroll) {
        return reject("loop unrolling disabled by policy");
    }
    auto *prepare = loop->prepare_block();
    auto *body = loop->body_block();
    auto *update = loop->update_block();
    auto *merge = loop->merge_block();
    if (prepare == nullptr || body == nullptr || update == nullptr ||
        merge == nullptr) {
        return reject("malformed structural loop roles");
    }
    if (prepare == body || prepare == update || body == update ||
        merge == prepare || merge == body || merge == update) {
        return reject("malformed structural loop roles");
    }

    // A counted loop needs exactly one entry edge and one backedge through the
    // declared update block.
    auto *prepare_terminator = prepare->terminator();
    if (prepare_terminator == nullptr ||
        prepare_terminator->derived_instruction_tag() !=
            xir::DerivedInstructionTag::CONDITIONAL_BRANCH) {
        return reject("loop prepare is not a conditional branch");
    }
    auto *condition_branch =
        static_cast<const xir::ConditionalBranchInst *>(prepare_terminator);
    if (condition_branch->true_block() != body ||
        condition_branch->false_block() != merge) {
        return reject("noncanonical loop prepare targets");
    }
    auto *body_terminator = body->terminator();
    if (body_terminator == nullptr ||
        body_terminator->derived_instruction_tag() !=
            xir::DerivedInstructionTag::BRANCH) {
        return reject("loop body has an early exit or nested construct");
    }
    if (static_cast<const xir::BranchInst *>(body_terminator)->target_block() !=
        update) {
        return reject("loop body does not branch to its update block");
    }
    auto *update_terminator = update->terminator();
    if (update_terminator == nullptr ||
        update_terminator->derived_instruction_tag() !=
            xir::DerivedInstructionTag::BRANCH) {
        return reject("loop update has an early exit or nested construct");
    }
    if (static_cast<const xir::BranchInst *>(update_terminator)->target_block() !=
        prepare) {
        return reject("loop update does not branch back to its prepare block");
    }

    // The prepare terminator must compare a single induction Phi against an
    // exact integral constant bound.
    auto *comparison = as_arithmetic(condition_branch->condition());
    if (comparison == nullptr || comparison->operand_count() != 2u) {
        return reject("loop condition is not a two-operand arithmetic compare");
    }
    auto compare_op = comparison->op();
    auto is_ordered_compare =
        compare_op == xir::ArithmeticOp::BINARY_LESS ||
        compare_op == xir::ArithmeticOp::BINARY_LESS_EQUAL ||
        compare_op == xir::ArithmeticOp::BINARY_GREATER ||
        compare_op == xir::ArithmeticOp::BINARY_GREATER_EQUAL;
    if (!is_ordered_compare) {
        return reject("loop condition is not an ordered comparison");
    }

    const xir::PhiInst *induction = nullptr;
    int64_t bound = 0;
    // Normalize the comparison to `phi <relation> bound`; a bound on the left
    // side flips the relation.
    auto relation = compare_op;
    auto flip = [](xir::ArithmeticOp op) noexcept {
        switch (op) {
            case xir::ArithmeticOp::BINARY_LESS:
                return xir::ArithmeticOp::BINARY_GREATER;
            case xir::ArithmeticOp::BINARY_LESS_EQUAL:
                return xir::ArithmeticOp::BINARY_GREATER_EQUAL;
            case xir::ArithmeticOp::BINARY_GREATER:
                return xir::ArithmeticOp::BINARY_LESS;
            case xir::ArithmeticOp::BINARY_GREATER_EQUAL:
                return xir::ArithmeticOp::BINARY_LESS_EQUAL;
            default: return op;
        }
    };
    auto *lhs = comparison->operand(0u);
    auto *rhs = comparison->operand(1u);
    if (lhs != nullptr && lhs->isa<xir::PhiInst>()) {
        induction = static_cast<const xir::PhiInst *>(lhs);
        if (!is_integral_constant(rhs, bound)) {
            return reject("loop bound is not an exact integral constant");
        }
    } else if (rhs != nullptr && rhs->isa<xir::PhiInst>()) {
        induction = static_cast<const xir::PhiInst *>(rhs);
        if (!is_integral_constant(lhs, bound)) {
            return reject("loop bound is not an exact integral constant");
        }
        relation = flip(compare_op);
    } else {
        return reject("loop condition has no induction Phi operand");
    }
    if (induction->parent_block() != prepare) {
        return reject("induction Phi is not owned by the loop prepare block");
    }

    // Exactly one incoming from the update block (the step) and one from the
    // entry edge (the initial value).
    int64_t initial = 0;
    int64_t step = 0;
    auto step_found = false;
    auto initial_found = false;
    for (auto i = 0u; i < induction->incoming_count(); ++i) {
        auto incoming = induction->incoming(i);
        if (incoming.block == update) {
            if (step_found) {
                return reject("induction Phi has more than one backedge incoming");
            }
            auto *next = as_arithmetic(incoming.value);
            if (next == nullptr || next->operand_count() != 2u) {
                return reject("induction step is not a two-operand arithmetic op");
            }
            auto *step_lhs = next->operand(0u);
            auto *step_rhs = next->operand(1u);
            const xir::Value *other = nullptr;
            auto add = next->op() == xir::ArithmeticOp::BINARY_ADD;
            auto sub = next->op() == xir::ArithmeticOp::BINARY_SUB;
            if (!add && !sub) {
                return reject("induction step is not an add or subtract");
            }
            if (step_lhs == induction) {
                other = step_rhs;
            } else if (add && step_rhs == induction) {
                other = step_lhs;
            } else {
                return reject("induction step does not use the induction Phi");
            }
            if (!is_integral_constant(other, step)) {
                return reject("induction step is not an exact integral constant");
            }
            if (sub) { step = -step; }
            if (step == 0) { return reject("induction step is zero"); }
            step_found = true;
        } else {
            if (initial_found) {
                return reject("induction Phi has more than one entry incoming");
            }
            if (!is_integral_constant(incoming.value, initial)) {
                return reject("induction initial value is not an exact constant");
            }
            initial_found = true;
        }
    }
    if (!step_found || !initial_found) {
        return reject("induction Phi is not a two-incoming counted loop");
    }

    // Convert the comparison into an exact trip count. A relative bound larger
    // than 32 bits is left to the ordinary loop path rather than unrolled.
    constexpr auto bound_limit = int64_t{1} << 31;
    auto delta = int64_t{0};
    uint64_t trip_count = 0;
    if (relation == xir::ArithmeticOp::BINARY_LESS ||
        relation == xir::ArithmeticOp::BINARY_LESS_EQUAL) {
        if (step <= 0) { return reject("increasing comparison with non-positive step"); }
        if (bound < initial) { return reject("loop trip count is zero"); }
        delta = bound - initial;
        if (delta >= bound_limit) { return reject("loop trip count exceeds the proof limit"); }
        trip_count = relation == xir::ArithmeticOp::BINARY_LESS ?
                         static_cast<uint64_t>((delta + step - 1) / step) :
                         static_cast<uint64_t>(delta / step + 1);
    } else {
        if (step >= 0) { return reject("decreasing comparison with non-negative step"); }
        if (initial < bound) { return reject("loop trip count is zero"); }
        delta = initial - bound;
        if (delta >= bound_limit) { return reject("loop trip count exceeds the proof limit"); }
        auto magnitude = -step;
        trip_count = relation == xir::ArithmeticOp::BINARY_GREATER ?
                         static_cast<uint64_t>((delta + magnitude - 1) / magnitude) :
                         static_cast<uint64_t>(delta / magnitude + 1);
    }
    if (trip_count == 0u || trip_count > options.max_trip_count) {
        return reject("loop trip count is not an exact small constant");
    }

    // No payload may be duplicated unless it is pure value arithmetic.
    if (!block_payload_is_duplicable(prepare, prepare_terminator) ||
        !block_payload_is_duplicable(body, body_terminator) ||
        !block_payload_is_duplicable(update, update_terminator)) {
        return reject("loop payload is not safe to duplicate");
    }

    // The expanded body is the checked code-size cost of the decision.
    auto body_instructions = block_instruction_count(prepare) +
                             block_instruction_count(body) +
                             block_instruction_count(update);
    auto budget = static_cast<uint64_t>(body_instructions) * trip_count;
    if (budget > options.max_expanded_instructions) {
        return reject("unrolled loop exceeds the code-size budget");
    }

    SpirvLoopControlDecision decision;
    decision.mask = spv::LoopControlMask::Unroll;
    decision.trip_count = trip_count;
    decision.expanded_instruction_budget = budget;
    decision.reason = "exact constant trip count within the code-size budget";
    return decision;
}

ControlFlowOptimizationPlan ControlFlowOptimizationPlan::create(
    const xir::FunctionDefinition *function,
    Options options) noexcept {
    ControlFlowOptimizationPlan plan;
    if (function == nullptr || !options.enable_loop_unroll) { return plan; }
    auto closure = collect_spirv_codegen_structural_closure(function);
    for (auto *block : closure) {
        for (auto *instruction : block->instructions()) {
            if (instruction == nullptr ||
                instruction->derived_instruction_tag() !=
                    xir::DerivedInstructionTag::LOOP) {
                continue;
            }
            auto *loop = static_cast<const xir::LoopInst *>(instruction);
            plan._loop_decisions.emplace(
                loop, plan_spirv_loop_control(loop, options));
        }
    }
    return plan;
}

const SpirvLoopControlDecision &ControlFlowOptimizationPlan::loop_control(
    const xir::LoopInst *loop) const noexcept {
    auto iter = _loop_decisions.find(loop);
    return iter == _loop_decisions.end() ? _default_decision : iter->second;
}

}// namespace lc::spirv