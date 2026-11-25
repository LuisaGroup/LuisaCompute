#include "control_flow_optimization.h"

#include <cstring>
#include <limits>
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
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>

#include "structural_closure.h"

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;
namespace xir = luisa::compute::xir;

namespace {

struct IntegralTypeBounds {
    bool is_signed{false};
    int64_t min_s{0};
    int64_t max_s{0};
    uint64_t max_u{0u};
};

[[nodiscard]] bool get_integral_type_bounds(
    const Type *type, IntegralTypeBounds &bounds) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::INT8:
            bounds = {.is_signed = true,
                      .min_s = std::numeric_limits<int8_t>::min(),
                      .max_s = std::numeric_limits<int8_t>::max()};
            return true;
        case Type::Tag::INT16:
            bounds = {.is_signed = true,
                      .min_s = std::numeric_limits<int16_t>::min(),
                      .max_s = std::numeric_limits<int16_t>::max()};
            return true;
        case Type::Tag::INT32:
            bounds = {.is_signed = true,
                      .min_s = std::numeric_limits<int32_t>::min(),
                      .max_s = std::numeric_limits<int32_t>::max()};
            return true;
        case Type::Tag::INT64:
            bounds = {.is_signed = true,
                      .min_s = std::numeric_limits<int64_t>::min(),
                      .max_s = std::numeric_limits<int64_t>::max()};
            return true;
        case Type::Tag::UINT8:
            bounds = {.is_signed = false,
                      .max_u = std::numeric_limits<uint8_t>::max()};
            return true;
        case Type::Tag::UINT16:
            bounds = {.is_signed = false,
                      .max_u = std::numeric_limits<uint16_t>::max()};
            return true;
        case Type::Tag::UINT32:
            bounds = {.is_signed = false,
                      .max_u = std::numeric_limits<uint32_t>::max()};
            return true;
        case Type::Tag::UINT64:
            bounds = {.is_signed = false,
                      .max_u = std::numeric_limits<uint64_t>::max()};
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool extract_typed_integral_constant(
    const xir::Value *value, const Type *expected_type,
    int64_t &out_s, uint64_t &out_u) noexcept {
    if (value == nullptr || expected_type == nullptr ||
        !value->isa<xir::Constant>()) {
        return false;
    }
    auto *constant = static_cast<const xir::Constant *>(value);
    if (constant->type() != expected_type) { return false; }
    out_s = 0;
    out_u = 0u;
    switch (expected_type->tag()) {
        case Type::Tag::INT8: {
            int8_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_s = v;
            return true;
        }
        case Type::Tag::INT16: {
            int16_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_s = v;
            return true;
        }
        case Type::Tag::INT32: {
            int32_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_s = v;
            return true;
        }
        case Type::Tag::INT64: {
            int64_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_s = v;
            return true;
        }
        case Type::Tag::UINT8: {
            uint8_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_u = v;
            return true;
        }
        case Type::Tag::UINT16: {
            uint16_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_u = v;
            return true;
        }
        case Type::Tag::UINT32: {
            uint32_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_u = v;
            return true;
        }
        case Type::Tag::UINT64: {
            uint64_t v{};
            std::memcpy(&v, constant->data(), sizeof(v));
            out_u = v;
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

// The instruction must be safe to duplicate or hint for unrolling: no
// synchronization/subgroup convergence-sensitive ops, no allocas, no calls,
// no atomics, no diagnostics. Ordinary arithmetic, memory/resource/ray-query
// accesses, and internal structured control flow stay eligible within the
// checked expanded-instruction budget.
[[nodiscard]] bool payload_is_duplicable(const xir::Instruction *instruction) noexcept {
    if (instruction == nullptr) { return false; }
    switch (instruction->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::ARITHMETIC:
        case xir::DerivedInstructionTag::CAST:
        case xir::DerivedInstructionTag::PHI:
        case xir::DerivedInstructionTag::BRANCH:
        case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
        case xir::DerivedInstructionTag::IF:
        case xir::DerivedInstructionTag::SWITCH:
        case xir::DerivedInstructionTag::LOOP:
        case xir::DerivedInstructionTag::SIMPLE_LOOP:
        case xir::DerivedInstructionTag::BREAK:
        case xir::DerivedInstructionTag::CONTINUE:
        case xir::DerivedInstructionTag::GEP:
        case xir::DerivedInstructionTag::LOAD:
        case xir::DerivedInstructionTag::STORE:
        case xir::DerivedInstructionTag::RESOURCE_QUERY:
        case xir::DerivedInstructionTag::RESOURCE_READ:
        case xir::DerivedInstructionTag::RESOURCE_WRITE:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ:
        case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE:
            return true;
        default:
            // ALLOCA, ATOMIC, THREAD_GROUP (barrier, subgroup, split barrier),
            // CALL, OUTLINE, PRINT, CLOCK, ASSERT, RETURN, UNREACHABLE stay
            // rejected.
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

// Collect all blocks belonging to the loop body between `body` and `update`.
// Rejects early exits to `merge`, jumps back to `prepare`, returns/unreachables,
// and any branch/break/continue targeting outside the loop body.
[[nodiscard]] bool collect_loop_body_blocks(
    const xir::LoopInst *loop,
    luisa::vector<const xir::BasicBlock *> &out_blocks) noexcept {
    auto *prepare = loop->prepare_block();
    auto *body = loop->body_block();
    auto *update = loop->update_block();
    auto *merge = loop->merge_block();
    luisa::unordered_set<const xir::BasicBlock *> visited;
    luisa::vector<const xir::BasicBlock *> worklist;
    auto enqueue = [&](const xir::BasicBlock *target) noexcept -> bool {
        if (target == nullptr || target == merge || target == prepare) {
            return false;
        }
        if (target == update) { return true; }
        if (visited.emplace(target).second) {
            worklist.emplace_back(target);
        }
        return true;
    };
    if (!enqueue(body)) { return false; }
    while (!worklist.empty()) {
        auto *bb = worklist.back();
        worklist.pop_back();
        out_blocks.emplace_back(bb);
        auto *term = bb->terminator();
        if (term == nullptr) { return false; }
        switch (term->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::BRANCH: {
                auto *br = static_cast<const xir::BranchInst *>(term);
                if (!enqueue(br->target_block())) { return false; }
                break;
            }
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto *cbr = static_cast<const xir::ConditionalBranchInst *>(term);
                if (!enqueue(cbr->true_block()) || !enqueue(cbr->false_block())) {
                    return false;
                }
                break;
            }
            case xir::DerivedInstructionTag::IF: {
                auto *if_inst = static_cast<const xir::IfInst *>(term);
                if (!enqueue(if_inst->true_block()) ||
                    !enqueue(if_inst->false_block()) ||
                    !enqueue(if_inst->merge_block())) {
                    return false;
                }
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto *sw = static_cast<const xir::SwitchInst *>(term);
                if (!enqueue(sw->default_block()) || !enqueue(sw->merge_block())) {
                    return false;
                }
                for (auto i = 0u; i < sw->case_count(); ++i) {
                    if (!enqueue(sw->case_block(i))) { return false; }
                }
                break;
            }
            case xir::DerivedInstructionTag::LOOP: {
                auto *inner = static_cast<const xir::LoopInst *>(term);
                if (!enqueue(inner->prepare_block()) ||
                    !enqueue(inner->body_block()) ||
                    !enqueue(inner->update_block()) ||
                    !enqueue(inner->merge_block())) {
                    return false;
                }
                break;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                auto *inner = static_cast<const xir::SimpleLoopInst *>(term);
                if (!enqueue(inner->body_block()) ||
                    !enqueue(inner->merge_block())) {
                    return false;
                }
                break;
            }
            case xir::DerivedInstructionTag::BREAK: {
                auto *brk = static_cast<const xir::BreakInst *>(term);
                // Only breaks targeting an inner loop's merge block stay inside
                // the outer loop; breaking to outer `merge` is rejected by enqueue.
                if (!enqueue(brk->target_block())) { return false; }
                break;
            }
            case xir::DerivedInstructionTag::CONTINUE: {
                auto *cont = static_cast<const xir::ContinueInst *>(term);
                if (!enqueue(cont->target_block())) { return false; }
                break;
            }
            default:
                return false;
        }
    }
    return !out_blocks.empty();
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
    luisa::vector<const xir::BasicBlock *> body_blocks;
    if (!collect_loop_body_blocks(loop, body_blocks)) {
        return reject("loop body has an early exit or unsupported control flow");
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
    const xir::Value *bound_value = nullptr;
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
        bound_value = rhs;
    } else if (rhs != nullptr && rhs->isa<xir::PhiInst>()) {
        induction = static_cast<const xir::PhiInst *>(rhs);
        bound_value = lhs;
        relation = flip(compare_op);
    } else {
        return reject("loop condition has no induction Phi operand");
    }
    if (induction->parent_block() != prepare) {
        return reject("induction Phi is not owned by the loop prepare block");
    }
    auto *induction_type = induction->type();
    IntegralTypeBounds bounds{};
    if (!get_integral_type_bounds(induction_type, bounds)) {
        return reject("induction Phi does not have a supported scalar integer type");
    }
    int64_t bound_s = 0;
    uint64_t bound_u = 0u;
    if (!extract_typed_integral_constant(bound_value, induction_type, bound_s, bound_u)) {
        return reject("loop bound is not an exact typed integral constant");
    }

    // Exactly one incoming from the update block (the step) and one from the
    // entry edge (the initial value).
    int64_t initial_s = 0;
    uint64_t initial_u = 0u;
    bool step_is_positive = false;
    uint64_t step_magnitude = 0u;
    auto step_found = false;
    auto initial_found = false;
    for (auto i = 0u; i < induction->incoming_count(); ++i) {
        auto incoming = induction->incoming(i);
        if (incoming.block == update) {
            if (step_found) {
                return reject("induction Phi has more than one backedge incoming");
            }
            auto *next = as_arithmetic(incoming.value);
            if (next == nullptr || next->operand_count() != 2u ||
                next->type() != induction_type) {
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
            int64_t raw_step_s = 0;
            uint64_t raw_step_u = 0u;
            if (!extract_typed_integral_constant(
                    other, induction_type, raw_step_s, raw_step_u)) {
                return reject("induction step is not an exact typed integral constant");
            }
            if (bounds.is_signed) {
                if (raw_step_s == 0) { return reject("induction step is zero"); }
                if (add) {
                    step_is_positive = raw_step_s > 0;
                    step_magnitude = step_is_positive ?
                                         static_cast<uint64_t>(raw_step_s) :
                                         (uint64_t{0} - static_cast<uint64_t>(raw_step_s));
                } else {
                    if (raw_step_s == bounds.min_s) {
                        return reject("signed subtraction step hits minimum type bound");
                    }
                    step_is_positive = raw_step_s < 0;
                    step_magnitude = step_is_positive ?
                                         (uint64_t{0} - static_cast<uint64_t>(raw_step_s)) :
                                         static_cast<uint64_t>(raw_step_s);
                }
            } else {
                if (raw_step_u == 0u) { return reject("induction step is zero"); }
                step_is_positive = add;
                step_magnitude = raw_step_u;
            }
            step_found = true;
        } else {
            if (initial_found) {
                return reject("induction Phi has more than one entry incoming");
            }
            if (!extract_typed_integral_constant(
                    incoming.value, induction_type, initial_s, initial_u)) {
                return reject("induction initial value is not an exact typed constant");
            }
            initial_found = true;
        }
    }
    if (!step_found || !initial_found) {
        return reject("induction Phi is not a two-incoming counted loop");
    }

    // Convert the comparison into an exact trip count in unsigned arithmetic,
    // and prove the induction variable cannot wrap past the type bounds before
    // exiting. A relative bound >= 2^31 is left to the ordinary loop path.
    constexpr auto bound_limit = uint64_t{1} << 31;
    uint64_t dist = 0u;
    uint64_t headroom = 0u;
    const bool increasing =
        relation == xir::ArithmeticOp::BINARY_LESS ||
        relation == xir::ArithmeticOp::BINARY_LESS_EQUAL;
    const bool strict =
        relation == xir::ArithmeticOp::BINARY_LESS ||
        relation == xir::ArithmeticOp::BINARY_GREATER;
    if (increasing) {
        if (!step_is_positive) {
            return reject("increasing comparison with non-positive step");
        }
        if (bounds.is_signed) {
            if (bound_s < initial_s) { return reject("loop trip count is zero"); }
            dist = static_cast<uint64_t>(bound_s) -
                   static_cast<uint64_t>(initial_s);
            headroom = static_cast<uint64_t>(bounds.max_s) -
                       static_cast<uint64_t>(initial_s);
        } else {
            if (bound_u < initial_u) { return reject("loop trip count is zero"); }
            dist = bound_u - initial_u;
            headroom = bounds.max_u - initial_u;
        }
    } else {
        if (step_is_positive) {
            return reject("decreasing comparison with non-negative step");
        }
        if (bounds.is_signed) {
            if (initial_s < bound_s) { return reject("loop trip count is zero"); }
            dist = static_cast<uint64_t>(initial_s) -
                   static_cast<uint64_t>(bound_s);
            headroom = static_cast<uint64_t>(initial_s) -
                       static_cast<uint64_t>(bounds.min_s);
        } else {
            if (initial_u < bound_u) { return reject("loop trip count is zero"); }
            dist = initial_u - bound_u;
            headroom = initial_u;
        }
    }
    if (dist >= bound_limit) {
        return reject("loop trip count exceeds the proof limit");
    }
    uint64_t trip_count = 0u;
    if (strict) {
        if (dist == 0u) { return reject("loop trip count is zero"); }
        trip_count = ((dist - 1u) / step_magnitude) + 1u;
    } else {
        trip_count = (dist / step_magnitude) + 1u;
    }
    if (trip_count == 0u || trip_count > options.max_trip_count) {
        return reject("loop trip count is not an exact small constant");
    }
    if (trip_count > headroom / step_magnitude) {
        return reject("induction variable can wrap before loop exit");
    }

    // No payload may be duplicated unless every instruction in the loop region
    // is safe to duplicate/hint.
    if (!block_payload_is_duplicable(prepare, prepare_terminator) ||
        !block_payload_is_duplicable(update, update_terminator)) {
        return reject("loop payload is not safe to duplicate");
    }
    auto body_instructions = block_instruction_count(prepare) +
                             block_instruction_count(update);
    for (auto *bb : body_blocks) {
        if (!block_payload_is_duplicable(bb, nullptr)) {
            return reject("loop payload is not safe to duplicate");
        }
        body_instructions += block_instruction_count(bb);
    }

    // The expanded body is the checked code-size cost of the decision.
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