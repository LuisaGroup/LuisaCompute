#include <luisa/xir/passes/loop_rotation.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/phi.h>

#include "helpers.h"
#include "natural_loop.h"

namespace luisa::compute::xir {

namespace detail {

namespace {

// Whether the loop-exit condition can be re-evaluated outside the header by
// substituting every header phi operand with a mapped value. Supported
// shapes: a header phi itself, a single arithmetic instruction in the header
// whose operands are phis or loop-invariant values, or a loop-invariant
// value directly.
[[nodiscard]] bool condition_is_clonable(Value *condition, BasicBlock *header,
                                         const luisa::unordered_map<PhiInst *, Value *> &substitution) noexcept {
    if (condition == nullptr) { return false; }
    auto is_supported_operand = [&](Value *operand) noexcept {
        if (operand == nullptr) { return false; }
        if (!operand->isa<Instruction>()) { return true; }
        auto *inst = static_cast<Instruction *>(operand);
        if (inst->parent_block() != header) { return true; }
        return inst->isa<PhiInst>() &&
               substitution.contains(static_cast<PhiInst *>(inst));
    };
    if (!condition->isa<Instruction>()) { return true; }
    auto *inst = static_cast<Instruction *>(condition);
    if (inst->parent_block() != header) { return true; }
    if (inst->isa<PhiInst>()) {
        return substitution.contains(static_cast<PhiInst *>(inst));
    }
    if (!inst->isa<ArithmeticInst>()) { return false; }
    for (auto *use : inst->operand_uses()) {
        if (!is_supported_operand(use->value())) { return false; }
    }
    return true;
}

// Clone the loop-exit condition into the builder's current block,
// substituting header phi operands with their mapped values. Must only be
// called after condition_is_clonable succeeded.
[[nodiscard]] Value *clone_condition_with_substitution(
    XIRBuilder &builder, Value *condition, BasicBlock *header,
    const luisa::unordered_map<PhiInst *, Value *> &substitution) noexcept {
    auto substitute = [&](Value *operand) noexcept -> Value * {
        if (operand != nullptr && operand->isa<PhiInst>() &&
            static_cast<PhiInst *>(operand)->parent_block() == header) {
            auto iter = substitution.find(static_cast<PhiInst *>(operand));
            if (iter != substitution.end()) { return iter->second; }
        }
        return operand;
    };
    if (!condition->isa<ArithmeticInst>() ||
        static_cast<ArithmeticInst *>(condition)->parent_block() != header) {
        return substitute(condition);
    }
    auto *arith = static_cast<ArithmeticInst *>(condition);
    luisa::vector<Value *> operands;
    operands.reserve(arith->operand_count());
    for (auto i = 0u; i < arith->operand_count(); ++i) {
        operands.emplace_back(substitute(arith->operand(i)));
    }
    return builder.call(arith->type(), arith->op(), operands);
}

[[nodiscard]] bool loop_has_external_value_uses(const NaturalLoop &loop) noexcept {
    auto external_use = false;
    auto check_block = [&](BasicBlock *block) noexcept {
        block->traverse_instructions([&](Instruction *inst) noexcept {
            if (external_use) { return; }
            for (auto *use : inst->use_list()) {
                auto *user = use->user();
                if (user != nullptr && user->isa<Instruction>()) {
                    auto *user_block = static_cast<Instruction *>(user)->parent_block();
                    if (user_block != nullptr && !loop.contains(user_block)) {
                        external_use = true;
                        return;
                    }
                }
            }
        });
    };
    check_block(loop.header);
    for (auto *block : loop.body_blocks) { check_block(block); }
    return external_use;
}

[[nodiscard]] bool exit_blocks_have_phis(const NaturalLoop &loop) noexcept {
    for (auto *exit : loop.exit_blocks) {
        auto has_phi = false;
        exit->traverse_instructions([&](Instruction *inst) noexcept {
            has_phi = has_phi || inst->isa<PhiInst>();
        });
        if (has_phi) { return true; }
    }
    return false;
}

// Rotation executes the original header only for iterations whose guard
// succeeds. Therefore every non-Phi header instruction other than the branch
// must be observationally pure: otherwise the zero-trip path, and the final
// failing top check, would lose side effects.
[[nodiscard]] bool header_is_speculatable(BasicBlock *header) noexcept {
    for (auto *inst : header->instructions()) {
        if (inst->isa<PhiInst>() || inst->is_terminator()) { continue; }
        if (!get_memory_info(inst).is_pure()) { return false; }
    }
    return true;
}

[[nodiscard]] bool try_rotate_loop(FunctionDefinition *def, const NaturalLoop &loop) noexcept {
    auto *header = loop.header;
    // Require a canonical shape: a preheader, exactly one latch, and a single
    // exit reached from the header's conditional branch.
    if (loop.preheader == nullptr || loop.latches.size() != 1u ||
        loop.exit_blocks.size() != 1u || loop.exit_edges.size() != 1u ||
        loop.exit_edges.front().first != loop.header) {
        return false;
    }
    auto *preheader = loop.preheader;
    auto *latch = loop.latches.front();
    auto *exit_block = loop.exit_blocks.front();
    auto *terminator = header->terminator();
    if (terminator == nullptr || !terminator->isa<ConditionalBranchInst>()) {
        return false;
    }
    auto *header_branch = static_cast<ConditionalBranchInst *>(terminator);
    auto *true_target = header_branch->true_block();
    auto *false_target = header_branch->false_block();
    // Identify the in-loop (body) and out-of-loop (exit) successors.
    BasicBlock *body_target = nullptr;
    if (true_target == exit_block && false_target != exit_block) {
        body_target = false_target;
    } else if (false_target == exit_block && true_target != exit_block) {
        body_target = true_target;
    } else {
        return false;
    }
    if (!loop.contains(body_target) || loop.contains(exit_block)) { return false; }
    if (loop.exit_edges.front().second != exit_block) { return false; }
    // The preheader must unconditionally branch to the header, and the latch
    // must unconditionally branch back to it.
    auto *preheader_terminator = preheader->terminator();
    if (preheader_terminator == nullptr ||
        !preheader_terminator->isa<BranchInst>() ||
        static_cast<BranchInst *>(preheader_terminator)->target_block() != header) {
        return false;
    }
    if (latch != header) {
        auto *latch_terminator = latch->terminator();
        if (latch_terminator == nullptr || !latch_terminator->isa<BranchInst>() ||
            static_cast<BranchInst *>(latch_terminator)->target_block() != header) {
            return false;
        }
    }
    // Reject loops whose values escape (the guard path would bypass their
    // definitions) or whose exit blocks carry phis to rewrite.
    if (!header_is_speculatable(header) ||
        loop_has_external_value_uses(loop) ||
        exit_blocks_have_phis(loop)) {
        return false;
    }

    auto *condition = header_branch->condition();
    // Build the phi substitutions: for the guard, each header phi maps to its
    // preheader incoming value; for the latch, to its latch incoming value.
    luisa::unordered_map<PhiInst *, Value *> guard_substitution;
    luisa::unordered_map<PhiInst *, Value *> latch_substitution;
    for (auto *inst : header->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        Value *from_preheader = nullptr;
        Value *from_latch = nullptr;
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            auto incoming = phi->incoming(i);
            if (incoming.block == preheader) { from_preheader = incoming.value; }
            if (incoming.block == latch) { from_latch = incoming.value; }
        }
        if (from_preheader == nullptr || from_latch == nullptr) { return false; }
        guard_substitution.emplace(phi, from_preheader);
        latch_substitution.emplace(phi, from_latch);
    }
    if (!condition_is_clonable(condition, header, guard_substitution) ||
        !condition_is_clonable(condition, header, latch_substitution)) {
        return false;
    }
    // Rotation replaces the header/latch branch roles and clones the
    // condition into two dynamic sites. There is no generally correct owner
    // for semantic metadata attached to those values after the rewrite.
    if (!header_branch->metadata_list().empty() ||
        (condition != nullptr &&
         !condition->metadata_list().empty()) ||
        (latch != header &&
         !latch->terminator()->metadata_list().empty())) {
        return false;
    }
    if (latch == header && header_branch->prev() == nullptr) {
        // A single-block loop needs an insertion point before its terminator
        // for the next-iteration condition clone.
        return false;
    }

    // All checks passed; mutate. First the guard: it evaluates the condition
    // once before the first iteration and branches to the header or the exit.
    // Preserve the original arm order: when the body was the false target,
    // the condition selects the exit on true and the loop must mirror that.
    auto body_is_true_target = body_target == true_target;
    XIRBuilder builder;
    auto *guard = def->create_basic_block();
    guard->set_name("rotated_guard");
    builder.set_insertion_point(guard);
    auto *guard_condition = clone_condition_with_substitution(
        builder, condition, header, guard_substitution);
    if (body_is_true_target) {
        builder.cond_br(guard_condition, header, exit_block);
    } else {
        builder.cond_br(guard_condition, exit_block, header);
    }

    // Retarget the preheader to the guard and update the header phis to
    // enter from the guard instead of the preheader.
    static_cast<BranchInst *>(preheader_terminator)->set_target_block(guard);
    for (auto *inst : header->instructions()) {
        if (!inst->isa<PhiInst>()) { break; }
        auto *phi = static_cast<PhiInst *>(inst);
        for (auto i = 0u; i < phi->incoming_count(); ++i) {
            if (phi->incoming(i).block == preheader) {
                auto value = phi->incoming(i).value;
                phi->set_incoming(i, value, guard);
            }
        }
    }

    if (latch == header) {
        // Single-block loop: the bottom check replaces the top check in
        // place; the original condition computation becomes dead.
        builder.set_insertion_point(header_branch->prev());
        auto *latch_condition = clone_condition_with_substitution(
            builder, condition, header, latch_substitution);
        header_branch->set_condition(latch_condition);
    } else {
        // The header unconditionally enters the body...
        builder.set_insertion_point(header_branch);
        builder.br(body_target);
        static_cast<void>(header_branch->remove_self());
        // ...and the latch performs the bottom check.
        auto *latch_terminator = latch->terminator();
        builder.set_insertion_point(latch_terminator);
        auto *latch_condition = clone_condition_with_substitution(
            builder, condition, header, latch_substitution);
        if (body_is_true_target) {
            builder.cond_br(latch_condition, header, exit_block);
        } else {
            builder.cond_br(latch_condition, exit_block, header);
        }
        static_cast<void>(latch_terminator->remove_self());
    }
    return true;
}

}// namespace

static void loop_rotation_run(FunctionDefinition *def, LoopRotationInfo &info) noexcept {
    if (def == nullptr) { return; }
    if (contains_structured_control_flow(def)) {
        ++info.structured_cfg_error_count;
        LUISA_WARNING_WITH_LOCATION(
            "Loop rotation rejected structured CFG; run destructure_cfg first. "
            "IR was left unchanged.");
        return;
    }
    auto dom_tree = compute_dom_tree(def);
    auto loops = discover_natural_loops(def, dom_tree);
    // Only innermost loops are rotated. A rotated (bottom-checked) loop
    // round-trips through restructure_cfg as a simple_loop with a break, but
    // a rotated outer loop around a nested inner one would require the
    // for-loop (prepare/body/update) formation path to preserve latch
    // conditions, which it currently does not support.
    for (auto &loop : loops) {
        auto contains_nested_loop = false;
        for (auto &other : loops) {
            if (&other == &loop) { continue; }
            if (loop.contains(other.header)) {
                contains_nested_loop = true;
                break;
            }
        }
        if (contains_nested_loop) { continue; }
        if (try_rotate_loop(def, loop)) {
            ++info.rotated_loop_count;
        }
    }
}

[[nodiscard]] static bool loop_rotation_preflight_module(
    Module *module, LoopRotationInfo &info) noexcept {
    if (module == nullptr) { return true; }
    for (auto *function : module->function_list()) {
        auto *def = function == nullptr ? nullptr : function->definition();
        if (def != nullptr && contains_structured_control_flow(def)) {
            ++info.structured_cfg_error_count;
        }
    }
    if (info.structured_cfg_error_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "Loop rotation rejected a module containing structured CFG; "
            "run destructure_cfg first. The entire module was left unchanged.");
        return false;
    }
    return true;
}

}// namespace detail

LoopRotationInfo loop_rotation_pass_run_on_function(FunctionDefinition *def) noexcept {
    LoopRotationInfo info;
    detail::loop_rotation_run(def, info);
    return info;
}

LoopRotationInfo loop_rotation_pass_run_on_module(Module *module,
                                                  PassReport *report) noexcept {
    LoopRotationInfo info;
    if (detail::loop_rotation_preflight_module(module, info)) {
        if (module != nullptr) {
            for (auto *function : module->function_list()) {
                detail::loop_rotation_run(function == nullptr ? nullptr :
                                                    function->definition(),
                            info);
            }
        }
    }
    if (report != nullptr) {
        report->set("rotated_loop_count", info.rotated_loop_count);
        report->set("structured_cfg_error_count", info.structured_cfg_error_count);
    }
    return info;
}

}// namespace luisa::compute::xir
