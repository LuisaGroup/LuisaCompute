#include "entry.h"
#include <algorithm>
#include <luisa/core/logging.h>

namespace lc::spirv {

namespace {

[[nodiscard]] bool block_reaches(const xir::BasicBlock *from,
                                 const xir::BasicBlock *to) noexcept {
    if (from == nullptr || to == nullptr) { return false; }
    if (from == to) { return true; }
    luisa::unordered_set<const xir::BasicBlock *> visited;
    luisa::vector<const xir::BasicBlock *> worklist;
    worklist.emplace_back(from);
    while (!worklist.empty()) {
        auto *bb = worklist.back();
        worklist.pop_back();
        if (bb == nullptr || !visited.emplace(bb).second) { continue; }
        bool found = false;
        bb->traverse_successors(true, [&](const xir::BasicBlock *succ) noexcept {
            if (succ == to) {
                found = true;
            } else if (!visited.contains(succ)) {
                worklist.emplace_back(succ);
            }
        });
        if (found) { return true; }
    }
    return false;
}

[[nodiscard]] bool block_has_predecessors(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return false; }
    bool has = false;
    bb->traverse_predecessors(true, [&](const xir::BasicBlock *) noexcept {
        has = true;
    });
    return has;
}

[[nodiscard]] bool block_reaches_excluding(const xir::BasicBlock *from,
                                           const xir::BasicBlock *to,
                                           const xir::BasicBlock *exclude) noexcept {
    if (from == nullptr || to == nullptr) { return false; }
    if (from == to) { return true; }
    if (from == exclude) { return false; }
    luisa::unordered_set<const xir::BasicBlock *> visited;
    luisa::vector<const xir::BasicBlock *> worklist;
    worklist.emplace_back(from);
    while (!worklist.empty()) {
        auto *bb = worklist.back();
        worklist.pop_back();
        if (bb == nullptr || !visited.emplace(bb).second || bb == exclude) { continue; }
        bool found = false;
        bb->traverse_successors(true, [&](const xir::BasicBlock *succ) noexcept {
            if (succ == to) {
                found = true;
            } else if (!visited.contains(succ) && succ != exclude) {
                worklist.emplace_back(succ);
            }
        });
        if (found) { return true; }
    }
    return false;
}

[[nodiscard]] const xir::BasicBlock *single_forward_target(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr) { return nullptr; }
    auto &insts = bb->instructions();
    if (insts.empty() || insts.front() != insts.back()) { return nullptr; }
    auto *inst = insts.front();
    switch (inst->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::BRANCH:
            return static_cast<const xir::BranchInst *>(inst)->target_block();
        case xir::DerivedInstructionTag::BREAK:
            return static_cast<const xir::BreakInst *>(inst)->target_block();
        case xir::DerivedInstructionTag::CONTINUE:
            return static_cast<const xir::ContinueInst *>(inst)->target_block();
        default: break;
    }
    return nullptr;
}

[[nodiscard]] bool block_is_terminal_exit(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr || !bb->is_terminated()) { return false; }
    switch (bb->terminator()->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::RETURN:
        case xir::DerivedInstructionTag::UNREACHABLE:
        case xir::DerivedInstructionTag::RASTER_DISCARD:
            return true;
        default: break;
    }
    return false;
}

}// namespace

spv::Block *SpirvCodegenEntry::_loop_boundary_forward_target(
    const xir::BasicBlock *bb,
    bool preserve_active_selection_merges) noexcept {
    if (bb == nullptr || _loop_boundary_stack.empty()) { return nullptr; }
    luisa::unordered_set<const xir::BasicBlock *> visited;
    for (auto *current = bb; current != nullptr && visited.emplace(current).second;) {
        if (preserve_active_selection_merges) {
            if (auto it = _block_map.find(current); it != _block_map.end() &&
                                                    !_outer_merge_stack.empty() &&
                                                    it->second == _outer_merge_stack.back()) {
                return nullptr;
            }
        }
        if (auto it = _loop_header_redirect.find(current); it != _loop_header_redirect.end()) {
            return it->second;
        }
        for (auto iter = _loop_boundary_stack.rbegin(); iter != _loop_boundary_stack.rend(); ++iter) {
            if (iter->first == current) { return iter->second; }
        }
        current = single_forward_target(current);
    }
    return nullptr;
}

spv::Block *SpirvCodegenEntry::_direct_loop_boundary_forward_target(const xir::BasicBlock *bb) noexcept {
    if (bb == nullptr || _loop_boundary_stack.empty()) { return nullptr; }
    if (auto it = _loop_header_redirect.find(bb); it != _loop_header_redirect.end()) {
        return it->second;
    }
    for (auto iter = _loop_boundary_stack.rbegin(); iter != _loop_boundary_stack.rend(); ++iter) {
        if (iter->first == bb) { return iter->second; }
    }
    if (auto *single_forward = single_forward_target(bb)) {
        if (auto it = _loop_header_redirect.find(single_forward); it != _loop_header_redirect.end()) {
            return it->second;
        }
        for (auto iter = _loop_boundary_stack.rbegin(); iter != _loop_boundary_stack.rend(); ++iter) {
            if (iter->first == single_forward) { return iter->second; }
        }
    }
    return nullptr;
}

bool SpirvCodegenEntry::_is_direct_structured_branch_target(const xir::BasicBlock *target) const noexcept {
    if (target == nullptr) { return false; }
    if (_loop_header_redirect.find(target) != _loop_header_redirect.end()) {
        return true;
    }
    for (auto &&[xb, _] : _loop_boundary_stack) {
        if (xb == target) { return true; }
    }
    return false;
}

spv::Block *SpirvCodegenEntry::_resolve_branch_target(const xir::BasicBlock *target) noexcept {
    if (target == nullptr) { return nullptr; }
    if (auto it = _branch_target_redirect.find(target); it != _branch_target_redirect.end()) {
        if (it->second != nullptr) {
            return it->second;
        }
        _branch_target_redirect.erase(it);
    }
    if (auto it = _loop_header_redirect.find(target); it != _loop_header_redirect.end()) {
        if (it->second != nullptr) { return it->second; }
    }
    if (auto *forward_target = _loop_boundary_forward_target(target, true)) {
        if (!_is_direct_structured_branch_target(target)) {
            _forwarded_blocks.emplace(target);
        }
        return forward_target;
    }
    return _get_or_create_block(target);
}

// ── Structured If → SPIR-V SelectionMerge ──────────────────────────────
// Emit a structured if-else as a SPIR-V OpSelectionMerge + OpBranchConditional.
//
// Design invariants:
//  1. Post-dominance determines whether the declared merge block is the real
//     SPIR-V selection merge, a synthetic dead-end merge, or a fall-through.
//  2. Nested constructs whose arm coincides with an outer merge block get a
//     synthetic clone to avoid dominance-ordering violations.
//  3. The real merge block is emitted lazily: only when it is the actual
//     continuation of this selection, or it has no predecessors.  Otherwise it
//     is emitted later when its other predecessor(s) are processed.
//  4. Blocks that forward to a loop boundary or are already used as another
//     construct's merge are tracked via _branch_target_redirect, _used_merge_blocks,
//     and _forwarded_blocks to prevent duplicate emission.
//
// Helper lambdas:
//  - bind_or_get: map XIR block -> SPIR-V block (create if new).
//  - make_arm_target: clone arm if it coincides with an outer merge.
//  - emit_arm: recursively emit an arm and its successors, then branch to target.
//
void SpirvCodegenEntry::_emit_if_inst(const xir::IfInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto &function = _builder.getBuildPoint()->getParent();
    auto bind_or_get = [&](const xir::BasicBlock *xb, bool &is_fresh) -> spv::Block * {
        if (auto it = _block_map.find(xb); it != _block_map.end()) {
            is_fresh = false;
            return it->second;
        }
        is_fresh = true;
        auto blk = new spv::Block(_builder.getUniqueId(), function);
        _block_map.emplace(xb, blk);
        return blk;
    };
    bool true_fresh = false, false_fresh = false, merge_fresh = false;
    auto true_block = bind_or_get(inst->true_block(), true_fresh);
    auto false_block = bind_or_get(inst->false_block(), false_fresh);
    auto merge_block = bind_or_get(inst->merge_block(), merge_fresh);
    // Determine whether the declared merge block actually post-dominates each
    // arm.  A proper two-sided if has the merge block strictly post-dominating
    // both arms.  If the merge only post-dominates one arm, the other arm is
    // the fall-through continuation and the SPIR-V selection merge must be
    // that arm's block.
    auto strictly_post_dominates = [&](const xir::BasicBlock *a, const xir::BasicBlock *b) noexcept {
        auto *mutable_a = const_cast<xir::BasicBlock *>(a);
        auto *mutable_b = const_cast<xir::BasicBlock *>(b);
        return _dom_tree != nullptr &&
               _dom_tree->contains(mutable_a) &&
               _dom_tree->contains(mutable_b) &&
               _dom_tree->strictly_post_dominates(mutable_a, mutable_b);
    };
    bool true_post_dom = inst->true_block() == inst->merge_block() ||
                         strictly_post_dominates(inst->merge_block(), inst->true_block());
    bool false_post_dom = inst->false_block() == inst->merge_block() ||
                          strictly_post_dominates(inst->merge_block(), inst->false_block());
    auto true_reaches_merge = block_reaches(inst->true_block(), inst->merge_block());
    auto false_reaches_merge = block_reaches(inst->false_block(), inst->merge_block());
    if (true_reaches_merge && false_reaches_merge && (true_post_dom || false_post_dom)) {
        true_post_dom = true;
        false_post_dom = true;
    }
    bool merge_already_used = _used_merge_blocks.contains(merge_block->getId());

    // Compute the ultimate block the declared merge block forwards to by
    // following a chain of empty forward blocks (e.g. merge -> next -> ...).
    // This identifies the true structural continuation of the construct when
    // one arm bypasses the declared merge block entirely.
    const xir::BasicBlock *merge_ultimate_forward = nullptr;
    {
        luisa::unordered_set<const xir::BasicBlock *> forward_seen;
        auto *forward_cursor = inst->merge_block();
        forward_seen.emplace(forward_cursor);
        while (auto *next = single_forward_target(forward_cursor)) {
            if (!forward_seen.emplace(next).second) { break; }// cycle guard
            forward_cursor = next;
        }
        if (forward_cursor != inst->merge_block() &&
            forward_cursor != inst->true_block() &&
            forward_cursor != inst->false_block()) {
            merge_ultimate_forward = forward_cursor;
        }
    }
    // An arm "bypasses" the declared merge when the merge does not reach the
    // arm, but the arm still converges to the merge's ultimate forward target
    // (which then post-dominates the arm).  In this situation neither the
    // fall-through scheme (the arm is not the continuation) nor a synthetic
    // dead-end merge is structurally correct; instead the declared merge is
    // kept as the SPIR-V selection merge and the forward target is redirected
    // back to the merge while the bypassing arm is emitted, so nested merge
    // blocks chain through this merge instead of jumping past the whole
    // construct.
    auto arm_bypasses_merge = [&](const xir::BasicBlock *arm) noexcept {
        return merge_ultimate_forward != nullptr &&
               !block_reaches(inst->merge_block(), arm) &&
               block_reaches(arm, merge_ultimate_forward) &&
               _dom_tree != nullptr &&
               _dom_tree->strictly_post_dominates(
                   const_cast<xir::BasicBlock *>(merge_ultimate_forward),
                   const_cast<xir::BasicBlock *>(arm));
    };
    bool bypass_with_merge_redirect = false;
    bool bypass_true_arm = false;

    // In a normal two-sided if both arms reach the merge block and we can use
    // it as the SPIR-V selection merge.  If neither arm reaches it, we need a
    // synthetic dead-end merge.  If exactly one arm reaches it, the other arm
    // is the fall-through continuation: use that arm's block as the SPIR-V
    // selection merge and make the reaching arm branch to the original merge.
    // When the merge block coincides with one arm and the other arm is a
    // terminal exit (return/unreachable/discard), the merge arm is the live
    // continuation of the construct rather than a dead end, so no synthetic
    // dead-end merge is needed — the merge arm is emitted with its content.
    bool non_merge_arm_terminal =
        (inst->merge_block() == inst->true_block() && block_is_terminal_exit(inst->false_block())) ||
        (inst->merge_block() == inst->false_block() && block_is_terminal_exit(inst->true_block()));
    spv::Block *synthetic_merge = nullptr;
    spv::Block *selection_merge_target = merge_block;
    spv::Block *true_branch_target = merge_block;
    spv::Block *false_branch_target = merge_block;
    if (merge_already_used || (!true_post_dom && !false_post_dom && !non_merge_arm_terminal)) {
        // Both arms exit the construct (or the merge is already reserved), so a
        // synthetic merge is required for SPIR-V structural correctness.
        synthetic_merge = new spv::Block(_builder.getUniqueId(), function);
        selection_merge_target = synthetic_merge;
        true_branch_target = synthetic_merge;
        false_branch_target = synthetic_merge;
    } else if (!true_post_dom && false_post_dom) {
        if (block_is_terminal_exit(inst->true_block())) {
            true_branch_target = nullptr;
        } else if (arm_bypasses_merge(inst->true_block())) {
            // True arm bypasses the merge: keep the merge as the selection
            // merge and chain the bypassing region through it (see above).
            bypass_with_merge_redirect = true;
            bypass_true_arm = true;
            true_branch_target = merge_block;
        } else {
            // True arm falls through: the true target is also the SPIR-V merge.
            selection_merge_target = true_block;
            true_branch_target = nullptr;// fall through
        }
        false_branch_target = merge_block;
    } else if (true_post_dom && !false_post_dom) {
        if (block_is_terminal_exit(inst->false_block())) {
            false_branch_target = nullptr;
        } else if (arm_bypasses_merge(inst->false_block())) {
            // False arm bypasses the merge: keep the merge as the selection
            // merge and chain the bypassing region through it (see above).
            bypass_with_merge_redirect = true;
            false_branch_target = merge_block;
        } else {
            // False arm falls through: the false target is also the SPIR-V merge.
            selection_merge_target = false_block;
            false_branch_target = nullptr;// fall through
        }
        true_branch_target = merge_block;
    }
    spv::Block *merge_forward_target = nullptr;
    auto find_merge_forward_target = [&]() noexcept -> spv::Block * {
        return _direct_loop_boundary_forward_target(inst->merge_block());
    };
    if (synthetic_merge != nullptr || selection_merge_target != merge_block) {
        merge_forward_target = find_merge_forward_target();
    }
    if (synthetic_merge == nullptr && selection_merge_target != merge_block) {
        if (merge_forward_target != nullptr) {
            synthetic_merge = new spv::Block(_builder.getUniqueId(), function);
            selection_merge_target = synthetic_merge;
            true_branch_target = synthetic_merge;
            false_branch_target = synthetic_merge;
            if (!_is_direct_structured_branch_target(inst->merge_block())) {
                _forwarded_blocks.emplace(inst->merge_block());
            }
        }
    }
    _used_merge_blocks.emplace(selection_merge_target->getId());
    if (synthetic_merge != nullptr) {
        // Mark the real merge as used so no later construct picks it as its
        // merge block while it is still waiting to be emitted.
        _used_merge_blocks.emplace(merge_block->getId());
    }
    // If an arm coincides with an outer construct's merge block, emit its
    // instructions into a fresh SPIR-V block instead.  This keeps the outer
    // merge block free to be emitted after the inner construct, avoiding the
    // dominance-ordering conflict described above.
    auto make_arm_target = [&](const xir::BasicBlock *xb, spv::Block *arm_block) -> spv::Block * {
        if (xb == inst->merge_block()) { return selection_merge_target; }
        bool is_outer_merge = std::find(_outer_merge_stack.begin(), _outer_merge_stack.end(), arm_block) !=
                              _outer_merge_stack.end();
        if (!is_outer_merge) { return arm_block; }
        auto *synthetic = new spv::Block(_builder.getUniqueId(), function);
        return synthetic;
    };
    auto true_target = make_arm_target(inst->true_block(), true_block);
    auto false_target = make_arm_target(inst->false_block(), false_block);

    // Track this construct's merge so nested constructs can tell whether one of
    // their arms is really an outer merge block.  Using an outer merge as an arm
    // would place the outer merge after the inner construct in the binary while
    // the inner construct's merge is dominated by that arm, causing a dominance
    // ordering violation.  We avoid that by cloning the arm into a fresh block.
    _outer_merge_stack.push_back(selection_merge_target);

    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(selection_merge_target->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
    _builder.createConditionalBranch(cond, true_target, false_target);
    auto emit_arm = [&](const xir::BasicBlock *xb, spv::Block *target, spv::Block *branch_to) noexcept {
        if (xb == inst->merge_block()) { return; }
        if (!_added_blocks.contains(target)) {
            function.addBlock(target);
            _added_blocks.emplace(target);
        }
        auto old_merge_redirect = _branch_target_redirect.find(inst->merge_block());
        auto old_merge_redirect_block = old_merge_redirect == _branch_target_redirect.end() ? nullptr : old_merge_redirect->second;
        if (merge_forward_target != nullptr || merge_already_used) {
            _branch_target_redirect[inst->merge_block()] = selection_merge_target;
        }
        auto pending_base = _pending_blocks.size();
        _builder.setBuildPoint(target);
        auto emitted = _emit_block(xb, target);
        while (_pending_blocks.size() > pending_base) {
            luisa::vector<const xir::BasicBlock *> sub_blocks;
            while (_pending_blocks.size() > pending_base) {
                auto *pending = _pending_blocks.back();
                _pending_blocks.pop_back();
                if (pending == inst->merge_block() || _forwarded_blocks.contains(pending)) { continue; }
                sub_blocks.emplace_back(pending);
            }
            std::reverse(sub_blocks.begin(), sub_blocks.end());
            for (auto *pending : sub_blocks) {
                _emit_block(pending);
            }
        }
        if (old_merge_redirect == _branch_target_redirect.end()) {
            _branch_target_redirect.erase(inst->merge_block());
        } else {
            _branch_target_redirect[inst->merge_block()] = old_merge_redirect_block;
        }
        auto *tail = _builder.getBuildPoint();
        if (emitted && branch_to != nullptr && tail != nullptr && !tail->isTerminated()) {
            _builder.createBranch(false, branch_to);
        }
    };
    // When an arm bypasses the declared merge, redirect the merge's ultimate
    // forward target back to the merge block while that arm is emitted.
    // Nested merge blocks that would otherwise jump directly to the forward
    // target then branch to this merge instead, forming a properly nested
    // chain of merge blocks that converges to the forward target only through
    // this construct's merge.  The redirect is removed before the real merge
    // block itself is emitted so that the merge still forwards to the
    // (possibly redirected-by-an-outer-construct) forward target.
    auto emit_arm_guarded = [&](const xir::BasicBlock *xb, spv::Block *target,
                                spv::Block *branch_to, bool is_bypass_arm) noexcept {
        spv::Block *old_redirect = nullptr;
        bool had_old_redirect = false;
        bool redirect_installed = false;
        if (bypass_with_merge_redirect && is_bypass_arm) {
            if (auto it = _branch_target_redirect.find(merge_ultimate_forward);
                it != _branch_target_redirect.end()) {
                had_old_redirect = true;
                old_redirect = it->second;
            }
            _branch_target_redirect[merge_ultimate_forward] = merge_block;
            redirect_installed = true;
        }
        emit_arm(xb, target, branch_to);
        if (redirect_installed) {
            if (had_old_redirect) {
                _branch_target_redirect[merge_ultimate_forward] = old_redirect;
            } else {
                _branch_target_redirect.erase(merge_ultimate_forward);
            }
        }
    };
    emit_arm_guarded(inst->true_block(), true_target, true_branch_target, bypass_true_arm);
    emit_arm_guarded(inst->false_block(), false_target, false_branch_target, !bypass_true_arm);
    if (synthetic_merge != nullptr) {
        function.addBlock(synthetic_merge);
        _added_blocks.emplace(synthetic_merge);
        _builder.setBuildPoint(synthetic_merge);
        if (merge_forward_target != nullptr) {
            _builder.createBranch(false, merge_forward_target);
        } else if (merge_already_used) {
            // The real merge was already emitted as another construct's merge;
            // forward to it so this selection still converges there.
            _builder.createBranch(false, merge_block);
        } else {
            // The merge is unreachable from both arms; terminate the synthetic
            // block so it does not pull the real merge forward in the binary.
            _builder.createNoResultOp(spv::Op::OpUnreachable);
        }
    }
    // Emit the real merge block now only if it is the actual continuation of
    // this selection.  If it is unreachable from the arms but has other
    // predecessors, it will be emitted when those predecessors are processed.
    // A block in _used_merge_blocks may only be reserved by an enclosing
    // construct, such as a loop update block. Do not emit it here unless it has
    // actually been emitted already; otherwise the merge/update block can read
    // SSA values produced by pending loop-body blocks that are not mapped yet.
    bool real_merge_already_emitted = _emitted_blocks.contains(inst->merge_block());
    bool merge_is_arm = inst->merge_block() == inst->true_block() ||
                        inst->merge_block() == inst->false_block();
    bool emit_real_merge = (merge_forward_target == nullptr || merge_is_arm) &&
                           (synthetic_merge == nullptr || real_merge_already_emitted ||
                            !block_has_predecessors(inst->merge_block()));
    if (emit_real_merge) {
        if (!_added_blocks.contains(merge_block)) {
            function.addBlock(merge_block);
            _added_blocks.emplace(merge_block);
        }
        _builder.setBuildPoint(merge_block);
        _emit_block(inst->merge_block(), merge_block);
        if (!_builder.getBuildPoint()->isTerminated()) {
            _builder.createNoResultOp(spv::Op::OpUnreachable);
        }
    }
    _outer_merge_stack.pop_back();
}

void SpirvCodegenEntry::_emit_loop_inst(const xir::LoopInst *inst) noexcept {
    auto pending_base = _pending_blocks.size();
    auto header = &_builder.makeNewBlock();
    _added_blocks.emplace(header);
    auto prepare = _get_or_create_block(inst->prepare_block());
    auto body = _get_or_create_block(inst->body_block());
    auto update = _get_or_create_block(inst->update_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _used_merge_blocks.emplace(merge->getId());
    _used_merge_blocks.emplace(update->getId());
    _loop_boundary_stack.emplace_back(inst->merge_block(), merge);
    _loop_boundary_stack.emplace_back(inst->update_block(), update);
    _loop_header_redirect.emplace(inst->prepare_block(), header);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, update, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, prepare);
    _emit_block(inst->prepare_block());
    _emit_block(inst->body_block());
    // Collect pending sub-blocks and emit them in FIFO (insertion) order
    // so definitions that dominate uses are emitted first.  Nested loops may
    // enqueue new blocks (e.g., the continuation after a nested loop) while we
    // are emitting previously enqueued blocks, so drain the pending list in a
    // fixed-point loop before emitting the update and merge blocks.
    while (true) {
        luisa::vector<const xir::BasicBlock *> sub_blocks;
        while (_pending_blocks.size() > pending_base) {
            auto *bb = _pending_blocks.back();
            _pending_blocks.pop_back();
            if (bb == inst->update_block() || bb == inst->merge_block()) {
                continue;
            }
            sub_blocks.push_back(bb);
        }
        if (sub_blocks.empty()) { break; }
        std::reverse(sub_blocks.begin(), sub_blocks.end());
        for (auto *bb : sub_blocks) {
            _emit_block(bb);
        }
    }
    _emit_block(inst->update_block());
    _emit_block(inst->merge_block());
    _loop_boundary_stack.pop_back();
    _loop_boundary_stack.pop_back();
}

void SpirvCodegenEntry::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept {
    auto pending_base = _pending_blocks.size();
    auto header = &_builder.makeNewBlock();
    _added_blocks.emplace(header);
    auto continue_block = &_builder.makeNewBlock();
    _added_blocks.emplace(continue_block);
    auto body = _get_or_create_block(inst->body_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _used_merge_blocks.emplace(merge->getId());
    _used_merge_blocks.emplace(continue_block->getId());
    _loop_boundary_stack.emplace_back(inst->merge_block(), merge);
    _loop_boundary_stack.emplace_back(inst->body_block(), continue_block);
    _loop_header_redirect.emplace(inst->body_block(), continue_block);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, continue_block, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, body);
    _emit_block(inst->body_block());
    // Collect pending sub-blocks and emit them in FIFO (insertion) order.
    // Drain in a fixed-point loop because nested loops may enqueue additional
    // blocks while we are emitting previously enqueued ones.
    while (true) {
        luisa::vector<const xir::BasicBlock *> sub_blocks;
        while (_pending_blocks.size() > pending_base) {
            auto *bb = _pending_blocks.back();
            _pending_blocks.pop_back();
            if (bb == inst->merge_block()) {
                continue;
            }
            sub_blocks.push_back(bb);
        }
        if (sub_blocks.empty()) { break; }
        std::reverse(sub_blocks.begin(), sub_blocks.end());
        for (auto *bb : sub_blocks) {
            _emit_block(bb);
        }
    }
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, continue_block);
    }
    _builder.setBuildPoint(continue_block);
    _builder.createBranch(false, header);
    _emit_block(inst->merge_block());
    _loop_boundary_stack.pop_back();
    _loop_boundary_stack.pop_back();
}

void SpirvCodegenEntry::_emit_switch_inst(const xir::SwitchInst *inst) noexcept {
    auto selector = _emit_value(inst->value());
    auto case_count = inst->case_count();
    auto &function = _builder.getBuildPoint()->getParent();
    auto pending_base = _pending_blocks.size();
    auto bind_or_get = [&](const xir::BasicBlock *xb, bool &is_fresh) -> spv::Block * {
        if (auto it = _block_map.find(xb); it != _block_map.end()) {
            is_fresh = false;
            return it->second;
        }
        is_fresh = true;
        auto blk = new spv::Block(_builder.getUniqueId(), function);
        _block_map.emplace(xb, blk);
        return blk;
    };
    std::vector<spv::Block *> segment_blocks;
    std::vector<bool> segment_fresh;
    segment_blocks.reserve(case_count + 1);
    segment_fresh.reserve(case_count + 1);
    for (uint i = 0u; i < case_count; ++i) {
        bool fresh = false;
        segment_blocks.push_back(bind_or_get(inst->case_block(i), fresh));
        segment_fresh.push_back(fresh);
    }
    {
        bool fresh = false;
        segment_blocks.push_back(bind_or_get(inst->default_block(), fresh));
        segment_fresh.push_back(fresh);
    }
    bool merge_fresh = false;
    auto merge_block = bind_or_get(inst->merge_block(), merge_fresh);

    // Detect if any case body branches to another case body or back to
    // the switch header. In structured SPIR-V, both require a loop wrapper.
    auto *switch_xir_bb = inst->parent_block();
    bool needs_loop_wrapper = false;
    {
        luisa::unordered_set<const xir::BasicBlock *> segment_entries;
        for (uint i = 0u; i < case_count; ++i) {
            segment_entries.emplace(inst->case_block(i));
        }
        segment_entries.emplace(inst->default_block());
        luisa::unordered_set<const xir::BasicBlock *> structural_exits;
        auto *def = switch_xir_bb == nullptr ? nullptr : switch_xir_bb->parent_function();
        if (def != nullptr) {
            for (auto *bb : def->basic_blocks()) {
                if (!bb->is_terminated()) { continue; }
                auto *term = bb->terminator();
                if (auto *cfm = term->control_flow_merge(); cfm != nullptr) {
                    if (auto *merge = cfm->merge_block(); merge != nullptr) {
                        structural_exits.emplace(merge);
                    }
                }
                if (term->isa<xir::LoopInst>()) {
                    auto *loop = static_cast<const xir::LoopInst *>(term);
                    if (auto *prepare = loop->prepare_block(); prepare != nullptr) { structural_exits.emplace(prepare); }
                    if (auto *update = loop->update_block(); update != nullptr) { structural_exits.emplace(update); }
                } else if (term->isa<xir::SimpleLoopInst>()) {
                    auto *loop = static_cast<const xir::SimpleLoopInst *>(term);
                    if (auto *body = loop->body_block(); body != nullptr) { structural_exits.emplace(body); }
                }
            }
        }
        for (uint i = 0u; i <= case_count && !needs_loop_wrapper; ++i) {
            auto *start = i < case_count ? inst->case_block(i) : inst->default_block();
            luisa::unordered_set<const xir::BasicBlock *> visited;
            luisa::vector<const xir::BasicBlock *> work;
            work.push_back(start);
            while (!work.empty()) {
                auto *bb = work.back();
                work.pop_back();
                if (bb == nullptr || !visited.emplace(bb).second) { continue; }
                if (bb == switch_xir_bb) {
                    needs_loop_wrapper = true;
                    break;
                }
                // Cross-segment branch: reached a different segment entry
                if (bb != start && segment_entries.contains(bb)) {
                    needs_loop_wrapper = true;
                    break;
                }
                if (bb != start && structural_exits.contains(bb)) { continue; }
                if (bb->is_terminated()) {
                    bb->traverse_successors(false, [&](const xir::BasicBlock *succ) noexcept {
                        if (succ != switch_xir_bb && !segment_entries.contains(succ) &&
                            succ != start && structural_exits.contains(succ)) { return; }
                        work.push_back(succ);
                    });
                    if (auto *cfm = bb->terminator()->control_flow_merge()) {
                        if (auto *merge = cfm->merge_block()) {
                            work.push_back(merge);
                        }
                    }
                }
            }
        }
    }

    spv::Block *loop_header = nullptr;
    spv::Block *loop_continue = nullptr;
    spv::Block *loop_merge = nullptr;
    if (needs_loop_wrapper) {
        auto *switch_spv_block = new spv::Block(_builder.getUniqueId(), function);
        loop_header = &_builder.makeNewBlock();
        loop_continue = &_builder.makeNewBlock();
        loop_merge = &_builder.makeNewBlock();
        _added_blocks.emplace(loop_header);
        _added_blocks.emplace(loop_continue);
        _added_blocks.emplace(loop_merge);

        _builder.createBranch(false, loop_header);
        _block_map[switch_xir_bb] = loop_header;
        _loop_header_redirect.emplace(switch_xir_bb, loop_continue);

        _builder.setBuildPoint(loop_header);
        _builder.createLoopMerge(loop_merge, loop_continue, spv::LoopControlMask::MaskNone, {});
        _builder.createBranch(false, switch_spv_block);

        function.addBlock(switch_spv_block);
        _builder.setBuildPoint(switch_spv_block);
    }

    spv::Block *synthetic_merge = nullptr;
    spv::Block *selection_merge_target = merge_block;
    if (_used_merge_blocks.contains(merge_block->getId())) {
        synthetic_merge = new spv::Block(_builder.getUniqueId(), function);
        selection_merge_target = synthetic_merge;
    }
    _used_merge_blocks.emplace(selection_merge_target->getId());
    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(selection_merge_target->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
    auto switch_inst = new spv::Instruction(spv::Op::OpSwitch);
    switch_inst->reserveOperands((case_count * 2) + 2);
    switch_inst->addIdOperand(selector);
    auto default_target = (segment_blocks[case_count] == merge_block && synthetic_merge != nullptr) ? selection_merge_target : segment_blocks[case_count];
    switch_inst->addIdOperand(default_target->getId());
    default_target->addPredecessor(_builder.getBuildPoint());
    for (uint i = 0u; i < case_count; ++i) {
        auto case_target = (segment_blocks[i] == merge_block && synthetic_merge != nullptr) ? selection_merge_target : segment_blocks[i];
        switch_inst->addImmediateOperand(inst->case_value(i));
        switch_inst->addIdOperand(case_target->getId());
        case_target->addPredecessor(_builder.getBuildPoint());
    }
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(switch_inst));
    for (uint i = 0u; i < case_count; ++i) {
        if (inst->case_block(i) == inst->merge_block()) { continue; }
        if (segment_fresh[i]) {
            function.addBlock(segment_blocks[i]);
            _added_blocks.emplace(segment_blocks[i]);
        }
        _builder.setBuildPoint(segment_blocks[i]);
        auto emitted = _emit_block(inst->case_block(i));
        if (emitted && !_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, selection_merge_target);
        }
    }
    if (inst->default_block() != inst->merge_block()) {
        if (segment_fresh[case_count]) {
            function.addBlock(segment_blocks[case_count]);
            _added_blocks.emplace(segment_blocks[case_count]);
        }
        _builder.setBuildPoint(segment_blocks[case_count]);
        auto emitted = _emit_block(inst->default_block());
        if (emitted && !_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, selection_merge_target);
        }
    }
    // Collect pending sub-blocks and emit them in FIFO (insertion) order.
    luisa::vector<const xir::BasicBlock *> sub_blocks;
    while (_pending_blocks.size() > pending_base) {
        auto *bb = _pending_blocks.back();
        _pending_blocks.pop_back();
        if (bb == inst->merge_block()) { continue; }
        sub_blocks.push_back(bb);
    }
    std::reverse(sub_blocks.begin(), sub_blocks.end());
    for (auto *bb : sub_blocks) {
        _emit_block(bb);
    }
    if (synthetic_merge != nullptr) {
        function.addBlock(synthetic_merge);
        _added_blocks.emplace(synthetic_merge);
        _builder.setBuildPoint(synthetic_merge);
        _builder.createBranch(false, merge_block);
    }
    if (merge_fresh) {
        function.addBlock(merge_block);
        _added_blocks.emplace(merge_block);
    }

    if (needs_loop_wrapper) {
        _builder.setBuildPoint(loop_continue);
        _builder.createBranch(false, loop_header);
        _builder.setBuildPoint(loop_merge);
        _builder.createNoResultOp(spv::Op::OpUnreachable);
    }

    _builder.setBuildPoint(merge_block);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_branch_inst(const xir::BranchInst *inst) noexcept {
    auto target = inst->target_block();
    auto target_redirect = _branch_target_redirect.find(target);
    auto target_is_redirected = target_redirect != _branch_target_redirect.end() &&
                                target_redirect->second != nullptr;
    auto spv_target = _resolve_branch_target(target);
    _builder.createBranch(false, spv_target);
    if (!target_is_redirected && !_emitted_blocks.contains(target) && !_forwarded_blocks.contains(target)) {
        _pending_blocks.push_back(target);
    }
}

void SpirvCodegenEntry::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto get_target = [&](const xir::BasicBlock *target) -> spv::Block * {
        return _resolve_branch_target(target);
    };
    auto true_block = get_target(inst->true_block());
    auto false_block = get_target(inst->false_block());

    // If this conditional branch is not already part of a loop header or an
    // IfInst selection, but it is inside a loop, it is likely a loop-boundary
    // branch (break/continue) produced by restructure_cfg. Wrap it in a
    // selection construct using the target that continues the loop as the merge
    // block. We identify the continue target by reachability to the loop's
    // continue/update block.
    auto *current = _builder.getBuildPoint();
    bool has_loop_merge = false;
    bool has_selection_merge = false;
    if (current != nullptr) {
        for (auto &instr : current->getInstructions()) {
            auto op = instr->getOpCode();
            if (op == spv::Op::OpLoopMerge) { has_loop_merge = true; }
            if (op == spv::Op::OpSelectionMerge) { has_selection_merge = true; }
        }
    }
    bool selection_merge_added = false;
    auto add_selection_merge = [&](spv::Block *merge_target) noexcept {
        auto selection_merge_inst = new spv::Instruction(spv::Op::OpSelectionMerge);
        selection_merge_inst->reserveOperands(2);
        selection_merge_inst->addIdOperand(merge_target->getId());
        selection_merge_inst->addImmediateOperand(spv::SelectionControlMask::MaskNone);
        _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge_inst));
        selection_merge_added = true;
    };
    if (!has_loop_merge && !has_selection_merge && !_loop_boundary_stack.empty()) {
        // Stack layout per loop: (merge, continue/update). The innermost loop
        // boundaries are at the top two entries.
        auto *loop_merge_xb = _loop_boundary_stack[_loop_boundary_stack.size() - 2].first;
        auto *loop_continue_xb = _loop_boundary_stack.back().first;
        auto true_is_loop_boundary = inst->true_block() == loop_merge_xb ||
                                     inst->true_block() == loop_continue_xb;
        auto false_is_loop_boundary = inst->false_block() == loop_merge_xb ||
                                      inst->false_block() == loop_continue_xb;
        auto true_reaches_continue = !true_is_loop_boundary &&
                                     block_reaches_excluding(inst->true_block(), loop_continue_xb, loop_merge_xb);
        auto false_reaches_continue = !false_is_loop_boundary &&
                                      block_reaches_excluding(inst->false_block(), loop_continue_xb, loop_merge_xb);
        if (true_reaches_continue != false_reaches_continue &&
            !true_is_loop_boundary && !false_is_loop_boundary) {
            auto selection_merge = true_reaches_continue ? true_block : false_block;
            add_selection_merge(selection_merge);
        }
    }

    // A divergent conditional branch that is not covered by an enclosing
    // structured loop or selection (e.g. one produced for irreducible CFGs
    // such as coroutine state machines) still forms a selection in SPIR-V:
    // the validator requires it to be preceded by an OpSelectionMerge whenever
    // both targets are fresh (not yet a merge/branch target).  The merge point
    // is the block where the two arms reconverge: when one arm's region flows
    // into the other target's block, that block is the merge; otherwise fall
    // back to the immediate post-dominator of the current block.  When one arm
    // is a terminal exit, the other arm is the continuation and serves as the
    // merge.  The merge is only emitted when it is not already another
    // construct's merge block — in that case the branch is already structured
    // through that construct and needs no merge of its own.
    if (!has_loop_merge && !has_selection_merge && !selection_merge_added &&
        inst->true_block() != inst->false_block() &&
        !_is_direct_structured_branch_target(inst->true_block()) &&
        !_is_direct_structured_branch_target(inst->false_block())) {
        spv::Block *selection_merge = nullptr;
        if (block_reaches(inst->false_block(), inst->true_block())) {
            // The false arm's region flows back to the true target block.
            selection_merge = true_block;
        } else if (block_reaches(inst->true_block(), inst->false_block())) {
            // The true arm's region flows back to the false target block.
            selection_merge = false_block;
        } else if (block_is_terminal_exit(inst->true_block())) {
            selection_merge = false_block;
        } else if (block_is_terminal_exit(inst->false_block())) {
            selection_merge = true_block;
        } else if (_dom_tree != nullptr) {
            if (auto *ipdom = _dom_tree->immediate_post_dominator(
                    const_cast<xir::BasicBlock *>(inst->parent_block()));
                ipdom != nullptr && ipdom != inst->parent_block()) {
                selection_merge = _resolve_branch_target(ipdom);
            }
        }
        // Do not reuse a block that is already the merge of an enclosing
        // construct: that would make it the merge of two headers (invalid),
        // and the branch is already structured through that construct anyway.
        bool merge_is_enclosing = std::find(_outer_merge_stack.begin(),
                                            _outer_merge_stack.end(),
                                            selection_merge) != _outer_merge_stack.end();
        if (selection_merge != nullptr && !merge_is_enclosing) {
            add_selection_merge(selection_merge);
        }
    }

    _builder.createConditionalBranch(cond, true_block, false_block);
    auto should_enqueue = [&](const xir::BasicBlock *target) noexcept {
        auto target_redirect = _branch_target_redirect.find(target);
        auto target_is_redirected = target_redirect != _branch_target_redirect.end() &&
                                    target_redirect->second != nullptr;
        return !target_is_redirected && !_emitted_blocks.contains(target) && !_forwarded_blocks.contains(target);
    };
    if (should_enqueue(inst->true_block())) {
        _pending_blocks.push_back(inst->true_block());
    }
    if (should_enqueue(inst->false_block())) {
        _pending_blocks.push_back(inst->false_block());
    }
}

}// namespace lc::spirv
