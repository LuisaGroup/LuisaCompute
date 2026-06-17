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

}// namespace

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
    // If the merge block is not reachable from either arm (both arms exit the
    // construct via break/continue/return), using it as the SPIR-V selection
    // merge would force glslang's inReadableOrder() to emit the dead merge
    // before any later block that dominates it.  Use a synthetic dead-end merge
    // instead, and let the real merge block be emitted when it is actually
    // reached via control flow.
    bool merge_already_used = _used_merge_blocks.contains(merge_block->getId());
    bool merge_reachable_from_arms = block_reaches(inst->true_block(), inst->merge_block()) ||
                                     block_reaches(inst->false_block(), inst->merge_block());
    bool needs_synthetic_merge = merge_already_used || !merge_reachable_from_arms;

    spv::Block *synthetic_merge = nullptr;
    spv::Block *selection_merge_target = merge_block;
    if (needs_synthetic_merge) {
        synthetic_merge = new spv::Block(_builder.getUniqueId(), function);
        selection_merge_target = synthetic_merge;
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
    _outer_merge_stack.push_back(merge_block);

    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(selection_merge_target->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
    _builder.createConditionalBranch(cond, true_target, false_target);
    auto emit_arm = [&](const xir::BasicBlock *xb, spv::Block *target) noexcept {
        if (xb == inst->merge_block()) { return; }
        if (!_added_blocks.contains(target)) {
            function.addBlock(target);
            _added_blocks.emplace(target);
        }
        _builder.setBuildPoint(target);
        _emit_block(xb, target);
        if (!_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, selection_merge_target);
        }
    };
    emit_arm(inst->true_block(), true_target);
    emit_arm(inst->false_block(), false_target);
    if (synthetic_merge != nullptr) {
        function.addBlock(synthetic_merge);
        _added_blocks.emplace(synthetic_merge);
        _builder.setBuildPoint(synthetic_merge);
        if (merge_already_used) {
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
    bool emit_real_merge = synthetic_merge == nullptr || real_merge_already_emitted ||
                           !block_has_predecessors(inst->merge_block());
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
    // so definitions that dominate uses are emitted first.
    luisa::vector<const xir::BasicBlock *> sub_blocks;
    while (_pending_blocks.size() > pending_base) {
        auto *bb = _pending_blocks.back();
        _pending_blocks.pop_back();
        if (bb == inst->update_block() || bb == inst->merge_block()) {
            continue;
        }
        sub_blocks.push_back(bb);
    }
    std::reverse(sub_blocks.begin(), sub_blocks.end());
    for (auto *bb : sub_blocks) {
        _emit_block(bb);
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
    luisa::vector<const xir::BasicBlock *> sub_blocks;
    while (_pending_blocks.size() > pending_base) {
        auto *bb = _pending_blocks.back();
        _pending_blocks.pop_back();
        if (bb == inst->merge_block()) {
            continue;
        }
        sub_blocks.push_back(bb);
    }
    std::reverse(sub_blocks.begin(), sub_blocks.end());
    for (auto *bb : sub_blocks) {
        _emit_block(bb);
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
                    needs_loop_wrapper = true; break;
                }
                // Cross-segment branch: reached a different segment entry
                if (bb != start && segment_entries.contains(bb)) {
                    needs_loop_wrapper = true; break;
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
    auto default_target = (segment_blocks[case_count] == merge_block && synthetic_merge != nullptr)
                              ? selection_merge_target : segment_blocks[case_count];
    switch_inst->addIdOperand(default_target->getId());
    default_target->addPredecessor(_builder.getBuildPoint());
    for (uint i = 0u; i < case_count; ++i) {
        auto case_target = (segment_blocks[i] == merge_block && synthetic_merge != nullptr)
                               ? selection_merge_target : segment_blocks[i];
        switch_inst->addImmediateOperand(inst->case_value(i));
        switch_inst->addIdOperand(case_target->getId());
        case_target->addPredecessor(_builder.getBuildPoint());
    }
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(switch_inst));
    for (uint i = 0u; i < case_count; ++i) {
        if (inst->case_block(i) == inst->merge_block()) { continue; }
        if (segment_fresh[i]) { function.addBlock(segment_blocks[i]); _added_blocks.emplace(segment_blocks[i]); }
        _builder.setBuildPoint(segment_blocks[i]);
        _emit_block(inst->case_block(i));
        if (!_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, selection_merge_target);
        }
    }
    if (inst->default_block() != inst->merge_block()) {
        if (segment_fresh[case_count]) { function.addBlock(segment_blocks[case_count]); _added_blocks.emplace(segment_blocks[case_count]); }
        _builder.setBuildPoint(segment_blocks[case_count]);
        _emit_block(inst->default_block());
        if (!_builder.getBuildPoint()->isTerminated()) {
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
    if (merge_fresh) { function.addBlock(merge_block); _added_blocks.emplace(merge_block); }

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
    spv::Block *spv_target = nullptr;
    if (auto it = _loop_header_redirect.find(target); it != _loop_header_redirect.end()) {
        spv_target = it->second;
    } else {
        spv_target = _get_or_create_block(target);
    }
    _builder.createBranch(false, spv_target);
    if (!_emitted_blocks.contains(target)) {
        _pending_blocks.push_back(target);
    }
}

void SpirvCodegenEntry::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto get_target = [&](const xir::BasicBlock *target) -> spv::Block * {
        if (auto it = _loop_header_redirect.find(target); it != _loop_header_redirect.end()) {
            return it->second;
        }
        return _get_or_create_block(target);
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
    if (!has_loop_merge && !has_selection_merge && !_loop_boundary_stack.empty()) {
        // Stack layout per loop: (merge, continue/update). The innermost loop
        // boundaries are at the top two entries.
        auto *loop_merge_xb = _loop_boundary_stack[_loop_boundary_stack.size() - 2].first;
        auto *loop_continue_xb = _loop_boundary_stack.back().first;
        auto true_reaches_continue = block_reaches_excluding(inst->true_block(), loop_continue_xb, loop_merge_xb);
        auto false_reaches_continue = block_reaches_excluding(inst->false_block(), loop_continue_xb, loop_merge_xb);
        spv::Block *selection_merge = nullptr;
        if (true_reaches_continue && !false_reaches_continue) {
            selection_merge = true_block;
        } else if (false_reaches_continue && !true_reaches_continue) {
            selection_merge = false_block;
        }
        if (selection_merge != nullptr) {
            auto selection_merge_inst = new spv::Instruction(spv::Op::OpSelectionMerge);
            selection_merge_inst->reserveOperands(2);
            selection_merge_inst->addIdOperand(selection_merge->getId());
            selection_merge_inst->addImmediateOperand(spv::SelectionControlMask::MaskNone);
            _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge_inst));
        }
    }

    _builder.createConditionalBranch(cond, true_block, false_block);
    if (!_emitted_blocks.contains(inst->true_block())) {
        _pending_blocks.push_back(inst->true_block());
    }
    if (!_emitted_blocks.contains(inst->false_block())) {
        _pending_blocks.push_back(inst->false_block());
    }
}

}// namespace lc::spirv
