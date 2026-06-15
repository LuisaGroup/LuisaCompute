#include "entry.h"

namespace lc::spirv {
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
    _builder.createConditionalBranch(cond, true_block, false_block);
    if (true_fresh) { function.addBlock(true_block); _added_blocks.emplace(true_block); }
    _builder.setBuildPoint(true_block);
    _emit_block(inst->true_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, selection_merge_target);
    }
    if (false_fresh) { function.addBlock(false_block); _added_blocks.emplace(false_block); }
    _builder.setBuildPoint(false_block);
    _emit_block(inst->false_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, selection_merge_target);
    }
    if (synthetic_merge != nullptr) {
        function.addBlock(synthetic_merge);
        _added_blocks.emplace(synthetic_merge);
        _builder.setBuildPoint(synthetic_merge);
        _builder.createBranch(false, merge_block);
    }
    if (merge_fresh) { function.addBlock(merge_block); _added_blocks.emplace(merge_block); }
    _builder.setBuildPoint(merge_block);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_loop_inst(const xir::LoopInst *inst) noexcept {
    auto header = &_builder.makeNewBlock();
    _added_blocks.emplace(header);
    auto prepare = _get_or_create_block(inst->prepare_block());
    auto body = _get_or_create_block(inst->body_block());
    auto update = _get_or_create_block(inst->update_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _used_merge_blocks.emplace(merge->getId());
    _used_merge_blocks.emplace(update->getId());
    _loop_header_redirect.emplace(inst->prepare_block(), header);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, update, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, prepare);
    _emit_block(inst->prepare_block());
    _emit_block(inst->body_block());
    while (!_pending_blocks.empty()) {
        auto *bb = _pending_blocks.back();
        _pending_blocks.pop_back();
        if (bb == inst->update_block() || bb == inst->merge_block()) {
            continue;
        }
        _emit_block(bb);
    }
    _emit_block(inst->update_block());
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept {
    auto header = &_builder.makeNewBlock();
    _added_blocks.emplace(header);
    auto continue_block = &_builder.makeNewBlock();
    _added_blocks.emplace(continue_block);
    auto body = _get_or_create_block(inst->body_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _used_merge_blocks.emplace(merge->getId());
    _used_merge_blocks.emplace(continue_block->getId());
    _loop_header_redirect.emplace(inst->body_block(), continue_block);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, continue_block, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, body);
    _emit_block(inst->body_block());
    while (!_pending_blocks.empty()) {
        auto *bb = _pending_blocks.back();
        _pending_blocks.pop_back();
        if (bb == inst->merge_block()) {
            continue;
        }
        _emit_block(bb);
    }
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, continue_block);
    }
    _builder.setBuildPoint(continue_block);
    _builder.createBranch(false, header);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_switch_inst(const xir::SwitchInst *inst) noexcept {
    auto selector = _emit_value(inst->value());
    auto case_count = inst->case_count();
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
        if (segment_fresh[i]) { function.addBlock(segment_blocks[i]); _added_blocks.emplace(segment_blocks[i]); }
        _builder.setBuildPoint(segment_blocks[i]);
        _emit_block(inst->case_block(i));
        if (!_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, selection_merge_target);
        }
    }
    if (segment_fresh[case_count]) { function.addBlock(segment_blocks[case_count]); _added_blocks.emplace(segment_blocks[case_count]); }
    _builder.setBuildPoint(segment_blocks[case_count]);
    _emit_block(inst->default_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, selection_merge_target);
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
        function.addBlock(loop_merge);
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
    _builder.createConditionalBranch(cond, true_block, false_block);
    if (!_emitted_blocks.contains(inst->true_block())) {
        _pending_blocks.push_back(inst->true_block());
    }
    if (!_emitted_blocks.contains(inst->false_block())) {
        _pending_blocks.push_back(inst->false_block());
    }
}

}// namespace lc::spirv
