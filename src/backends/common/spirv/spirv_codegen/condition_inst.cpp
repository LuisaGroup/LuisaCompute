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
    if (true_fresh) { function.addBlock(true_block); }
    _builder.setBuildPoint(true_block);
    _emit_block(inst->true_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, selection_merge_target);
    }
    if (false_fresh) { function.addBlock(false_block); }
    _builder.setBuildPoint(false_block);
    _emit_block(inst->false_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, selection_merge_target);
    }
    if (synthetic_merge != nullptr) {
        function.addBlock(synthetic_merge);
        _builder.setBuildPoint(synthetic_merge);
        _builder.createBranch(false, merge_block);
    }
    if (merge_fresh) { function.addBlock(merge_block); }
    _builder.setBuildPoint(merge_block);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_loop_inst(const xir::LoopInst *inst) noexcept {
    auto prepare = _get_or_create_block(inst->prepare_block());
    auto body = _get_or_create_block(inst->body_block());
    auto update = _get_or_create_block(inst->update_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _used_merge_blocks.emplace(merge->getId());
    auto header = &_builder.makeNewBlock();
    _loop_header_redirect.emplace(inst->prepare_block(), header);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, update, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, prepare);
    _emit_block(inst->prepare_block());
    _emit_block(inst->body_block());
    _emit_block(inst->update_block());
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept {
    auto body = _get_or_create_block(inst->body_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _used_merge_blocks.emplace(merge->getId());
    auto header = &_builder.makeNewBlock();
    auto continue_block = &_builder.makeNewBlock();
    // Redirect branches to body_block to continue_block so that continues/breaks
    // inside the body don't create multiple back-edges to the header.
    _loop_header_redirect.emplace(inst->body_block(), continue_block);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, continue_block, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, body);
    _emit_block(inst->body_block());
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
    switch_inst->addIdOperand(segment_blocks[case_count]->getId());
    segment_blocks[case_count]->addPredecessor(_builder.getBuildPoint());
    for (uint i = 0u; i < case_count; ++i) {
        switch_inst->addImmediateOperand(inst->case_value(i));
        switch_inst->addIdOperand(segment_blocks[i]->getId());
        segment_blocks[i]->addPredecessor(_builder.getBuildPoint());
    }
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(switch_inst));
    for (uint i = 0u; i < case_count; ++i) {
        if (segment_fresh[i]) { function.addBlock(segment_blocks[i]); }
        _builder.setBuildPoint(segment_blocks[i]);
        _emit_block(inst->case_block(i));
        if (!_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, selection_merge_target);
        }
    }
    if (segment_fresh[case_count]) { function.addBlock(segment_blocks[case_count]); }
    _builder.setBuildPoint(segment_blocks[case_count]);
    _emit_block(inst->default_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, selection_merge_target);
    }
    if (synthetic_merge != nullptr) {
        function.addBlock(synthetic_merge);
        _builder.setBuildPoint(synthetic_merge);
        _builder.createBranch(false, merge_block);
    }
    if (merge_fresh) { function.addBlock(merge_block); }
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
    _emit_block(target);
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
    _emit_block(inst->true_block());
    _emit_block(inst->false_block());
}

}// namespace lc::spirv