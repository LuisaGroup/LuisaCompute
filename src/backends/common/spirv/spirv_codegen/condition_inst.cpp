#include "entry.h"

namespace lc::spirv {
void SpirvCodegenEntry::_emit_if_inst(const xir::IfInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto &function = _builder.getBuildPoint()->getParent();
    auto true_block = new spv::Block(_builder.getUniqueId(), function);
    auto false_block = new spv::Block(_builder.getUniqueId(), function);
    auto merge_block = new spv::Block(_builder.getUniqueId(), function);
    _block_map[inst->true_block()] = true_block;
    _block_map[inst->false_block()] = false_block;
    _block_map[inst->merge_block()] = merge_block;
    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(merge_block->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::unique_ptr<spv::Instruction>(selection_merge));
    _builder.createConditionalBranch(cond, true_block, false_block);
    function.addBlock(true_block);
    _builder.setBuildPoint(true_block);
    _emit_block(inst->true_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, merge_block);
    }
    function.addBlock(false_block);
    _builder.setBuildPoint(false_block);
    _emit_block(inst->false_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, merge_block);
    }
    function.addBlock(merge_block);
    _builder.setBuildPoint(merge_block);
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_loop_inst(const xir::LoopInst *inst) noexcept {
    auto prepare = _get_or_create_block(inst->prepare_block());
    auto body = _get_or_create_block(inst->body_block());
    auto update = _get_or_create_block(inst->update_block());
    auto merge = _get_or_create_block(inst->merge_block());
    // Create a dedicated loop header block to avoid OpLoopMerge in the same block as other control flow.
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
    // Create a dedicated loop header block and a separate continue block.
    // The body block cannot be the continue target if it contains breaks.
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
    std::vector<spv::Block *> segment_blocks;
    segment_blocks.reserve(case_count + 1);
    for (uint i = 0u; i <= case_count; ++i) {
        segment_blocks.push_back(new spv::Block(_builder.getUniqueId(), function));
    }
    auto merge_block = new spv::Block(_builder.getUniqueId(), function);
    for (uint i = 0u; i < case_count; ++i) {
        _block_map[inst->case_block(i)] = segment_blocks[i];
    }
    _block_map[inst->default_block()] = segment_blocks[case_count];
    _block_map[inst->merge_block()] = merge_block;
    auto selection_merge = new spv::Instruction(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2);
    selection_merge->addIdOperand(merge_block->getId());
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
        function.addBlock(segment_blocks[i]);
        _builder.setBuildPoint(segment_blocks[i]);
        _emit_block(inst->case_block(i));
        if (!_builder.getBuildPoint()->isTerminated()) {
            _builder.createBranch(false, merge_block);
        }
    }
    function.addBlock(segment_blocks[case_count]);
    _builder.setBuildPoint(segment_blocks[case_count]);
    _emit_block(inst->default_block());
    if (!_builder.getBuildPoint()->isTerminated()) {
        _builder.createBranch(false, merge_block);
    }
    function.addBlock(merge_block);
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