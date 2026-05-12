#include "entry.h"
#include <luisa/core/logging.h>

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
    _loop_header_info.emplace(inst->prepare_block(), std::make_pair(merge, update));
    _builder.createBranch(false, prepare);
    _emit_block(inst->prepare_block());
    _emit_block(inst->body_block());
    _emit_block(inst->update_block());
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept {
    auto body = _get_or_create_block(inst->body_block());
    auto merge = _get_or_create_block(inst->merge_block());
    _loop_header_info.emplace(inst->body_block(), std::make_pair(merge, body));
    _builder.createBranch(false, body);
    _emit_block(inst->body_block());
    _emit_block(inst->merge_block());
}

void SpirvCodegenEntry::_emit_switch_inst(const xir::SwitchInst *inst) noexcept {
    LUISA_NOT_IMPLEMENTED("SPIR-V switch instruction.");
}

void SpirvCodegenEntry::_emit_branch_inst(const xir::BranchInst *inst) noexcept {
    _builder.createBranch(false, _get_or_create_block(inst->target_block()));
    _emit_block(inst->target_block());
}

void SpirvCodegenEntry::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    auto cond = _emit_value(inst->condition());
    auto true_block = _get_or_create_block(inst->true_block());
    auto false_block = _get_or_create_block(inst->false_block());
    _builder.createConditionalBranch(cond, true_block, false_block);
    _emit_block(inst->true_block());
    _emit_block(inst->false_block());
}

}// namespace lc::spirv