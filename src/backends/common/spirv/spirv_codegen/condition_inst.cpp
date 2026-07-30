#include "entry.h"
#include "instruction_layout.h"

#include <algorithm>
#include <luisa/core/logging.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>

namespace lc::spirv {

void SpirvCodegenEntry::_emit_if_inst(const xir::IfInst *inst) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr, "SPIR-V If emission requires a control-flow plan.");
    auto &region = _control_flow_plan->if_region(inst);
    auto condition = _emit_value(inst->condition());
    auto *true_target = _physical_block(region.true_target);
    auto *false_target = _physical_block(region.false_target);
    auto *merge_target = _physical_block(region.merge_target);
    LUISA_ASSERT(true_target != nullptr && false_target != nullptr && merge_target != nullptr,
                 "SPIR-V If region contains an unbound physical block.");

    if (region.emit_selection_merge) {
        auto merge = std::make_unique<spv::Instruction>(
            spv::Op::OpSelectionMerge);
        merge->reserveOperands(2u);
        merge->addIdOperand(merge_target->getId());
        merge->addImmediateOperand(
            spv::SelectionControlMask::MaskNone);
        _builder.getBuildPoint()->addInstruction(std::move(merge));
    }
    _builder.createConditionalBranch(condition, true_target, false_target);
}

void SpirvCodegenEntry::_emit_loop_inst(const xir::LoopInst *inst) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr, "SPIR-V Loop emission requires a control-flow plan.");
    auto &region = _control_flow_plan->loop_region(inst);
    auto *entry = _physical_block(region.entry_target);
    LUISA_ASSERT(entry != nullptr,
                 "SPIR-V Loop region contains an unbound entry block.");
    _builder.createBranch(false, entry);
}

void SpirvCodegenEntry::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr, "SPIR-V SimpleLoop emission requires a control-flow plan.");
    auto &region = _control_flow_plan->simple_loop_region(inst);
    auto *header = _synthetic_blocks.at(region.header_synthetic_index);
    auto *continue_block = _synthetic_blocks.at(region.continue_synthetic_index);
    auto *body = _xir_block_entry(region.body);
    auto *merge = _physical_block(region.merge_target);
    LUISA_ASSERT(header != nullptr && continue_block != nullptr && body != nullptr && merge != nullptr,
                 "SPIR-V SimpleLoop region contains an unbound physical block.");
    LUISA_ASSERT(continue_block != merge, "SPIR-V SimpleLoop merge and continue targets must be distinct.");

    auto *owner_tail = _builder.getBuildPoint();
    _builder.createBranch(false, header);
    _builder.setBuildPoint(header);
    _builder.createLoopMerge(merge, continue_block, spv::LoopControlMask::MaskNone, {});
    _builder.createBranch(false, body);
    _builder.setBuildPoint(continue_block);
    _builder.createBranch(false, header);
    _builder.setBuildPoint(owner_tail);
}

void SpirvCodegenEntry::_emit_switch_inst(const xir::SwitchInst *inst) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr,
                 "SPIR-V Switch emission requires a control-flow plan.");
    auto &region = _control_flow_plan->switch_region(inst);
    auto selector = _emit_value(inst->value());
    auto *selector_type = inst->value()->type();
    auto layout = plan_spirv_switch_instruction(
        selector_type, inst->case_count());
    if (!layout) {
        LUISA_ERROR_WITH_LOCATION("{}", layout.diagnostic);
    }
    auto selector_bit_width = layout.selector_bit_width;
    if (selector_type->is_bool()) {
        auto uint_type = _builder.makeUintType(32u);
        selector = _builder.createTriOp(
            spv::Op::OpSelect, uint_type, selector,
            _builder.makeUintConstant(1u), _builder.makeUintConstant(0u));
        selector_bit_width = 32u;
    }
    auto *logical_header = _builder.getBuildPoint();
    auto *physical_switch_header = logical_header;
    if (region.loop_wrapped) {
        auto *dispatch =
            _synthetic_blocks.at(region.dispatch_synthetic_index);
        auto *continue_target =
            _synthetic_blocks.at(region.continue_synthetic_index);
        auto *loop_merge = _physical_block(region.loop_merge_target);
        LUISA_ASSERT(dispatch != nullptr && continue_target != nullptr &&
                         loop_merge != nullptr &&
                         dispatch != continue_target &&
                         dispatch != loop_merge &&
                         continue_target != loop_merge,
                     "SPIR-V wrapped Switch has invalid physical loop roles.");
        _builder.createLoopMerge(
            loop_merge, continue_target,
            spv::LoopControlMask::MaskNone, {});
        _builder.createBranch(false, dispatch);
        _builder.setBuildPoint(dispatch);
        physical_switch_header = dispatch;
    }
    auto *merge_target = _physical_block(region.merge_target);
    auto *default_target = _physical_block(region.default_target);
    LUISA_ASSERT(merge_target != nullptr && default_target != nullptr,
                 "SPIR-V Switch has an unbound merge or default target.");
    auto selection_merge = std::make_unique<spv::Instruction>(spv::Op::OpSelectionMerge);
    selection_merge->reserveOperands(2u);
    selection_merge->addIdOperand(merge_target->getId());
    selection_merge->addImmediateOperand(spv::SelectionControlMask::MaskNone);
    _builder.getBuildPoint()->addInstruction(std::move(selection_merge));

    auto switch_instruction = std::make_unique<spv::Instruction>(spv::Op::OpSwitch);
    switch_instruction->reserveOperands(layout.operand_word_count);
    switch_instruction->addIdOperand(selector);
    switch_instruction->addIdOperand(default_target->getId());
    luisa::vector<spv::Block *> predecessor_targets{default_target};
    LUISA_ASSERT(region.case_operand_order.size() == inst->case_count(),
                 "SPIR-V Switch case order is incomplete.");
    for (auto case_index : region.case_operand_order) {
        LUISA_ASSERT(case_index < inst->case_count(),
                     "SPIR-V Switch case order contains an invalid index.");
        auto literal = inst->case_value(case_index);
        LUISA_ASSERT(
            literal == xir::SwitchInst::canonicalize_case_value(
                           inst->value()->type(), literal),
            "SPIR-V Switch case literal exceeds its selector bit width.");
        switch_instruction->addImmediateOperand(static_cast<uint32_t>(literal));
        if (selector_bit_width == 64u) {
            switch_instruction->addImmediateOperand(static_cast<uint32_t>(literal >> 32u));
        }
        auto *case_target = _physical_block(region.case_targets[case_index]);
        LUISA_ASSERT(case_target != nullptr,
                     "SPIR-V Switch has an unbound case target.");
        switch_instruction->addIdOperand(case_target->getId());
        if (std::find(predecessor_targets.begin(), predecessor_targets.end(), case_target) ==
            predecessor_targets.end()) {
            predecessor_targets.emplace_back(case_target);
        }
    }
    physical_switch_header->addInstruction(std::move(switch_instruction));
    for (auto *target : predecessor_targets) {
        target->addPredecessor(physical_switch_header);
    }
    if (region.loop_wrapped) {
        _builder.setBuildPoint(logical_header);
    }
}

void SpirvCodegenEntry::_emit_loop_merge(
    const ControlFlowPlan::LoopRegion &region) noexcept {
    auto *continue_target = _physical_block(region.continue_target);
    auto *merge_target = _physical_block(region.merge_target);
    LUISA_ASSERT(continue_target != nullptr && merge_target != nullptr &&
                     continue_target != merge_target,
                 "SPIR-V Loop prepare has invalid merge/continue targets.");
    _builder.createLoopMerge(
        merge_target, continue_target,
        spv::LoopControlMask::MaskNone, {});
}

void SpirvCodegenEntry::_emit_branch_inst(const xir::BranchInst *inst) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr, "SPIR-V Branch emission requires a control-flow plan.");
    auto target = _control_flow_plan->edge_target(inst);
    if (auto *loop = _control_flow_plan->loop_with_prepare(
            inst->parent_block())) {
        LUISA_ASSERT(
            loop->prepare_kind == SpirvLoopPrepareKind::UNCONDITIONAL &&
                loop->prepare == inst->parent_block() &&
                inst->target_block() == loop->body &&
                target == loop->body_target,
            "SPIR-V codegen received a noncanonical unconditional "
            "Loop.prepare.");
        auto *body_target = _physical_block(target);
        LUISA_ASSERT(body_target != nullptr,
                     "SPIR-V Loop prepare has an unbound body target.");
        _emit_loop_merge(*loop);
        _builder.createBranch(false, body_target);
        return;
    }
    _builder.createBranch(false, _physical_block(target));
}

void SpirvCodegenEntry::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr, "SPIR-V ConditionalBranch emission requires a control-flow plan.");
    auto *loop = _control_flow_plan->loop_with_prepare(inst->parent_block());
    LUISA_ASSERT(loop != nullptr &&
                     loop->prepare_kind ==
                         SpirvLoopPrepareKind::CONDITIONAL &&
                     loop->prepare == inst->parent_block() &&
                     inst->true_block() == loop->body && inst->false_block() == loop->merge,
                 "SPIR-V codegen received a raw ConditionalBranch outside canonical Loop.prepare.");
    auto condition = _emit_value(inst->condition());
    auto *body_target = _physical_block(loop->body_target);
    auto *merge_target = _physical_block(loop->merge_target);
    LUISA_ASSERT(body_target != nullptr && merge_target != nullptr,
                 "SPIR-V Loop prepare has an unbound body or merge target.");
    _emit_loop_merge(*loop);
    _builder.createConditionalBranch(condition,
                                     body_target,
                                     merge_target);
}

spv::Block *SpirvCodegenEntry::_resolve_branch_target(const xir::BasicBlock *bb) const noexcept {
    LUISA_ASSERT(_control_flow_plan != nullptr && _current_xir_block != nullptr &&
                     _current_xir_block->is_terminated(),
                 "SPIR-V branch target lookup requires an active planned XIR terminator.");
    auto *terminator = _current_xir_block->terminator();
    switch (terminator->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::BREAK:
            LUISA_ASSERT(static_cast<const xir::BreakInst *>(terminator)->target_block() == bb,
                         "SPIR-V Break target differs from its control-flow plan.");
            break;
        case xir::DerivedInstructionTag::CONTINUE:
            LUISA_ASSERT(static_cast<const xir::ContinueInst *>(terminator)->target_block() == bb,
                         "SPIR-V Continue target differs from its control-flow plan.");
            break;
        case xir::DerivedInstructionTag::BRANCH:
            LUISA_ASSERT(static_cast<const xir::BranchInst *>(terminator)->target_block() == bb,
                         "SPIR-V Branch target differs from its control-flow plan.");
            break;
        default:
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V branch target lookup called for unplanned terminator {}.",
                xir::to_string(terminator->derived_instruction_tag()));
    }
    return _physical_block(_control_flow_plan->edge_target(terminator));
}

}// namespace lc::spirv
