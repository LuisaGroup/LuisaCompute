#include "control_flow_plan.h"
#include "structural_closure.h"

#include <algorithm>
#include <utility>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/break.h>
#include <luisa/xir/instructions/continue.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/passes/dom_tree.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

SpirvLoopPreparePlan plan_spirv_loop_prepare(
    const xir::LoopInst *loop) noexcept {
    if (loop == nullptr) {
        return {.diagnostic =
                    "Native XIR-to-SPIR-V requires a non-null Loop."};
    }
    if (loop->operand_count() != 1u) {
        return {.diagnostic =
                    "Native XIR-to-SPIR-V requires Loop to have exactly one "
                    "Loop.prepare operand."};
    }
    auto *prepare_value =
        loop->operand(xir::LoopInst::operand_index_prepare_block);
    if (prepare_value == nullptr || !prepare_value->isa<xir::BasicBlock>()) {
        return {.diagnostic =
                    "Native XIR-to-SPIR-V requires the Loop.prepare operand "
                    "to be a non-null BasicBlock."};
    }
    auto *prepare = static_cast<const xir::BasicBlock *>(prepare_value);
    auto *body = loop->body_block();
    auto *merge = loop->merge_block();
    if (body == nullptr || merge == nullptr) {
        return {.diagnostic =
                    "Native XIR-to-SPIR-V requires non-null Loop.body and "
                    "Loop.merge blocks."};
    }
    if (!prepare->is_terminated()) {
        return {.diagnostic =
                    "Native XIR-to-SPIR-V requires Loop.prepare to be "
                    "terminated."};
    }
    auto *terminator = prepare->terminator();
    if (terminator->isa<xir::BranchInst>()) {
        auto *branch = static_cast<const xir::BranchInst *>(terminator);
        if (branch->operand_count() != 1u ||
            branch->operand(
                xir::BranchTerminatorInstruction::operand_index_target) !=
                body) {
            return {.diagnostic =
                        "Native XIR-to-SPIR-V requires canonical "
                        "unconditional Loop.prepare shape: Branch(Loop.body)."};
        }
        return {.kind = SpirvLoopPrepareKind::UNCONDITIONAL};
    }
    if (terminator->isa<xir::ConditionalBranchInst>()) {
        auto *branch =
            static_cast<const xir::ConditionalBranchInst *>(terminator);
        auto *condition = branch->operand_count() >
                                  xir::ConditionalBranchTerminatorInstruction::
                                      operand_index_condition ?
                              branch->operand(
                                  xir::ConditionalBranchTerminatorInstruction::
                                      operand_index_condition) :
                              nullptr;
        auto *true_target = branch->operand_count() >
                                    xir::ConditionalBranchTerminatorInstruction::
                                        operand_index_true_target ?
                                branch->operand(
                                    xir::ConditionalBranchTerminatorInstruction::
                                        operand_index_true_target) :
                                nullptr;
        auto *false_target = branch->operand_count() >
                                     xir::ConditionalBranchTerminatorInstruction::
                                         operand_index_false_target ?
                                 branch->operand(
                                     xir::ConditionalBranchTerminatorInstruction::
                                         operand_index_false_target) :
                                 nullptr;
        if (branch->operand_count() != 3u || condition == nullptr ||
            condition->type() != Type::of<bool>() || true_target != body ||
            false_target != merge) {
            return {.diagnostic =
                        "Native XIR-to-SPIR-V requires canonical conditional "
                        "Loop.prepare shape: ConditionalBranch(bool, "
                        "Loop.body, Loop.merge)."};
        }
        return {.kind = SpirvLoopPrepareKind::CONDITIONAL};
    }
    return {.diagnostic =
                "Native XIR-to-SPIR-V requires Loop.prepare to end with "
                "Branch(Loop.body) or ConditionalBranch(bool, Loop.body, "
                "Loop.merge)."};
}

ControlFlowPlan::Target ControlFlowPlan::Target::xir(const xir::BasicBlock *block) noexcept {
    return Target{.kind = Kind::XIR_BLOCK, .xir_block = block};
}

ControlFlowPlan::Target ControlFlowPlan::Target::synthetic(size_t index) noexcept {
    return Target{.kind = Kind::SYNTHETIC_BLOCK, .synthetic_index = index};
}

void ControlFlowPlan::_add_role(const xir::BasicBlock *block, BlockRole role) noexcept {
    LUISA_ASSERT(block != nullptr, "SPIR-V control-flow plan: null block for role {}.", static_cast<uint32_t>(role));
    auto iter = _block_indices.find(block);
    LUISA_ASSERT(iter != _block_indices.end(),
                 "SPIR-V control-flow plan: structured block {} is outside the executable/structural closure.",
                 static_cast<const void *>(block));
    _blocks[iter->second].roles |= static_cast<uint32_t>(role);
}

void ControlFlowPlan::_register_merge(const xir::BasicBlock *block,
                                      const xir::Instruction *owner) noexcept {
    LUISA_ASSERT(block != nullptr, "SPIR-V control-flow plan: construct has a null merge block.");
    if (auto [iter, inserted] = _merge_owners.emplace(block, owner); !inserted) {
        LUISA_ERROR_WITH_LOCATION(
            "SPIR-V control-flow plan rejected merge block {}: it is owned by both {} and {}. "
            "Normalized XIR requires one merge owner per block.",
            static_cast<const void *>(block),
            xir::to_string(iter->second->derived_instruction_tag()),
            xir::to_string(owner->derived_instruction_tag()));
    }
}

size_t ControlFlowPlan::_add_synthetic(SyntheticBlockKind kind,
                                       const xir::Instruction *owner,
                                       Target continuation) noexcept {
    auto index = _synthetic_blocks.size();
    _synthetic_blocks.emplace_back(SyntheticBlockPlan{
        .kind = kind,
        .owner = owner,
        .ordinal = index,
        .continuation = continuation});
    return index;
}

ControlFlowPlan::Target ControlFlowPlan::_resolve_loop_boundary_target(const xir::BasicBlock *block) const noexcept {
    LUISA_ASSERT(block != nullptr, "SPIR-V control-flow plan: null branch target.");
    if (auto iter = _simple_loop_body_indices.find(block); iter != _simple_loop_body_indices.end()) {
        return Target::synthetic(_simple_loop_regions[iter->second].continue_synthetic_index);
    }
    LUISA_ASSERT(_block_indices.contains(block),
                 "SPIR-V control-flow plan: branch target {} is outside the executable/structural closure.",
                 static_cast<const void *>(block));
    return Target::xir(block);
}

ControlFlowPlan::Target ControlFlowPlan::_resolve_ordinary_target(
    const xir::BasicBlock *source,
    const xir::BasicBlock *target) const noexcept {
    LUISA_ASSERT(source != nullptr && target != nullptr,
                 "SPIR-V control-flow plan: null ordinary edge endpoint.");
    if (auto merge_iter = _merge_targets.find(target);
        merge_iter != _merge_targets.end()) {
        auto scope_iter = _merge_scopes.find(target);
        LUISA_ASSERT(scope_iter != _merge_scopes.end(),
                     "SPIR-V construct merge has no frozen logical scope.");
        if (scope_iter->second.contains(source)) {
            return merge_iter->second;
        }
    }
    if (auto switch_iter = _wrapped_switch_header_indices.find(target);
        switch_iter != _wrapped_switch_header_indices.end()) {
        auto source_iter = _wrapped_switch_backedge_sources.find(target);
        LUISA_ASSERT(source_iter != _wrapped_switch_backedge_sources.end(),
                     "SPIR-V wrapped Switch has no frozen backedge-source set.");
        if (source_iter->second.contains(source)) {
            auto &region = _switch_regions.at(switch_iter->second);
            if (source == target && region.has_header_case_target) {
                return Target::synthetic(region.header_case_synthetic_index);
            }
            return Target::synthetic(region.continue_synthetic_index);
        }
    }
    return _resolve_loop_boundary_target(target);
}

ControlFlowPlan::FunctionEntryBoundaryValidation
ControlFlowPlan::validate_function_entry_boundary(
    const xir::FunctionDefinition *function) noexcept {
    LUISA_ASSERT(function != nullptr && function->body_block() != nullptr,
                 "SPIR-V function-entry validation requires a function body.");
    auto *entry = function->body_block();
    FunctionEntryBoundaryValidation validation;
    luisa::unordered_set<const xir::BasicBlock *> predecessors;
    for (auto *block : collect_spirv_codegen_structural_closure(function)) {
        if (block == entry) {
            for (auto *instruction : block->instructions()) {
                validation.phi_count += instruction->isa<xir::PhiInst>();
            }
        }
        if (!block->is_terminated()) { continue; }
        for (auto *operand_use : block->terminator()->operand_uses()) {
            if (operand_use->value() == entry) {
                predecessors.emplace(block);
            }
        }
    }
    validation.logical_predecessor_count = predecessors.size();
    return validation;
}

ControlFlowPlan::PhysicalLoopBoundaryValidation
ControlFlowPlan::validate_physical_loop_boundary(
    luisa::span<const PhysicalLoopPredecessorFacts> predecessors) noexcept {
    PhysicalLoopBoundaryValidation validation;
    validation.reachable_predecessor_count = predecessors.size();
    for (auto &&predecessor : predecessors) {
        if (!predecessor.dominated_by_header) {
            validation.entry_edge_count++;
            continue;
        }
        validation.backedge_edge_count++;
        // These two fields describe the unique backedge. Multiple backedges are
        // rejected by the count invariant, so their aggregate values are kept
        // conservative and cannot accidentally make validation succeed.
        if (validation.backedge_edge_count == 1u) {
            validation.backedge_dominated_by_continue_target =
                predecessor.dominated_by_continue_target;
            validation.backedge_dominated_by_merge_target =
                predecessor.dominated_by_merge_target;
        } else {
            validation.backedge_dominated_by_continue_target = false;
            validation.backedge_dominated_by_merge_target = true;
        }
    }
    return validation;
}

ControlFlowPlan ControlFlowPlan::_create(
    const xir::FunctionDefinition *function,
    bool enforce_physical_loop_boundaries) noexcept {
    LUISA_ASSERT(function != nullptr && function->body_block() != nullptr,
                 "SPIR-V control-flow plan requires a function definition with a body.");
    ControlFlowPlan plan;
    plan._function = function;
    auto entry_boundary = validate_function_entry_boundary(function);
    if (!entry_boundary.succeeded()) {
        if (!enforce_physical_loop_boundaries) {
            plan._planning_diagnostic = luisa::format(
                "SPIR-V control-flow plan rejected function body entry with {} "
                "logical predecessor(s) and {} Phi instruction(s).",
                entry_boundary.logical_predecessor_count,
                entry_boundary.phi_count);
            return plan;
        }
        LUISA_ERROR_WITH_LOCATION(
            "SPIR-V control-flow plan rejected function body entry with {} "
            "logical predecessor(s) and {} Phi instruction(s). The physical "
            "SPIR-V function entry is a backend-owned boundary and cannot be "
            "targeted or contain OpPhi.",
            entry_boundary.logical_predecessor_count,
            entry_boundary.phi_count);
    }
    auto reject_planning_precondition =
        [&](luisa::string diagnostic) noexcept {
            if (!enforce_physical_loop_boundaries) {
                plan._planning_diagnostic = std::move(diagnostic);
                return true;
            }
            LUISA_ERROR_WITH_LOCATION("{}", diagnostic);
        };

    auto append_block = [&](const xir::BasicBlock *block) noexcept {
        LUISA_ASSERT(block != nullptr && block->parent_function() == function,
                     "SPIR-V control-flow plan encountered a foreign block.");
        LUISA_ASSERT(block->is_terminated(),
                     "SPIR-V control-flow plan rejected unterminated XIR block {}.",
                     static_cast<const void *>(block));
        if (plan._block_indices.contains(block)) { return; }
        auto index = plan._blocks.size();
        plan._block_indices.emplace(block, index);
        plan._blocks.emplace_back(BlockPlan{
            .block = block,
            .roles = block == function->body_block() ? static_cast<uint32_t>(BlockRole::FUNCTION_ENTRY) : 0u,
            .schedule_index = index});
    };
    for (auto *block : collect_spirv_codegen_structural_closure(function)) {
        append_block(block);
    }
    // Function-owned blocks outside this closure are true orphans. They do not
    // participate in executable SPIR-V and are deliberately omitted; codegen
    // legality must not depend on optional DCE at opt0.

    auto dom = xir::compute_dom_tree(
        const_cast<xir::FunctionDefinition *>(function));
    // XIR keeps loop-exit guards as IfInst so every high-level transform sees
    // structured control flow. SPIR-V deliberately has a narrower exception:
    // a conditional branch that selects an enclosing loop break/continue path
    // needs no OpSelectionMerge. Emitting one would create a selection whose
    // non-merge arm immediately exits that selection.
    struct LoopBoundaryGuardPlan {
        const xir::BasicBlock *target{nullptr};
        const xir::BasicBlock *logical_exit_predecessor{nullptr};
        luisa::vector<const xir::BasicBlock *> pruned_blocks;
    };
    auto loop_boundary_guard_plan =
        [&](const xir::IfInst *inst) noexcept
        -> LoopBoundaryGuardPlan {
        auto *header = inst->parent_block();
        auto *merge = inst->merge_block();
        auto true_is_merge = inst->true_block() == merge;
        auto false_is_merge = inst->false_block() == merge;
        if (true_is_merge == false_is_merge) { return {}; }
        auto *exit_entry =
            true_is_merge ? inst->false_block() :
                            inst->true_block();

        struct LoopBoundary {
            const xir::BasicBlock *owner;
            const xir::BasicBlock *continue_target;
            const xir::BasicBlock *merge;
        };
        luisa::vector<LoopBoundary> candidates;
        for (auto &block_plan : plan._blocks) {
            auto *block = block_plan.block;
            auto *term = block->terminator();
            const xir::BasicBlock *continue_target = nullptr;
            const xir::BasicBlock *loop_merge = nullptr;
            if (term->isa<xir::LoopInst>()) {
                auto *loop =
                    static_cast<const xir::LoopInst *>(term);
                continue_target = loop->update_block();
                if (continue_target == nullptr) {
                    continue_target = loop->prepare_block();
                }
                loop_merge = loop->merge_block();
            } else if (term->isa<xir::SimpleLoopInst>()) {
                auto *loop = static_cast<
                    const xir::SimpleLoopInst *>(term);
                continue_target = loop->body_block();
                loop_merge = loop->merge_block();
            } else {
                continue;
            }
            if (continue_target == nullptr || loop_merge == nullptr ||
                !dom.contains(const_cast<xir::BasicBlock *>(block)) ||
                !dom.contains(const_cast<xir::BasicBlock *>(header)) ||
                !dom.dominates(
                    const_cast<xir::BasicBlock *>(block),
                    const_cast<xir::BasicBlock *>(header)) ||
                (dom.contains(
                     const_cast<xir::BasicBlock *>(loop_merge)) &&
                 dom.dominates(
                     const_cast<xir::BasicBlock *>(loop_merge),
                     const_cast<xir::BasicBlock *>(header)))) {
                continue;
            }
            candidates.emplace_back(LoopBoundary{
                block, continue_target, loop_merge});
        }

        for (auto boundary : candidates) {
            luisa::unordered_set<const xir::BasicBlock *>
                visited;
            luisa::vector<const xir::BasicBlock *> pruned_blocks;
            auto *expected_predecessor = header;
            auto *block = exit_entry;
            while (block != nullptr &&
                   visited.emplace(block).second) {
                if (block == boundary.continue_target ||
                    block == boundary.merge) {
                    return LoopBoundaryGuardPlan{
                        .target = block,
                        .logical_exit_predecessor =
                            pruned_blocks.empty() ?
                                nullptr :
                                pruned_blocks.back(),
                        .pruned_blocks =
                            std::move(pruned_blocks)};
                }
                if (block == merge || !block->is_terminated()) {
                    break;
                }
                auto *term = block->terminator();
                auto iter = block->instructions().begin();
                auto only_terminator =
                    iter != block->instructions().end() &&
                    *iter == term;
                if (!only_terminator) { break; }

                // Pruning is only semantics-preserving when this proxy
                // chain is owned exclusively by the guard. A shared block
                // could carry another physical entry and must remain a
                // normal structured construct.
                auto predecessor_count = size_t{0u};
                auto has_unexpected_predecessor = false;
                block->traverse_predecessors(
                    false,
                    [&](const xir::BasicBlock *predecessor) noexcept {
                        ++predecessor_count;
                        has_unexpected_predecessor |=
                            predecessor != expected_predecessor;
                    });
                if (predecessor_count != 1u ||
                    has_unexpected_predecessor) {
                    break;
                }
                pruned_blocks.emplace_back(block);
                if (term->isa<xir::BreakInst>()) {
                    if (static_cast<const xir::BreakInst *>(
                            term)
                            ->target_block() ==
                        boundary.merge) {
                        return LoopBoundaryGuardPlan{
                            .target = boundary.merge,
                            .logical_exit_predecessor = block,
                            .pruned_blocks =
                                std::move(pruned_blocks)};
                    }
                    break;
                }
                if (term->isa<xir::ContinueInst>()) {
                    if (static_cast<const xir::ContinueInst *>(
                            term)
                            ->target_block() ==
                        boundary.continue_target) {
                        return LoopBoundaryGuardPlan{
                            .target =
                                boundary.continue_target,
                            .logical_exit_predecessor = block,
                            .pruned_blocks =
                                std::move(pruned_blocks)};
                    }
                    break;
                }
                if (!term->isa<xir::BranchInst>()) {
                    break;
                }
                expected_predecessor = block;
                block = static_cast<
                            const xir::BranchInst *>(term)
                            ->target_block();
            }
        }
        return {};
    };

    // First pass: freeze construct roles and allocate logical synthetic slots.
    // No ordinary edge is resolved until every loop entry/continue role is known.
    for (auto &block_plan : plan._blocks) {
        auto *block = block_plan.block;
        auto *terminator = block->terminator();
        switch (terminator->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                auto *inst = static_cast<const xir::IfInst *>(terminator);
                LUISA_ASSERT(inst->condition() != nullptr && inst->condition()->type() == Type::of<bool>(),
                             "SPIR-V control-flow plan rejected If with a non-boolean condition.");
                LUISA_ASSERT(inst->true_block() != nullptr && inst->false_block() != nullptr && inst->merge_block() != nullptr,
                             "SPIR-V control-flow plan rejected If with a null role block.");
                LUISA_ASSERT(inst->merge_block() != block,
                             "SPIR-V control-flow plan rejected If whose header is also its merge.");
                auto index = plan._if_regions.size();
                auto loop_boundary_guard =
                    loop_boundary_guard_plan(inst);
                auto *loop_boundary_exit_target =
                    loop_boundary_guard.target;
                auto emit_selection_merge =
                    loop_boundary_exit_target == nullptr;
                plan._if_regions.emplace_back(IfRegion{
                    .instruction = inst,
                    .header = block,
                    .true_target = Target::xir(inst->true_block()),
                    .false_target = Target::xir(inst->false_block()),
                    .merge_target = Target::xir(inst->merge_block()),
                    .emit_selection_merge = emit_selection_merge,
                    .loop_boundary_exit_target =
                        loop_boundary_exit_target,
                    .loop_boundary_exit_predecessor =
                        loop_boundary_guard
                            .logical_exit_predecessor});
                for (auto *pruned_block :
                     loop_boundary_guard.pruned_blocks) {
                    auto &pruned_plan =
                        plan._blocks.at(
                            plan._block_indices.at(pruned_block));
                    LUISA_ASSERT(
                        !pruned_plan.physically_pruned,
                        "SPIR-V loop-boundary proxy block was claimed "
                        "by multiple guards.");
                    pruned_plan.physically_pruned = true;
                }
                plan._if_indices.emplace(inst, index);
                plan._add_role(block, BlockRole::IF_HEADER);
                plan._add_role(inst->true_block(), BlockRole::IF_TRUE_ENTRY);
                plan._add_role(inst->false_block(), BlockRole::IF_FALSE_ENTRY);
                if (emit_selection_merge) {
                    plan._register_merge(inst->merge_block(), inst);
                    plan._add_role(
                        inst->merge_block(),
                        BlockRole::SELECTION_MERGE);
                }
                break;
            }
            case xir::DerivedInstructionTag::LOOP: {
                auto *inst = static_cast<const xir::LoopInst *>(terminator);
                auto *prepare = inst->prepare_block();
                auto *body = inst->body_block();
                auto *update = inst->update_block();
                auto *merge = inst->merge_block();
                LUISA_ASSERT(prepare != nullptr && body != nullptr && update != nullptr && merge != nullptr,
                             "SPIR-V control-flow plan rejected Loop with a null role block.");
                auto prepare_plan = plan_spirv_loop_prepare(inst);
                if (!prepare_plan.succeeded()) {
                    if (reject_planning_precondition(
                            std::move(prepare_plan.diagnostic))) {
                        return plan;
                    }
                }
                luisa::unordered_set<const xir::BasicBlock *> role_blocks;
                role_blocks.emplace(block);
                role_blocks.emplace(prepare);
                role_blocks.emplace(body);
                role_blocks.emplace(update);
                role_blocks.emplace(merge);
                LUISA_ASSERT(role_blocks.size() == 5u,
                             "SPIR-V control-flow plan rejected Loop with overlapping owner/prepare/body/update/merge roles.");
                auto index = plan._loop_regions.size();
                plan._loop_regions.emplace_back(LoopRegion{
                    .instruction = inst,
                    .owner = block,
                    .prepare = prepare,
                    .body = body,
                    .update = update,
                    .merge = merge,
                    .entry_target = Target::xir(prepare),
                    .body_target = Target::xir(body),
                    .continue_target = Target::xir(update),
                    .merge_target = Target::xir(merge),
                    .prepare_kind = prepare_plan.kind,
                });
                plan._loop_indices.emplace(inst, index);
                if (!plan._loop_prepare_indices.emplace(prepare, index).second) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected a prepare block "
                            "shared by multiple loops.")) {
                        return plan;
                    }
                }
                if (!plan._loop_update_indices.emplace(update, index).second) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected an update block "
                            "shared by multiple loops.")) {
                        return plan;
                    }
                }
                plan._register_merge(merge, inst);
                plan._add_role(block, BlockRole::LOOP_OWNER);
                plan._add_role(prepare, BlockRole::LOOP_PREPARE);
                plan._add_role(body, BlockRole::LOOP_BODY);
                plan._add_role(update, BlockRole::LOOP_UPDATE);
                plan._add_role(merge, BlockRole::LOOP_MERGE);
                break;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                auto *inst = static_cast<const xir::SimpleLoopInst *>(terminator);
                auto *body = inst->body_block();
                auto *merge = inst->merge_block();
                LUISA_ASSERT(body != nullptr && merge != nullptr,
                             "SPIR-V control-flow plan rejected SimpleLoop with a null role block.");
                LUISA_ASSERT(block != body && block != merge && body != merge,
                             "SPIR-V control-flow plan rejected SimpleLoop with overlapping owner/body/merge roles.");
                auto synthetic_header = plan._add_synthetic(
                    SyntheticBlockKind::SIMPLE_LOOP_HEADER, inst, Target::xir(body));
                auto synthetic_continue = plan._add_synthetic(
                    SyntheticBlockKind::SIMPLE_LOOP_CONTINUE, inst, Target::synthetic(synthetic_header));
                auto index = plan._simple_loop_regions.size();
                plan._simple_loop_regions.emplace_back(SimpleLoopRegion{
                    .instruction = inst,
                    .owner = block,
                    .body = body,
                    .merge = merge,
                    .merge_target = Target::xir(merge),
                    .header_synthetic_index = synthetic_header,
                    .continue_synthetic_index = synthetic_continue});
                plan._simple_loop_indices.emplace(inst, index);
                if (!plan._simple_loop_body_indices.emplace(body, index).second) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected a body block "
                            "shared by multiple SimpleLoops.")) {
                        return plan;
                    }
                }
                plan._register_merge(merge, inst);
                plan._add_role(block, BlockRole::SIMPLE_LOOP_OWNER);
                plan._add_role(body, BlockRole::SIMPLE_LOOP_BODY);
                plan._add_role(merge, BlockRole::SIMPLE_LOOP_MERGE);
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto *inst = static_cast<const xir::SwitchInst *>(terminator);
                auto *selector = inst->value();
                auto *selector_type = selector == nullptr ? nullptr : selector->type();
                LUISA_ASSERT(selector_type != nullptr && selector_type->is_scalar() &&
                                 (selector_type->is_bool() || selector_type->is_int() || selector_type->is_uint()),
                             "SPIR-V control-flow plan rejected Switch with a non-integral scalar selector.");
                LUISA_ASSERT(inst->default_block() != nullptr && inst->merge_block() != nullptr &&
                                 inst->merge_block() != block,
                             "SPIR-V control-flow plan rejected Switch with an invalid default/merge role.");
                SwitchRegion region{
                    .instruction = inst,
                    .header = block,
                    .default_target = Target::xir(inst->default_block()),
                    .merge_target = Target::xir(inst->merge_block())};
                region.case_targets.reserve(inst->case_count());
                region.case_operand_order.reserve(inst->case_count());
                luisa::unordered_set<xir::SwitchInst::case_value_type> case_values;
                for (size_t case_index = 0u; case_index < inst->case_count(); ++case_index) {
                    auto *case_block = inst->case_block(case_index);
                    auto case_value = inst->case_value(case_index);
                    auto canonical_value = xir::SwitchInst::canonicalize_case_value(
                        inst->value()->type(), case_value);
                    LUISA_ASSERT(case_block != nullptr && case_value == canonical_value &&
                                     case_values.emplace(canonical_value).second,
                                 "SPIR-V control-flow plan rejected Switch with a null case block, noncanonical case value, or duplicate case value.");
                    region.case_targets.emplace_back(Target::xir(case_block));
                    region.case_operand_order.emplace_back(case_index);
                    plan._add_role(case_block, BlockRole::SWITCH_CASE_ENTRY);
                }
                auto index = plan._switch_regions.size();
                plan._switch_indices.emplace(inst, index);
                plan._switch_regions.emplace_back(std::move(region));
                plan._register_merge(inst->merge_block(), inst);
                plan._add_role(block, BlockRole::SWITCH_HEADER);
                plan._add_role(inst->default_block(), BlockRole::SWITCH_DEFAULT_ENTRY);
                plan._add_role(inst->merge_block(), BlockRole::SWITCH_MERGE);
                break;
            }
            case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                if (reject_planning_precondition(
                        "SPIR-V control-flow plan rejected RayQueryLoopInst. "
                        "lower_ray_query_to_loop must run before codegen.")) {
                    return plan;
                }
            default: break;
        }
    }

    // Loop entry/continue roles are intrinsic physical roles, unlike construct
    // merges, which can be separated with a one-way proxy. Sharing one logical
    // block between two boundary roles would make ordinary edges ambiguous
    // (for example, a Loop.update that is also a SimpleLoop.body). Normalized
    // XIR must split such boundaries before this handoff.
    for (auto &block_plan : plan._blocks) {
        auto boundary_role_count =
            static_cast<uint32_t>(block_plan.has_role(BlockRole::LOOP_PREPARE)) +
            static_cast<uint32_t>(block_plan.has_role(BlockRole::LOOP_UPDATE)) +
            static_cast<uint32_t>(block_plan.has_role(BlockRole::SIMPLE_LOOP_BODY));
        if (boundary_role_count > 1u) {
            if (reject_planning_precondition(luisa::format(
                    "SPIR-V control-flow plan rejected block {} shared by "
                    "multiple loop entry/continue roles. restructure_cfg "
                    "must split the boundary before codegen.",
                    static_cast<const void *>(block_plan.block)))) {
                return plan;
            }
        }
    }

    // Freeze each construct's logical scope before resolving physical edges.
    // A source is inside an ordinary acyclic construct exactly when its header
    // dominates the source and the construct merge does not. A merge that is
    // itself an enclosing loop boundary is different: it can dominate the
    // construct header (Loop.prepare, SimpleLoop.body, and a cyclic Switch
    // header are the important cases), so subtracting its dominance region
    // would erase the whole scope.
    // In that cyclic-boundary case, header dominance alone identifies sources
    // owned by the construct.
    //
    // This distinction is essential
    // when one XIR block names both a selection/loop merge and another loop's
    // entry or continue role: only edges leaving the owning construct may pass
    // through its merge proxy; loop backedges and sibling continues must retain
    // their own structural target.
    for (auto [merge, owner] : plan._merge_owners) {
        auto *header = owner->parent_block();
        LUISA_ASSERT(header != nullptr &&
                         dom.contains(const_cast<xir::BasicBlock *>(header)),
                     "SPIR-V construct header is absent from the dominance tree.");
        auto merge_dominates_header =
            dom.contains(const_cast<xir::BasicBlock *>(merge)) &&
            dom.dominates(
                const_cast<xir::BasicBlock *>(merge),
                const_cast<xir::BasicBlock *>(header));
        if (merge_dominates_header &&
            !plan._loop_prepare_indices.contains(merge) &&
            !plan._simple_loop_body_indices.contains(merge) &&
            !plan.block(merge).has_role(BlockRole::SWITCH_HEADER)) {
            if (reject_planning_precondition(
                    "SPIR-V control-flow plan rejected a backward construct "
                    "merge to a block that is not an enclosing loop header. "
                    "Only Loop.prepare, SimpleLoop.body, and cyclic Switch "
                    "header boundaries can be normalized through a merge "
                    "proxy.")) {
                return plan;
            }
        }
        auto &scope = plan._merge_scopes[merge];
        for (auto &block_plan : plan._blocks) {
            auto *source = block_plan.block;
            if (source != merge &&
                dom.contains(const_cast<xir::BasicBlock *>(source)) &&
                dom.dominates(
                    const_cast<xir::BasicBlock *>(header),
                    const_cast<xir::BasicBlock *>(source)) &&
                (merge_dominates_header ||
                 !dom.contains(const_cast<xir::BasicBlock *>(merge)) ||
                 !dom.dominates(
                     const_cast<xir::BasicBlock *>(merge),
                     const_cast<xir::BasicBlock *>(source)))) {
                scope.emplace(source);
            }
        }
    }

    // OpSwitch target operands are semantically unordered, but SPIR-V imposes
    // a physical ordering constraint when one case construct falls through to
    // another: all literals for the source target must be contiguous and the
    // destination target must immediately follow them. Freeze that ordering in
    // the plan. Cycles back to the Switch header are normalized separately as a
    // real structured loop; cycles among case constructs have no equivalent
    // one-edge normalization and are rejected instead of being hidden in the
    // emitter or delegated to spirv-opt.
    for (size_t switch_index = 0u;
         switch_index < plan._switch_regions.size(); ++switch_index) {
        auto &region = plan._switch_regions[switch_index];
        auto *inst = region.instruction;
        auto *header = region.header;
        auto *merge = inst->merge_block();
        auto *default_block = inst->default_block();
        auto &scope = plan._merge_scopes.at(merge);

        // Ordinary dominance can extend through an enclosing loop's continue
        // block and into the next iteration. Those blocks are structured exits
        // of this Switch, not part of a case construct and not evidence of a
        // cycle local to the Switch.
        luisa::unordered_set<const xir::BasicBlock *>
            enclosing_boundary_targets;
        auto construct_encloses_switch = [&](const xir::BasicBlock *owner,
                                             const xir::BasicBlock *construct_merge) noexcept {
            return owner != header &&
                   dom.contains(const_cast<xir::BasicBlock *>(owner)) &&
                   dom.dominates(
                       const_cast<xir::BasicBlock *>(owner),
                       const_cast<xir::BasicBlock *>(header)) &&
                   (construct_merge == header ||
                    !dom.contains(const_cast<xir::BasicBlock *>(construct_merge)) ||
                    !dom.dominates(
                        const_cast<xir::BasicBlock *>(construct_merge),
                        const_cast<xir::BasicBlock *>(header)));
        };
        for (auto [construct_merge, owner] : plan._merge_owners) {
            if (construct_encloses_switch(
                    owner->parent_block(), construct_merge)) {
                enclosing_boundary_targets.emplace(construct_merge);
            }
        }
        for (auto &loop : plan._loop_regions) {
            if (construct_encloses_switch(loop.owner, loop.merge)) {
                enclosing_boundary_targets.emplace(loop.prepare);
                enclosing_boundary_targets.emplace(loop.update);
                enclosing_boundary_targets.emplace(loop.merge);
            }
        }
        for (auto &loop : plan._simple_loop_regions) {
            if (construct_encloses_switch(loop.owner, loop.merge)) {
                enclosing_boundary_targets.emplace(loop.body);
                enclosing_boundary_targets.emplace(loop.merge);
            }
        }
        for (auto &outer_switch : plan._switch_regions) {
            if (construct_encloses_switch(
                    outer_switch.header,
                    outer_switch.instruction->merge_block())) {
                // A branch to an enclosing cyclic Switch header resolves to
                // that Switch's synthetic continue target. Treat the logical
                // header as a boundary even before loop wrapping is finalized,
                // so planning is independent of Switch discovery order.
                enclosing_boundary_targets.emplace(outer_switch.header);
            }
        }

        // Each unique OpSwitch target denotes one case construct. The merge and
        // a cyclic edge to the logical Switch header are physical boundaries,
        // not ordinary case constructs in the normalized graph.
        luisa::vector<const xir::BasicBlock *> construct_entries;
        luisa::unordered_map<const xir::BasicBlock *, size_t>
            construct_entry_indices;
        auto add_construct_entry = [&](const xir::BasicBlock *block) noexcept {
            if (block == merge || block == header ||
                enclosing_boundary_targets.contains(block) ||
                construct_entry_indices.contains(block)) {
                return;
            }
            auto index = construct_entries.size();
            construct_entries.emplace_back(block);
            construct_entry_indices.emplace(block, index);
        };
        for (size_t case_index = 0u; case_index < inst->case_count();
             ++case_index) {
            add_construct_entry(inst->case_block(case_index));
        }
        add_construct_entry(default_block);

        auto construct_owner = [&](const xir::BasicBlock *block) noexcept
            -> const xir::BasicBlock * {
            const xir::BasicBlock *owner = nullptr;
            for (auto *entry : construct_entries) {
                if (dom.contains(const_cast<xir::BasicBlock *>(entry)) &&
                    dom.contains(const_cast<xir::BasicBlock *>(block)) &&
                    dom.dominates(
                        const_cast<xir::BasicBlock *>(entry),
                        const_cast<xir::BasicBlock *>(block))) {
                    if (owner != nullptr && owner != entry) {
                        if (reject_planning_precondition(luisa::format(
                                "SPIR-V control-flow plan rejected Switch case "
                                "block {} dominated by multiple case entries. "
                                "Case constructs must be disjoint before "
                                "codegen.",
                                static_cast<const void *>(block)))) {
                            return nullptr;
                        }
                    }
                    owner = entry;
                }
            }
            return owner;
        };

        luisa::unordered_map<const xir::BasicBlock *,
                             const xir::BasicBlock *>
            fallthrough_targets;
        luisa::unordered_map<const xir::BasicBlock *, size_t>
            fallthrough_indegrees;
        luisa::unordered_set<const xir::BasicBlock *> backedge_sources;
        auto has_nonlocal_exit = false;
        auto header_is_direct_target = default_block == header;
        for (size_t case_index = 0u; case_index < inst->case_count();
             ++case_index) {
            header_is_direct_target |=
                inst->case_block(case_index) == header;
        }
        auto header_is_enclosing_simple_loop_continue =
            plan._simple_loop_body_indices.contains(header);
        auto record_direct_exit_target =
            [&](const xir::BasicBlock *target) noexcept {
                if (target == merge ||
                    (target == header &&
                     !header_is_enclosing_simple_loop_continue) ||
                    !enclosing_boundary_targets.contains(target)) {
                    return;
                }
                region.direct_exit_targets.emplace(target);
                has_nonlocal_exit = true;
            };
        record_direct_exit_target(default_block);
        for (size_t case_index = 0u; case_index < inst->case_count();
             ++case_index) {
            record_direct_exit_target(inst->case_block(case_index));
        }
        if (header_is_direct_target &&
            !header_is_enclosing_simple_loop_continue) {
            backedge_sources.emplace(header);
        }

        for (auto &source_plan : plan._blocks) {
            auto *source = source_plan.block;
            if (source == header || !scope.contains(source)) { continue; }
            if (enclosing_boundary_targets.contains(source)) { continue; }
            auto *owner = construct_owner(source);
            if (!plan._planning_diagnostic.empty()) { return plan; }
            if (owner == nullptr) { continue; }
            for (auto *operand_use : source->terminator()->operand_uses()) {
                auto *value = operand_use->value();
                if (value == nullptr || !value->isa<xir::BasicBlock>()) {
                    continue;
                }
                auto *target =
                    static_cast<const xir::BasicBlock *>(value);
                if (target == header) {
                    if (header_is_enclosing_simple_loop_continue) {
                        has_nonlocal_exit = true;
                    } else {
                        backedge_sources.emplace(source);
                    }
                    continue;
                }
                if (target != merge &&
                    enclosing_boundary_targets.contains(target)) {
                    has_nonlocal_exit = true;
                    continue;
                }
                if (target == merge ||
                    !construct_entry_indices.contains(target)) {
                    if (target != merge && !scope.contains(target)) {
                        has_nonlocal_exit = true;
                    }
                    continue;
                }
                if (target == owner) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected a cyclic Switch "
                            "case construct targeting itself. Only cycles "
                            "through the Switch header can be normalized as a "
                            "structured loop.")) {
                        return plan;
                    }
                }
                auto [iter, inserted] =
                    fallthrough_targets.emplace(owner, target);
                if (!inserted && iter->second != target) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected a Switch case "
                            "construct that falls through to multiple case "
                            "targets.")) {
                        return plan;
                    }
                }
                if (inserted) { fallthrough_indegrees[target]++; }
            }
        }
        for (auto [target, indegree] : fallthrough_indegrees) {
            static_cast<void>(target);
            if (indegree > 1u) {
                if (reject_planning_precondition(luisa::format(
                        "SPIR-V control-flow plan rejected a Switch case target "
                        "with {} fallthrough predecessors; SPIR-V permits at "
                        "most one.",
                        indegree))) {
                    return plan;
                }
            }
        }

        // Group all literals sharing a target before applying adjacency. This
        // is both deterministic and matches the validator's interpretation of
        // duplicate target operands as one case construct.
        luisa::vector<const xir::BasicBlock *> operand_group_targets;
        luisa::vector<luisa::vector<size_t>> operand_groups;
        luisa::unordered_map<const xir::BasicBlock *, size_t>
            operand_group_indices;
        for (size_t case_index = 0u; case_index < inst->case_count();
             ++case_index) {
            auto *target = inst->case_block(case_index);
            auto [iter, inserted] = operand_group_indices.emplace(
                target, operand_group_targets.size());
            if (inserted) {
                operand_group_targets.emplace_back(target);
                operand_groups.emplace_back();
            }
            operand_groups[iter->second].emplace_back(case_index);
        }

        auto group_count = operand_groups.size();
        auto invalid_group = group_count;
        luisa::vector<size_t> next_groups(group_count, invalid_group);
        luisa::vector<size_t> group_indegrees(group_count, 0u);
        for (size_t source_group = 0u; source_group < group_count;
             ++source_group) {
            auto *source_target = operand_group_targets[source_group];
            auto fallthrough_iter =
                fallthrough_targets.find(source_target);
            if (fallthrough_iter == fallthrough_targets.end()) { continue; }
            auto *destination = fallthrough_iter->second;

            // SPIRV-Tools treats a default-only block as transparent for target
            // ordering: A -> default -> B requires A immediately before B,
            // while A -> default with no further case fallthrough imposes no
            // operand-order constraint.
            if (destination == default_block &&
                !operand_group_indices.contains(default_block)) {
                auto default_fallthrough =
                    fallthrough_targets.find(default_block);
                if (default_fallthrough == fallthrough_targets.end()) {
                    continue;
                }
                destination = default_fallthrough->second;
            }
            auto destination_iter =
                operand_group_indices.find(destination);
            if (destination_iter == operand_group_indices.end()) { continue; }
            auto destination_group = destination_iter->second;
            if (destination_group == source_group) {
                if (reject_planning_precondition(
                        "SPIR-V control-flow plan rejected a cyclic Switch "
                        "case fallthrough chain.")) {
                    return plan;
                }
            }
            next_groups[source_group] = destination_group;
            if (++group_indegrees[destination_group] > 1u) {
                if (reject_planning_precondition(
                        "SPIR-V control-flow plan rejected a Switch case "
                        "operand group with multiple required immediate "
                        "predecessors.")) {
                    return plan;
                }
            }
        }

        region.case_operand_order.clear();
        region.case_operand_order.reserve(inst->case_count());
        luisa::vector<uint8_t> ordered_groups(group_count, 0u);
        auto append_chain = [&](size_t root) noexcept {
            auto group = root;
            while (group != invalid_group) {
                if (ordered_groups[group] != 0u) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected a cyclic Switch "
                            "case fallthrough chain.")) {
                        return;
                    }
                }
                ordered_groups[group] = 1u;
                for (auto case_index : operand_groups[group]) {
                    region.case_operand_order.emplace_back(case_index);
                }
                group = next_groups[group];
            }
        };
        for (size_t group = 0u; group < group_count; ++group) {
            if (group_indegrees[group] == 0u) { append_chain(group); }
            if (!plan._planning_diagnostic.empty()) { return plan; }
        }
        for (auto ordered : ordered_groups) {
            if (ordered == 0u) {
                if (reject_planning_precondition(
                        "SPIR-V control-flow plan rejected a cyclic Switch "
                        "case fallthrough chain.")) {
                    return plan;
                }
            }
        }
        LUISA_ASSERT(region.case_operand_order.size() == inst->case_count(),
                     "SPIR-V Switch case ordering lost an operand.");

        if (!backedge_sources.empty()) {
            if (has_nonlocal_exit) {
                if (reject_planning_precondition(
                        "SPIR-V control-flow plan rejected a cyclic Switch with "
                        "a nonlocal branch exit. The synthetic loop can "
                        "represent case fallthrough, the Switch merge, and "
                        "backedges to the Switch header, but cannot change the "
                        "target of an outer break/continue without an explicit "
                        "exit-dispatch value.")) {
                    return plan;
                }
            }
            region.loop_wrapped = true;
            region.dispatch_synthetic_index = plan._add_synthetic(
                SyntheticBlockKind::SWITCH_DISPATCH, inst, Target{});
            region.continue_synthetic_index = plan._add_synthetic(
                SyntheticBlockKind::SWITCH_CONTINUE, inst,
                Target::xir(header));
            region.has_header_case_target = header_is_direct_target;
            if (header_is_direct_target) {
                region.header_case_synthetic_index = plan._add_synthetic(
                    SyntheticBlockKind::EDGE_TRAMPOLINE, inst,
                    Target::synthetic(region.continue_synthetic_index));
            }
            auto [_, inserted] =
                plan._wrapped_switch_header_indices.emplace(
                    header, switch_index);
            LUISA_ASSERT(inserted,
                         "SPIR-V Switch header was loop-wrapped more than once.");
            auto [sources_iter, sources_inserted] =
                plan._wrapped_switch_backedge_sources.emplace(
                    header, std::move(backedge_sources));
            static_cast<void>(sources_iter);
            LUISA_ASSERT(
                sources_inserted,
                "SPIR-V wrapped Switch backedges were frozen more than once.");
        }
    }

    // Second pass: give each construct an exact physical merge role. If an XIR
    // merge name is also a loop entry/continue role, a dedicated synthetic
    // merge proxy preserves both SPIR-V roles without moving the XIR block's
    // instructions. Edge resolution below is source-sensitive; the proxy is
    // never allowed to steal the other role's backedge or continue edges.
    auto bind_construct_merge = [&](const xir::Instruction *owner,
                                    const xir::BasicBlock *merge) noexcept {
        auto continuation = plan._resolve_loop_boundary_target(merge);
        auto wrapped_header_iter =
            plan._wrapped_switch_header_indices.find(merge);
        auto merge_is_wrapped_switch_header =
            wrapped_header_iter !=
            plan._wrapped_switch_header_indices.end();
        if (!merge_is_wrapped_switch_header &&
            plan.block(merge).has_role(BlockRole::SWITCH_HEADER) &&
            dom.contains(const_cast<xir::BasicBlock *>(merge)) &&
            dom.dominates(
                const_cast<xir::BasicBlock *>(merge),
                const_cast<xir::BasicBlock *>(owner->parent_block()))) {
            if (reject_planning_precondition(
                    "SPIR-V control-flow plan rejected a backward construct "
                    "merge to a noncyclic Switch header. The boundary has no "
                    "planned loop continue edge.")) {
                return Target{};
            }
        }
        if (merge_is_wrapped_switch_header) {
            auto &wrapped_region = plan._switch_regions.at(
                wrapped_header_iter->second);
            auto &wrapped_scope = plan._merge_scopes.at(
                wrapped_region.instruction->merge_block());
            // A construct nested in the cyclic Switch exits through the
            // synthetic continue boundary. An enclosing construct whose merge
            // starts the Switch instead converges to the logical header. Both
            // cases require a proxy so the physical loop header still has one
            // non-backedge predecessor.
            if (wrapped_scope.contains(owner->parent_block())) {
                continuation = Target::synthetic(
                    wrapped_region.continue_synthetic_index);
            }
        }
        auto target = continuation;
        auto &scope = plan._merge_scopes.at(merge);
        auto has_bypass_predecessor = false;
        for (auto &source_plan : plan._blocks) {
            auto *source = source_plan.block;
            for (auto *operand_use : source->terminator()->operand_uses()) {
                if (operand_use->value() == merge && !scope.contains(source)) {
                    has_bypass_predecessor = true;
                    break;
                }
            }
            if (has_bypass_predecessor) { break; }
        }
        // A merge proxy is required not only for overlapping loop roles, but
        // whenever the logical merge has a predecessor outside this construct.
        // Construct-owned exits converge at the proxy; bypass edges continue to
        // the logical block. Thus OpSelectionMerge/OpLoopMerge never claims a
        // block the header does not dominate.
        if (has_bypass_predecessor ||
            plan._loop_prepare_indices.contains(merge) ||
            plan._loop_update_indices.contains(merge) ||
            plan._simple_loop_body_indices.contains(merge) ||
            merge_is_wrapped_switch_header) {
            target = Target::synthetic(
                plan._add_synthetic(SyntheticBlockKind::EDGE_TRAMPOLINE, owner, continuation));
        }
        auto [_, inserted] = plan._merge_targets.emplace(merge, target);
        LUISA_ASSERT(inserted, "SPIR-V control-flow plan bound a construct merge more than once.");
        return target;
    };
    for (auto &region : plan._if_regions) {
        if (!region.emit_selection_merge) { continue; }
        region.merge_target = bind_construct_merge(region.instruction, region.merge_target.xir_block);
        if (!plan._planning_diagnostic.empty()) { return plan; }
    }
    for (auto &region : plan._loop_regions) {
        region.merge_target = bind_construct_merge(region.instruction, region.merge);
        if (!plan._planning_diagnostic.empty()) { return plan; }
    }
    for (auto &region : plan._simple_loop_regions) {
        region.merge_target = bind_construct_merge(region.instruction, region.merge);
        if (!plan._planning_diagnostic.empty()) { return plan; }
    }
    for (auto &region : plan._switch_regions) {
        if (!region.loop_wrapped) {
            region.merge_target = bind_construct_merge(
                region.instruction, region.instruction->merge_block());
            if (!plan._planning_diagnostic.empty()) { return plan; }
            continue;
        }

        // The loop and its nested selection require distinct physical merge
        // roles. A fixed pair of one-way blocks gives the selection exits the
        // chain selection-merge -> loop-merge -> logical-merge, and makes Phi
        // forwarding explicit in the immutable plan.
        auto *merge = region.instruction->merge_block();
        auto continuation = plan._resolve_loop_boundary_target(merge);
        auto loop_merge_index = plan._add_synthetic(
            SyntheticBlockKind::EDGE_TRAMPOLINE,
            region.instruction, continuation);
        auto selection_merge_index = plan._add_synthetic(
            SyntheticBlockKind::EDGE_TRAMPOLINE,
            region.instruction, Target::synthetic(loop_merge_index));
        region.loop_merge_target = Target::synthetic(loop_merge_index);
        region.merge_target = Target::synthetic(selection_merge_index);
        auto [_, inserted] =
            plan._merge_targets.emplace(merge, region.merge_target);
        LUISA_ASSERT(inserted,
                     "SPIR-V wrapped Switch merge was bound more than once.");
    }

    // XIR can express a nested selection exit with two adjacent logical merge
    // blocks in payload order:
    //
    //   inner arm [-> forwarding blocks] -> outer merge A -> inner merge B
    //
    // Emitting A as the outer physical merge and B as the inner physical merge
    // would leave the inner construct through A and then branch back into it
    // through B, which SPIR-V forbids. Preserve the optional one-edge arm
    // forwarding blocks, both merge blocks, and their payload order, but
    // rotate the physical merge roles: A becomes the inner merge and B becomes
    // the outer merge, yielding the properly nested path A -> B.
    struct NestedSelectionMergeCandidate {
        const xir::Instruction *outer_instruction;
        const xir::Instruction *inner_instruction;
        const xir::BasicBlock *outer_merge;
        const xir::BasicBlock *inner_merge;
        const xir::BasicBlock *inner_entry_predecessor;
    };
    auto is_ordinary_selection = [&](const xir::Instruction *instruction) noexcept {
        if (instruction->isa<xir::IfInst>()) {
            auto *if_inst =
                static_cast<const xir::IfInst *>(instruction);
            return plan._if_regions.at(
                                       plan._if_indices.at(if_inst))
                .emit_selection_merge;
        }
        if (instruction->isa<xir::SwitchInst>()) {
            auto *switch_inst = static_cast<const xir::SwitchInst *>(instruction);
            return !plan._switch_regions.at(
                                            plan._switch_indices.at(switch_inst))
                        .loop_wrapped;
        }
        return false;
    };
    auto find_selection_entry_predecessor =
        [&](const xir::Instruction *instruction,
            const xir::BasicBlock *block) noexcept {
            auto *predecessor =
                static_cast<const xir::BasicBlock *>(nullptr);
            auto inspect_entry =
                [&](const xir::BasicBlock *entry) noexcept {
                    auto *candidate =
                        static_cast<const xir::BasicBlock *>(nullptr);
                    if (entry == block) {
                        candidate = instruction->parent_block();
                    } else {
                        luisa::unordered_set<const xir::BasicBlock *> visited;
                        auto *current = entry;
                        while (visited.emplace(current).second &&
                               current->terminator()->isa<xir::BranchInst>()) {
                            auto *branch =
                                static_cast<const xir::BranchInst *>(
                                    current->terminator());
                            if (branch->target_block() == block) {
                                candidate = current;
                                break;
                            }
                            current = branch->target_block();
                        }
                    }
                    if (candidate == nullptr) { return; }
                    if (predecessor == nullptr) {
                        predecessor = candidate;
                    }
                };
            if (instruction->isa<xir::IfInst>()) {
                auto *if_inst =
                    static_cast<const xir::IfInst *>(instruction);
                inspect_entry(if_inst->true_block());
                inspect_entry(if_inst->false_block());
            } else {
                LUISA_ASSERT(
                    instruction->isa<xir::SwitchInst>(),
                    "SPIR-V nested selection rotation received a non-selection owner.");
                auto *switch_inst =
                    static_cast<const xir::SwitchInst *>(instruction);
                inspect_entry(switch_inst->default_block());
                for (size_t i = 0u;
                     i < switch_inst->case_count(); ++i) {
                    inspect_entry(switch_inst->case_block(i));
                }
            }
            // Return any matching predecessor here. The exact predecessor-set
            // check below is authoritative and rejects distinct arm chains
            // that enter the merge instead of silently skipping rotation.
            return predecessor;
        };
    luisa::vector<NestedSelectionMergeCandidate> rotation_candidates;
    for (auto [outer_merge, outer_instruction] : plan._merge_owners) {
        if (!is_ordinary_selection(outer_instruction) ||
            !outer_merge->terminator()->isa<xir::BranchInst>()) {
            continue;
        }
        auto *branch = static_cast<const xir::BranchInst *>(
            outer_merge->terminator());
        auto *inner_merge = branch->target_block();
        auto inner_owner_iter = plan._merge_owners.find(inner_merge);
        if (inner_owner_iter == plan._merge_owners.end() ||
            !is_ordinary_selection(inner_owner_iter->second)) {
            continue;
        }
        auto *inner_instruction = inner_owner_iter->second;
        auto *inner_header = inner_instruction->parent_block();
        auto *inner_entry_predecessor =
            find_selection_entry_predecessor(
                inner_instruction, outer_merge);
        if (inner_entry_predecessor == nullptr ||
            !plan._merge_scopes.at(outer_merge).contains(inner_header) ||
            !plan._merge_scopes.at(inner_merge).contains(outer_merge)) {
            continue;
        }

        // Rotation is exact only when the final inner-arm edge is the logical
        // merge's sole predecessor. Otherwise an ordinary outer exit would
        // have to execute A's payload before reaching B, which cannot be
        // represented by exchanging the two physical declarations alone.
        luisa::unordered_set<const xir::BasicBlock *> predecessors_of_outer_merge;
        for (auto &source_plan : plan._blocks) {
            for (auto *operand_use :
                 source_plan.block->terminator()->operand_uses()) {
                if (operand_use->value() == outer_merge) {
                    predecessors_of_outer_merge.emplace(source_plan.block);
                }
            }
        }
        if (predecessors_of_outer_merge.size() != 1u ||
            !predecessors_of_outer_merge.contains(
                inner_entry_predecessor)) {
            if (reject_planning_precondition(
                    "SPIR-V control-flow plan cannot rotate a nested "
                    "selection merge with additional logical predecessors. "
                    "restructure_cfg must split the shared arm/merge block.")) {
                return plan;
            }
        }
        if (plan._merge_targets.at(outer_merge) != Target::xir(outer_merge) ||
            plan._merge_targets.at(inner_merge) != Target::xir(inner_merge)) {
            if (reject_planning_precondition(
                    "SPIR-V control-flow plan cannot rotate nested selection "
                    "merges that already require physical merge proxies.")) {
                return plan;
            }
        }
        rotation_candidates.emplace_back(NestedSelectionMergeCandidate{
            .outer_instruction = outer_instruction,
            .inner_instruction = inner_instruction,
            .outer_merge = outer_merge,
            .inner_merge = inner_merge,
            .inner_entry_predecessor =
                inner_entry_predecessor});
    }
    std::sort(
        rotation_candidates.begin(), rotation_candidates.end(),
        [&](auto &&lhs, auto &&rhs) noexcept {
            return plan._block_indices.at(lhs.outer_merge) <
                   plan._block_indices.at(rhs.outer_merge);
        });
    luisa::unordered_set<const xir::BasicBlock *> rotated_merge_blocks;
    for (auto &&candidate : rotation_candidates) {
        if (rotated_merge_blocks.contains(candidate.outer_merge) ||
            rotated_merge_blocks.contains(candidate.inner_merge)) {
            if (reject_planning_precondition(
                    "SPIR-V control-flow plan rejected an overlapping nested "
                    "selection merge-rotation chain. restructure_cfg must "
                    "split the shared merge roles.")) {
                return plan;
            }
        }
        rotated_merge_blocks.emplace(candidate.outer_merge);
        rotated_merge_blocks.emplace(candidate.inner_merge);
        auto outer_target = Target::xir(candidate.inner_merge);
        auto inner_target = Target::xir(candidate.outer_merge);
        plan._merge_targets[candidate.outer_merge] = outer_target;
        plan._merge_targets[candidate.inner_merge] = inner_target;
        auto rotation_index =
            plan._nested_selection_merge_rotations.size();
        plan._nested_selection_merge_rotations.emplace_back(
            NestedSelectionMergeRotation{
                .outer_instruction = candidate.outer_instruction,
                .inner_instruction = candidate.inner_instruction,
                .outer_logical_merge = candidate.outer_merge,
                .inner_logical_merge = candidate.inner_merge,
                .outer_physical_merge = outer_target,
                .inner_physical_merge = inner_target});
        auto [inner_iter, inner_inserted] =
            plan._nested_selection_rotation_inner_indices.emplace(
                candidate.inner_instruction, rotation_index);
        static_cast<void>(inner_iter);
        LUISA_ASSERT(inner_inserted,
                     "SPIR-V nested selection received multiple merge rotations.");
        if (candidate.inner_entry_predecessor !=
            candidate.inner_instruction->parent_block()) {
            auto *terminator =
                candidate.inner_entry_predecessor->terminator();
            LUISA_ASSERT(
                terminator->isa<xir::BranchInst>() &&
                    static_cast<const xir::BranchInst *>(terminator)
                            ->target_block() ==
                        candidate.outer_merge,
                "SPIR-V nested selection rotation received an invalid "
                "entry forwarding edge.");
            auto [entry_iter, entry_inserted] =
                plan._nested_selection_rotation_entry_edge_targets.emplace(
                    terminator, inner_target);
            static_cast<void>(entry_iter);
            LUISA_ASSERT(
                entry_inserted,
                "SPIR-V nested selection entry edge was rotated twice.");
        }
        auto [forward_iter, forward_inserted] =
            plan._nested_selection_merge_forward_targets.emplace(
                candidate.outer_merge, candidate.inner_merge);
        static_cast<void>(forward_iter);
        LUISA_ASSERT(forward_inserted,
                     "SPIR-V nested selection merge forwarding was rotated twice.");
    }
    // The ordinary region records were initialized before rotations were
    // known. Refresh them from the final immutable merge-role table.
    for (auto &region : plan._if_regions) {
        if (region.emit_selection_merge) {
            region.merge_target =
                plan._merge_targets.at(
                    region.instruction->merge_block());
        }
    }
    for (auto &region : plan._switch_regions) {
        if (!region.loop_wrapped) {
            region.merge_target = plan._merge_targets.at(
                region.instruction->merge_block());
        }
    }
    for (auto &region : plan._loop_regions) {
        region.entry_target = plan._resolve_ordinary_target(
            region.owner, region.prepare);
        region.body_target = plan._resolve_ordinary_target(
            region.prepare, region.body);
        // OpLoopMerge's continue target owns the loop role itself. Selection
        // exits that happen to name the same XIR block are routed separately
        // by their source edge and must not replace this target globally.
        region.continue_target =
            plan._resolve_loop_boundary_target(region.update);
    }

    auto resolve_selection_operand =
        [&](const xir::Instruction *instruction,
            const xir::BasicBlock *source,
            const xir::BasicBlock *logical_target,
            const xir::BasicBlock *logical_merge,
            Target physical_merge) noexcept {
            if (logical_target == logical_merge) { return physical_merge; }
            if (auto rotation_iter =
                    plan._nested_selection_rotation_inner_indices.find(
                        instruction);
                rotation_iter !=
                plan._nested_selection_rotation_inner_indices.end()) {
                auto &rotation = plan._nested_selection_merge_rotations.at(
                    rotation_iter->second);
                if (logical_target == rotation.outer_logical_merge) {
                    return rotation.inner_physical_merge;
                }
            }
            return plan._resolve_ordinary_target(source, logical_target);
        };
    auto is_enclosing_boundary_target =
        [&](const xir::Instruction *instruction,
            const xir::BasicBlock *target) noexcept {
            auto *header = instruction->parent_block();
            if (auto owner_iter = plan._merge_owners.find(target);
                owner_iter != plan._merge_owners.end() &&
                owner_iter->second != instruction &&
                plan._merge_scopes.at(target).contains(header)) {
                return true;
            }
            auto construct_encloses_header =
                [&](const xir::BasicBlock *owner,
                    const xir::BasicBlock *merge) noexcept {
                    return owner != header &&
                           dom.contains(
                               const_cast<xir::BasicBlock *>(owner)) &&
                           dom.dominates(
                               const_cast<xir::BasicBlock *>(owner),
                               const_cast<xir::BasicBlock *>(header)) &&
                           (merge == header ||
                            !dom.contains(
                                const_cast<xir::BasicBlock *>(merge)) ||
                            !dom.dominates(
                                const_cast<xir::BasicBlock *>(merge),
                                const_cast<xir::BasicBlock *>(header)));
                };
            for (auto &loop : plan._loop_regions) {
                if ((target == loop.prepare || target == loop.update) &&
                    construct_encloses_header(loop.owner, loop.merge)) {
                    return true;
                }
            }
            for (auto &loop : plan._simple_loop_regions) {
                if (target == loop.body &&
                    construct_encloses_header(loop.owner, loop.merge)) {
                    return true;
                }
            }
            for (auto &outer_switch : plan._switch_regions) {
                if (target == outer_switch.header &&
                    outer_switch.instruction != instruction &&
                    construct_encloses_header(
                        outer_switch.header,
                        outer_switch.instruction->merge_block())) {
                    return true;
                }
            }
            return false;
        };

    // SPIR-V permits a conditional to branch directly out of an enclosing
    // construct without declaring a nested OpSelectionMerge. This is the
    // residual case in LLVM SPIRVStructurizer::
    // addHeaderToRemainingDivergentDAG: after empty forwarding paths have been
    // contracted, at most one distinct successor is an ordinary block; every
    // other successor is already an enclosing merge or continue role. A child
    // construct header is deliberately ordinary: it structures its own body,
    // not the conditional edge that chose whether to enter it.
    //
    // Third pass: resolve every executable edge against the frozen role table.
    for (auto &block_plan : plan._blocks) {
        auto *block = block_plan.block;
        auto *terminator = block->terminator();
        switch (terminator->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                auto *inst = static_cast<const xir::IfInst *>(terminator);
                auto &region = plan._if_regions[plan._if_indices.at(inst)];
                luisa::vector<std::pair<Target, Target>> exit_trampolines;
                auto resolve_if_operand =
                    [&](const xir::BasicBlock *logical_target) noexcept {
                        if (!region.emit_selection_merge &&
                            logical_target !=
                                inst->merge_block()) {
                            LUISA_ASSERT(
                                region.loop_boundary_exit_target !=
                                    nullptr,
                                "SPIR-V loop-boundary If lost its "
                                "planned exit target.");
                            return plan._resolve_ordinary_target(
                                block,
                                region.loop_boundary_exit_target);
                        }
                        auto target = resolve_selection_operand(
                            inst, block, logical_target,
                            inst->merge_block(), region.merge_target);
                        if (target == region.merge_target ||
                            !is_enclosing_boundary_target(
                                inst, logical_target)) {
                            return target;
                        }
                        for (auto [continuation, trampoline] :
                             exit_trampolines) {
                            if (continuation == target) {
                                return trampoline;
                            }
                        }
                        auto trampoline = Target::synthetic(
                            plan._add_synthetic(
                                SyntheticBlockKind::EDGE_TRAMPOLINE,
                                inst, target));
                        exit_trampolines.emplace_back(
                            target, trampoline);
                        return trampoline;
                    };
                region.true_target =
                    resolve_if_operand(inst->true_block());
                region.false_target =
                    resolve_if_operand(inst->false_block());
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto *inst = static_cast<const xir::SwitchInst *>(terminator);
                auto &region = plan._switch_regions[plan._switch_indices.at(inst)];
                luisa::vector<std::pair<Target, Target>> exit_trampolines;
                auto resolve_switch_operand =
                    [&](const xir::BasicBlock *logical_target) noexcept {
                        auto target = resolve_selection_operand(
                            inst, block, logical_target,
                            inst->merge_block(), region.merge_target);
                        if (!region.direct_exit_targets.contains(
                                logical_target)) {
                            return target;
                        }
                        for (auto [continuation, trampoline] :
                             exit_trampolines) {
                            if (continuation == target) { return trampoline; }
                        }
                        auto trampoline = Target::synthetic(
                            plan._add_synthetic(
                                SyntheticBlockKind::EDGE_TRAMPOLINE,
                                inst, target));
                        exit_trampolines.emplace_back(target, trampoline);
                        return trampoline;
                    };
                region.default_target = resolve_switch_operand(
                    inst->default_block());
                for (size_t case_index = 0u; case_index < inst->case_count(); ++case_index) {
                    region.case_targets[case_index] = resolve_switch_operand(
                        inst->case_block(case_index));
                }
                break;
            }
            case xir::DerivedInstructionTag::BRANCH: {
                auto *inst = static_cast<const xir::BranchInst *>(terminator);
                auto target = [&]() noexcept {
                    if (auto entry_iter =
                            plan._nested_selection_rotation_entry_edge_targets
                                .find(inst);
                        entry_iter !=
                        plan._nested_selection_rotation_entry_edge_targets
                            .end()) {
                        return entry_iter->second;
                    }
                    auto forward_iter =
                        plan._nested_selection_merge_forward_targets.find(
                            block);
                    if (forward_iter !=
                            plan._nested_selection_merge_forward_targets
                                .end() &&
                        forward_iter->second ==
                            inst->target_block()) {
                        return Target::xir(inst->target_block());
                    }
                    return plan._resolve_ordinary_target(
                        block, inst->target_block());
                }();
                if (auto loop_iter = plan._loop_prepare_indices.find(block);
                    loop_iter != plan._loop_prepare_indices.end()) {
                    auto &&loop = plan._loop_regions[loop_iter->second];
                    LUISA_ASSERT(
                        loop.prepare_kind ==
                                SpirvLoopPrepareKind::UNCONDITIONAL &&
                            inst->target_block() == loop.body &&
                            target == loop.body_target,
                        "SPIR-V control-flow plan lost the canonical "
                        "unconditional Loop.prepare edge.");
                }
                plan._edge_targets.emplace(inst, target);
                break;
            }
            case xir::DerivedInstructionTag::BREAK: {
                auto *inst = static_cast<const xir::BreakInst *>(terminator);
                LUISA_ASSERT(plan._merge_owners.contains(inst->target_block()),
                             "SPIR-V control-flow plan rejected Break whose target is not a construct merge.");
                plan._edge_targets.emplace(
                    inst, plan._resolve_ordinary_target(
                              block, inst->target_block()));
                break;
            }
            case xir::DerivedInstructionTag::CONTINUE: {
                auto *inst = static_cast<const xir::ContinueInst *>(terminator);
                auto target = inst->target_block();
                LUISA_ASSERT(plan._loop_update_indices.contains(target) ||
                                 plan._simple_loop_body_indices.contains(target),
                             "SPIR-V control-flow plan rejected Continue whose target is not a loop continue role.");
                plan._edge_targets.emplace(
                    inst, plan._resolve_ordinary_target(block, target));
                break;
            }
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto *inst = static_cast<const xir::ConditionalBranchInst *>(terminator);
                auto loop_iter = plan._loop_prepare_indices.find(block);
                if (loop_iter != plan._loop_prepare_indices.end()) {
                    auto &loop = plan._loop_regions[loop_iter->second];
                    LUISA_ASSERT(loop.prepare_kind ==
                                         SpirvLoopPrepareKind::CONDITIONAL &&
                                     inst->condition() != nullptr && inst->condition()->type() == Type::of<bool>() &&
                                     inst->true_block() == loop.body && inst->false_block() == loop.merge,
                                 "SPIR-V control-flow plan rejected noncanonical Loop.prepare ConditionalBranch; "
                                 "expected true=body and false=merge.");
                    plan._conditional_branch_targets.emplace(
                        inst,
                        std::array{loop.body_target,
                                   loop.merge_target});
                    break;
                }

                auto targets = std::array{
                    inst->true_block(), inst->false_block()};
                auto boundary_count = size_t{0u};
                luisa::unordered_set<const xir::BasicBlock *>
                    ordinary_targets;
                for (auto *target : targets) {
                    if (is_enclosing_boundary_target(inst, target)) {
                        ++boundary_count;
                    } else {
                        ordinary_targets.emplace(target);
                    }
                }
                if (boundary_count == 0u ||
                    ordinary_targets.size() > 1u) {
                    if (reject_planning_precondition(
                            "SPIR-V control-flow plan rejected raw "
                            "ConditionalBranch that is neither a canonical "
                            "Loop.prepare nor a direct parent-construct "
                            "boundary guard. restructure_cfg must convert "
                            "ordinary divergence to IfInst and contract empty "
                            "boundary forwarding paths.")) {
                        return plan;
                    }
                }
                plan._conditional_branch_targets.emplace(
                    inst,
                    std::array{
                        plan._resolve_ordinary_target(
                            block, targets[0u]),
                        plan._resolve_ordinary_target(
                            block, targets[1u])});
                break;
            }
            default: break;
        }
    }

    // Freeze the resolved physical graph, including synthetic blocks. This is
    // the graph SPIR-V will actually see, so loop entry/backedge legality must
    // be proved here rather than inferred from logical XIR predecessors.
    auto visit_physical_targets = [&](const BlockPlan &source_plan,
                                      auto &&visit) noexcept {
        if (source_plan.physically_pruned) { return; }
        auto *terminator = source_plan.block->terminator();
        switch (terminator->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                auto *inst = static_cast<const xir::IfInst *>(terminator);
                auto &region = plan._if_regions.at(plan._if_indices.at(inst));
                visit(region.true_target);
                visit(region.false_target);
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto *inst = static_cast<const xir::SwitchInst *>(terminator);
                auto &region = plan._switch_regions.at(plan._switch_indices.at(inst));
                if (region.loop_wrapped) {
                    visit(Target::synthetic(
                        region.dispatch_synthetic_index));
                    break;
                }
                visit(region.default_target);
                for (auto target : region.case_targets) { visit(target); }
                break;
            }
            case xir::DerivedInstructionTag::LOOP: {
                auto *inst = static_cast<const xir::LoopInst *>(terminator);
                visit(plan._loop_regions.at(plan._loop_indices.at(inst)).entry_target);
                break;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                auto *inst = static_cast<const xir::SimpleLoopInst *>(terminator);
                auto &region = plan._simple_loop_regions.at(
                    plan._simple_loop_indices.at(inst));
                visit(Target::synthetic(region.header_synthetic_index));
                break;
            }
            case xir::DerivedInstructionTag::BRANCH:
            case xir::DerivedInstructionTag::BREAK:
            case xir::DerivedInstructionTag::CONTINUE:
                visit(plan._edge_targets.at(terminator));
                break;
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto *inst = static_cast<
                    const xir::ConditionalBranchInst *>(terminator);
                auto iter = plan._conditional_branch_targets.find(inst);
                LUISA_ASSERT(
                    iter != plan._conditional_branch_targets.end(),
                    "SPIR-V physical target enumeration found an "
                    "unplanned ConditionalBranch.");
                visit(iter->second[0u]);
                visit(iter->second[1u]);
                break;
            }
            default: break;
        }
    };
    auto xir_block_count = plan._blocks.size();
    auto physical_block_count =
        xir_block_count + plan._synthetic_blocks.size();
    auto physical_index = [&](Target target) noexcept {
        if (target.kind == Target::Kind::XIR_BLOCK) {
            auto iter = plan._block_indices.find(target.xir_block);
            LUISA_ASSERT(iter != plan._block_indices.end(),
                         "SPIR-V physical target references an unplanned XIR block.");
            return iter->second;
        }
        LUISA_ASSERT(target.synthetic_index < plan._synthetic_blocks.size(),
                     "SPIR-V physical target references an invalid synthetic block.");
        return xir_block_count + target.synthetic_index;
    };
    luisa::vector<luisa::vector<size_t>> physical_successors(
        physical_block_count);
    luisa::vector<luisa::vector<size_t>> physical_predecessors(
        physical_block_count);
    auto add_physical_edge = [&](size_t source, Target target) noexcept {
        auto destination = physical_index(target);
        auto &outgoing = physical_successors[source];
        if (std::find(outgoing.begin(), outgoing.end(), destination) ==
            outgoing.end()) {
            outgoing.emplace_back(destination);
            physical_predecessors[destination].emplace_back(source);
        }
    };
    for (size_t source = 0u; source < xir_block_count; ++source) {
        visit_physical_targets(
            plan._blocks[source],
            [&](Target target) noexcept {
                add_physical_edge(source, target);
            });
    }
    for (size_t synthetic_index = 0u;
         synthetic_index < plan._synthetic_blocks.size();
         ++synthetic_index) {
        auto &synthetic = plan._synthetic_blocks[synthetic_index];
        if (synthetic.kind == SyntheticBlockKind::SWITCH_DISPATCH) {
            LUISA_ASSERT(synthetic.owner != nullptr &&
                             synthetic.owner->isa<xir::SwitchInst>(),
                         "SPIR-V Switch dispatch has an invalid owner.");
            auto *inst = static_cast<const xir::SwitchInst *>(
                synthetic.owner);
            auto &region = plan._switch_regions.at(
                plan._switch_indices.at(inst));
            LUISA_ASSERT(region.loop_wrapped &&
                             region.dispatch_synthetic_index ==
                                 synthetic_index,
                         "SPIR-V Switch dispatch disagrees with its region.");
            auto source = xir_block_count + synthetic_index;
            add_physical_edge(source, region.default_target);
            for (auto target : region.case_targets) {
                add_physical_edge(source, target);
            }
            continue;
        }
        add_physical_edge(xir_block_count + synthetic_index,
                          synthetic.continuation);
    }

    luisa::vector<uint8_t> physical_reachable(
        physical_block_count, 0u);
    luisa::vector<size_t> physical_worklist;
    auto physical_entry =
        plan._block_indices.at(function->body_block());
    physical_worklist.emplace_back(physical_entry);
    while (!physical_worklist.empty()) {
        auto source = physical_worklist.back();
        physical_worklist.pop_back();
        if (physical_reachable[source] != 0u) { continue; }
        physical_reachable[source] = 1u;
        for (auto destination : physical_successors[source]) {
            physical_worklist.emplace_back(destination);
        }
    }
    luisa::vector<luisa::vector<uint8_t>> physical_dominators(
        physical_block_count,
        luisa::vector<uint8_t>(physical_block_count, 0u));
    for (size_t block = 0u; block < physical_block_count; ++block) {
        if (physical_reachable[block] == 0u || block == physical_entry) {
            continue;
        }
        for (size_t candidate = 0u;
             candidate < physical_block_count; ++candidate) {
            physical_dominators[block][candidate] =
                physical_reachable[candidate];
        }
    }
    physical_dominators[physical_entry][physical_entry] = 1u;
    for (;;) {
        auto changed = false;
        for (size_t block = 0u; block < physical_block_count; ++block) {
            if (block == physical_entry ||
                physical_reachable[block] == 0u) {
                continue;
            }
            luisa::vector<uint8_t> next(physical_block_count, 1u);
            auto has_reachable_predecessor = false;
            for (auto predecessor : physical_predecessors[block]) {
                if (physical_reachable[predecessor] == 0u) { continue; }
                if (!has_reachable_predecessor) {
                    next = physical_dominators[predecessor];
                    has_reachable_predecessor = true;
                } else {
                    for (size_t candidate = 0u;
                         candidate < physical_block_count; ++candidate) {
                        next[candidate] &=
                            physical_dominators[predecessor][candidate];
                    }
                }
            }
            LUISA_ASSERT(has_reachable_predecessor,
                         "SPIR-V reachable physical block has no reachable predecessor.");
            next[block] = 1u;
            if (next != physical_dominators[block]) {
                physical_dominators[block] = std::move(next);
                changed = true;
            }
        }
        if (!changed) { break; }
    }

    // SPIR-V defines a backedge structurally: for every reachable physical
    // edge S -> H, H dominates S. Such an edge is legal only when H is an
    // actual physical loop header carrying OpLoopMerge. Auditing only the
    // predecessors of declared loops is incomplete: an ordinary cycle can
    // target a selection/loop merge (or any other block) and never appear in
    // a declared loop's predecessor set. Freeze the complete header set and
    // prove the property over every resolved physical edge before emission.
    luisa::unordered_set<size_t> physical_loop_headers;
    for (auto &loop : plan._loop_regions) {
        physical_loop_headers.emplace(
            plan._block_indices.at(loop.prepare));
    }
    for (auto &loop : plan._simple_loop_regions) {
        physical_loop_headers.emplace(
            physical_index(Target::synthetic(
                loop.header_synthetic_index)));
    }
    for (auto &region : plan._switch_regions) {
        if (region.loop_wrapped) {
            physical_loop_headers.emplace(
                plan._block_indices.at(region.header));
        }
    }
    for (auto source = size_t{0u};
         source < physical_block_count; ++source) {
        if (physical_reachable[source] == 0u) { continue; }
        for (auto target : physical_successors[source]) {
            if (physical_reachable[target] == 0u ||
                physical_dominators[source][target] == 0u ||
                physical_loop_headers.contains(target)) {
                continue;
            }
            if (reject_planning_precondition(luisa::format(
                    "SPIR-V control-flow plan rejected physical backedge "
                    "{} -> {} because its target is not a planned loop "
                    "header. restructure_cfg must recover every reachable "
                    "dominance cycle before codegen.",
                    source, target))) {
                return plan;
            }
        }
    }

    // A structured continue region may contain branches/selections, but all of
    // its exits must converge to one P-dominated physical predecessor. The
    // header must separately have one reachable non-P-dominated entry. Natural
    // loop membership alone is insufficient: a branch from after the declared
    // merge back to P is P-dominated too, but is not part of the declared
    // continue construct and would violate SPIR-V structured control flow.
    for (auto &loop : plan._loop_regions) {
        LUISA_ASSERT(loop.prepare_kind != SpirvLoopPrepareKind::INVALID,
                     "SPIR-V control-flow plan retained an invalid "
                     "Loop.prepare classification.");
        auto header = plan._block_indices.at(loop.prepare);
        auto continue_target = physical_index(loop.continue_target);
        auto merge_target = physical_index(loop.merge_target);
        // Role reachability is evidence for the physical boundary verdict, not
        // a precondition of its nonfatal query. In particular, a Loop whose
        // body exits directly can leave update unreachable; it then has zero
        // valid backedges and must return a failed verdict instead of asserting.
        luisa::vector<PhysicalLoopPredecessorFacts> predecessor_facts;
        predecessor_facts.reserve(physical_predecessors[header].size());
        for (auto predecessor : physical_predecessors[header]) {
            if (physical_reachable[predecessor] == 0u) { continue; }
            predecessor_facts.emplace_back(PhysicalLoopPredecessorFacts{
                .dominated_by_header =
                    physical_dominators[predecessor][header] != 0u,
                .dominated_by_continue_target =
                    physical_dominators[predecessor][continue_target] != 0u,
                .dominated_by_merge_target =
                    physical_dominators[predecessor][merge_target] != 0u,
            });
        }
        loop.physical_boundary = validate_physical_loop_boundary(
            luisa::span<const PhysicalLoopPredecessorFacts>{
                predecessor_facts.data(), predecessor_facts.size()});
        loop.physical_header_predecessor_count =
            loop.physical_boundary.reachable_predecessor_count;
        if (enforce_physical_loop_boundaries &&
            !loop.physical_boundary.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V control-flow plan rejected Loop.prepare with {} entry "
                "edge(s) and {} backedge(s); the unique backedge must be "
                "dominated by the declared continue target and must not be "
                "dominated by the declared merge target (continue={}, merge={}).",
                loop.physical_boundary.entry_edge_count,
                loop.physical_boundary.backedge_edge_count,
                loop.physical_boundary.backedge_dominated_by_continue_target,
                loop.physical_boundary.backedge_dominated_by_merge_target);
        }
    }

    // SimpleLoop has no explicit XIR prepare/update blocks, but emission gives
    // it the same physical SPIR-V shape: a synthetic loop header and a
    // synthetic continue target that branches back to that header. Validate
    // those physical blocks, not the logical body edge. In particular, an
    // ordinary edge from outside the SimpleLoop to its body is normalized to
    // the synthetic continue target and must therefore be rejected as a second
    // loop entry rather than mistaken for a legal backedge.
    for (auto &loop : plan._simple_loop_regions) {
        auto header = physical_index(Target::synthetic(
            loop.header_synthetic_index));
        auto continue_target = physical_index(Target::synthetic(
            loop.continue_synthetic_index));
        auto merge_target = physical_index(loop.merge_target);
        LUISA_ASSERT(header != continue_target && header != merge_target &&
                         continue_target != merge_target,
                     "SPIR-V SimpleLoop has overlapping synthetic header, "
                     "continue, or merge roles.");
        // Reachability is part of the boundary verdict, not a planner
        // precondition here. A malformed SimpleLoop may have no path to its
        // synthetic continue target; the nonfatal validator must report the
        // resulting missing backedge. Conversely, an intentional infinite loop
        // may leave its declared merge unreachable without invalidating the
        // header/continue dominance relation.
        luisa::vector<PhysicalLoopPredecessorFacts> predecessor_facts;
        predecessor_facts.reserve(physical_predecessors[header].size());
        for (auto predecessor : physical_predecessors[header]) {
            if (physical_reachable[predecessor] == 0u) { continue; }
            predecessor_facts.emplace_back(PhysicalLoopPredecessorFacts{
                .dominated_by_header =
                    physical_dominators[predecessor][header] != 0u,
                .dominated_by_continue_target =
                    physical_dominators[predecessor][continue_target] != 0u,
                .dominated_by_merge_target =
                    physical_dominators[predecessor][merge_target] != 0u,
            });
        }
        loop.physical_boundary = validate_physical_loop_boundary(
            luisa::span<const PhysicalLoopPredecessorFacts>{
                predecessor_facts.data(), predecessor_facts.size()});
        loop.physical_header_predecessor_count =
            loop.physical_boundary.reachable_predecessor_count;
        if (enforce_physical_loop_boundaries &&
            !loop.physical_boundary.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V control-flow plan rejected SimpleLoop synthetic header "
                "with {} entry edge(s) and {} backedge(s); the unique backedge "
                "must be dominated by the synthetic continue target and must "
                "not be dominated by the declared merge target "
                "(continue={}, merge={}).",
                loop.physical_boundary.entry_edge_count,
                loop.physical_boundary.backedge_edge_count,
                loop.physical_boundary
                    .backedge_dominated_by_continue_target,
                loop.physical_boundary
                    .backedge_dominated_by_merge_target);
        }
    }

    for (auto &region : plan._switch_regions) {
        if (!region.loop_wrapped) { continue; }
        auto header = plan._block_indices.at(region.header);
        auto continue_target = physical_index(Target::synthetic(
            region.continue_synthetic_index));
        auto merge_target = physical_index(region.loop_merge_target);
        LUISA_ASSERT(header != continue_target && header != merge_target &&
                         continue_target != merge_target,
                     "SPIR-V wrapped Switch has overlapping header, continue, "
                     "or loop-merge roles.");
        if (physical_reachable[header] == 0u ||
            physical_reachable[continue_target] == 0u ||
            physical_reachable[merge_target] == 0u) {
            if (reject_planning_precondition(
                    "SPIR-V wrapped Switch has an unreachable header, "
                    "continue, or loop-merge target.")) {
                return plan;
            }
        }
        luisa::vector<PhysicalLoopPredecessorFacts> predecessor_facts;
        predecessor_facts.reserve(physical_predecessors[header].size());
        for (auto predecessor : physical_predecessors[header]) {
            if (physical_reachable[predecessor] == 0u) { continue; }
            predecessor_facts.emplace_back(PhysicalLoopPredecessorFacts{
                .dominated_by_header =
                    physical_dominators[predecessor][header] != 0u,
                .dominated_by_continue_target =
                    physical_dominators[predecessor][continue_target] != 0u,
                .dominated_by_merge_target =
                    physical_dominators[predecessor][merge_target] != 0u,
            });
        }
        region.physical_boundary = validate_physical_loop_boundary(
            luisa::span<const PhysicalLoopPredecessorFacts>{
                predecessor_facts.data(), predecessor_facts.size()});
        region.physical_header_predecessor_count =
            region.physical_boundary.reachable_predecessor_count;
        if (enforce_physical_loop_boundaries &&
            !region.physical_boundary.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "SPIR-V control-flow plan rejected cyclic Switch header with "
                "{} entry edge(s) and {} backedge(s); the normalized unique "
                "backedge must be dominated by the synthetic continue target "
                "and must not be dominated by the synthetic loop merge "
                "(continue={}, merge={}).",
                region.physical_boundary.entry_edge_count,
                region.physical_boundary.backedge_edge_count,
                region.physical_boundary
                    .backedge_dominated_by_continue_target,
                region.physical_boundary
                    .backedge_dominated_by_merge_target);
        }
    }

    // A failed physical verdict is already sufficient for the nonfatal query;
    // do not feed that invalid graph into downstream Phi planning. When every
    // boundary succeeds, continue through the production-only edge/Phi plan as
    // a dry run as well, so dialect success proves that create() will not later
    // discover a planner-specific semantic failure.
    if (!enforce_physical_loop_boundaries) {
        auto physical_boundaries_succeeded = true;
        for (auto &&loop : plan._loop_regions) {
            physical_boundaries_succeeded &=
                loop.physical_boundary.succeeded();
        }
        for (auto &&loop : plan._simple_loop_regions) {
            physical_boundaries_succeeded &=
                loop.physical_boundary.succeeded();
        }
        for (auto &&region : plan._switch_regions) {
            if (region.loop_wrapped) {
                physical_boundaries_succeeded &=
                    region.physical_boundary.succeeded();
            }
        }
        if (!physical_boundaries_succeeded) { return plan; }
    }

    // Freeze the logical predecessor relation. The same relation drives exact
    // nearest-enclosing break/continue checks and Phi incoming validation.
    luisa::vector<luisa::vector<size_t>> successors(plan._blocks.size());
    luisa::vector<luisa::vector<size_t>> predecessors(plan._blocks.size());
    for (auto &block_plan : plan._blocks) {
        auto source_index = plan._block_indices.at(block_plan.block);
        for (auto *operand_use : block_plan.block->terminator()->operand_uses()) {
            auto *value = operand_use->value();
            if (value == nullptr || !value->isa<xir::BasicBlock>()) { continue; }
            auto *target = static_cast<const xir::BasicBlock *>(value);
            auto target_iter = plan._block_indices.find(target);
            LUISA_ASSERT(target_iter != plan._block_indices.end(),
                         "SPIR-V control-flow plan found an edge to a foreign block.");
            auto target_index = target_iter->second;
            if (std::find(successors[source_index].begin(), successors[source_index].end(), target_index) ==
                successors[source_index].end()) {
                successors[source_index].emplace_back(target_index);
                predecessors[target_index].emplace_back(source_index);
            }
        }
    }
    luisa::vector<uint8_t> reachable(plan._blocks.size(), 0u);
    luisa::vector<size_t> worklist;
    auto entry_index = plan._block_indices.at(function->body_block());
    worklist.emplace_back(entry_index);
    while (!worklist.empty()) {
        auto index = worklist.back();
        worklist.pop_back();
        if (reachable[index] != 0u) { continue; }
        reachable[index] = 1u;
        for (auto successor : successors[index]) { worklist.emplace_back(successor); }
    }
    // Compute ordinary dominators without a post-dominator dependency. This is
    // used only to identify the exact nearest enclosing structured loop for
    // BreakInst/ContinueInst; emission itself consumes the frozen edge table.
    auto block_count = plan._blocks.size();
    luisa::vector<luisa::vector<uint8_t>> dominators(
        block_count, luisa::vector<uint8_t>(block_count, 0u));
    for (size_t block_index = 0u; block_index < block_count; ++block_index) {
        if (reachable[block_index] == 0u || block_index == entry_index) { continue; }
        for (size_t candidate = 0u; candidate < block_count; ++candidate) {
            dominators[block_index][candidate] = reachable[candidate];
        }
    }
    dominators[entry_index][entry_index] = 1u;
    for (;;) {
        auto changed = false;
        for (size_t block_index = 0u; block_index < block_count; ++block_index) {
            if (block_index == entry_index || reachable[block_index] == 0u) { continue; }
            luisa::vector<uint8_t> next(block_count, 1u);
            auto has_predecessor = false;
            for (auto predecessor_index : predecessors[block_index]) {
                if (!has_predecessor) {
                    next = dominators[predecessor_index];
                    has_predecessor = true;
                } else {
                    for (size_t candidate = 0u; candidate < block_count; ++candidate) {
                        next[candidate] &= dominators[predecessor_index][candidate];
                    }
                }
            }
            LUISA_ASSERT(has_predecessor,
                         "SPIR-V reachable block has no logical predecessor.");
            next[block_index] = 1u;
            if (next != dominators[block_index]) {
                dominators[block_index] = std::move(next);
                changed = true;
            }
        }
        if (!changed) { break; }
    }
    auto dominates = [&](const xir::BasicBlock *lhs, const xir::BasicBlock *rhs) noexcept {
        auto lhs_index = plan._block_indices.at(lhs);
        auto rhs_index = plan._block_indices.at(rhs);
        return reachable[lhs_index] != 0u && reachable[rhs_index] != 0u &&
               dominators[rhs_index][lhs_index] != 0u;
    };
    auto dominator_depth = [&](const xir::BasicBlock *block) noexcept {
        auto depth = size_t{0u};
        for (auto bit : dominators[plan._block_indices.at(block)]) { depth += bit != 0u; }
        return depth;
    };
    for (auto &block_plan : plan._blocks) {
        auto *terminator = block_plan.block->terminator();
        auto tag = terminator->derived_instruction_tag();
        if (tag != xir::DerivedInstructionTag::BREAK &&
            tag != xir::DerivedInstructionTag::CONTINUE) {
            continue;
        }
        if (reachable[plan._block_indices.at(block_plan.block)] == 0u) { continue; }
        auto is_continue = tag == xir::DerivedInstructionTag::CONTINUE;
        const xir::BasicBlock *expected_target = nullptr;
        auto best_depth = size_t{0u};
        auto ambiguous = false;
        auto consider_scope = [&](const xir::BasicBlock *owner,
                                  const xir::BasicBlock *merge,
                                  const xir::BasicBlock *continue_target) noexcept {
            if (is_continue && continue_target == nullptr) { return; }
            if (owner == block_plan.block || !dominates(owner, block_plan.block) ||
                merge == block_plan.block || dominates(merge, block_plan.block)) {
                return;
            }
            auto *candidate = is_continue ? continue_target : merge;
            auto depth = dominator_depth(owner);
            if (expected_target == nullptr || depth > best_depth) {
                expected_target = candidate;
                best_depth = depth;
                ambiguous = false;
            } else if (depth == best_depth && expected_target != candidate) {
                ambiguous = true;
            }
        };
        for (auto &region : plan._loop_regions) {
            consider_scope(region.owner, region.merge, region.update);
        }
        for (auto &region : plan._simple_loop_regions) {
            consider_scope(region.owner, region.merge, region.body);
        }
        for (auto &region : plan._switch_regions) {
            consider_scope(region.header, region.instruction->merge_block(), nullptr);
        }
        auto *actual_target = tag == xir::DerivedInstructionTag::BREAK ?
                                  static_cast<const xir::BreakInst *>(terminator)->target_block() :
                                  static_cast<const xir::ContinueInst *>(terminator)->target_block();
        if (expected_target == nullptr || ambiguous || actual_target != expected_target) {
            if (reject_planning_precondition(luisa::format(
                    "SPIR-V control-flow plan rejected {} whose target is not "
                    "the nearest enclosing structured role.",
                    is_continue ? "ContinueInst" : "BreakInst"))) {
                return plan;
            }
        }
    }

    // Resolve the physical target of one logical CFG edge. The result may be a
    // synthetic forwarding block; Phi planning follows its immutable
    // continuation chain rather than guessing predecessors during emission.
    auto resolved_edge_target = [&](const xir::BasicBlock *predecessor,
                                    const xir::BasicBlock *logical_target) noexcept -> Target {
        auto *terminator = predecessor->terminator();
        switch (terminator->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                auto *inst = static_cast<const xir::IfInst *>(terminator);
                auto &region = plan._if_regions.at(plan._if_indices.at(inst));
                auto matches_true = inst->true_block() == logical_target;
                auto matches_false = inst->false_block() == logical_target;
                LUISA_ASSERT(matches_true || matches_false,
                             "SPIR-V Phi incoming is not an If successor.");
                if (matches_true && matches_false) {
                    LUISA_ASSERT(region.true_target == region.false_target,
                                 "SPIR-V degenerate If edges resolved to different physical targets.");
                }
                return matches_true ? region.true_target : region.false_target;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                auto *inst = static_cast<const xir::SwitchInst *>(terminator);
                auto &region = plan._switch_regions.at(plan._switch_indices.at(inst));
                auto matched = false;
                auto target = Target{};
                auto accept_target = [&](Target candidate) noexcept {
                    if (matched) {
                        LUISA_ASSERT(target == candidate,
                                     "SPIR-V duplicate Switch targets resolved to different physical blocks.");
                    } else {
                        target = candidate;
                        matched = true;
                    }
                };
                if (inst->default_block() == logical_target) {
                    accept_target(region.default_target);
                }
                for (size_t case_index = 0u; case_index < inst->case_count(); ++case_index) {
                    if (inst->case_block(case_index) == logical_target) {
                        accept_target(region.case_targets[case_index]);
                    }
                }
                LUISA_ASSERT(matched,
                             "SPIR-V Phi incoming is not a Switch successor.");
                return target;
            }
            case xir::DerivedInstructionTag::LOOP: {
                auto *inst = static_cast<const xir::LoopInst *>(terminator);
                auto &region = plan._loop_regions.at(plan._loop_indices.at(inst));
                LUISA_ASSERT(region.prepare == logical_target,
                             "SPIR-V Phi incoming is not a Loop entry edge.");
                return region.entry_target;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                auto *inst = static_cast<const xir::SimpleLoopInst *>(terminator);
                auto &region = plan._simple_loop_regions.at(plan._simple_loop_indices.at(inst));
                LUISA_ASSERT(region.body == logical_target,
                             "SPIR-V Phi incoming is not a SimpleLoop entry edge.");
                return Target::synthetic(region.header_synthetic_index);
            }
            case xir::DerivedInstructionTag::BRANCH:
            case xir::DerivedInstructionTag::BREAK:
            case xir::DerivedInstructionTag::CONTINUE: {
                auto *inst = terminator;
                auto *target = static_cast<const xir::BranchTerminatorInstruction *>(inst)->target_block();
                LUISA_ASSERT(target == logical_target,
                             "SPIR-V Phi incoming is not a branch successor.");
                return plan._edge_targets.at(inst);
            }
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: {
                auto *inst = static_cast<const xir::ConditionalBranchInst *>(terminator);
                auto target_iter =
                    plan._conditional_branch_targets.find(inst);
                LUISA_ASSERT(
                    target_iter !=
                        plan._conditional_branch_targets.end(),
                    "SPIR-V Phi incoming crossed an unplanned "
                    "ConditionalBranch.");
                if (inst->true_block() == logical_target) {
                    return target_iter->second[0u];
                }
                LUISA_ASSERT(inst->false_block() == logical_target,
                             "SPIR-V Phi incoming is not a ConditionalBranch successor.");
                return target_iter->second[1u];
            }
            default:
                static_cast<void>(reject_planning_precondition(luisa::format(
                    "SPIR-V Phi incoming predecessor {} has no executable "
                    "edge.",
                    xir::to_string(
                        terminator->derived_instruction_tag()))));
                return Target{};
        }
    };

    // Plan native OpPhi placement. Most Phis stay in their immutable XIR block
    // entry. SimpleLoop body Phis live in the synthetic loop header; auxiliary
    // Phis in forwarding blocks preserve values across continue/trampoline
    // convergence. Result IDs and operands are allocated by the emitter before
    // any ordinary instruction is emitted.
    for (auto &block_plan : plan._blocks) {
        auto *block = block_plan.block;
        auto saw_non_phi = false;
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<xir::PhiInst>()) {
                saw_non_phi = true;
                continue;
            }
            LUISA_ASSERT(!saw_non_phi,
                         "SPIR-V control-flow plan rejected PhiInst after a non-Phi instruction.");
            auto *phi = static_cast<const xir::PhiInst *>(instruction);
            LUISA_ASSERT(phi->type() != nullptr && phi->incoming_count() != 0u,
                         "SPIR-V control-flow plan rejected an empty or untyped PhiInst.");
            auto block_index = plan._block_indices.at(block);
            auto result_target = Target::xir(block);
            if (auto simple_iter = plan._simple_loop_body_indices.find(block);
                simple_iter != plan._simple_loop_body_indices.end()) {
                result_target = Target::synthetic(
                    plan._simple_loop_regions[simple_iter->second].header_synthetic_index);
            }
            PhiPlan phi_plan{
                .instruction = phi,
                .logical_block = block,
                .result_target = result_target};
            phi_plan.incomings.reserve(phi->incoming_count());
            luisa::unordered_set<const xir::BasicBlock *> incoming_blocks;
            for (size_t incoming_index = 0u; incoming_index < phi->incoming_count(); ++incoming_index) {
                auto incoming = phi->incoming(incoming_index);
                LUISA_ASSERT(incoming.value != nullptr && incoming.block != nullptr &&
                                 incoming.value->type() == phi->type(),
                             "SPIR-V control-flow plan rejected an invalid Phi incoming value.");
                auto predecessor_iter = plan._block_indices.find(incoming.block);
                if (predecessor_iter == plan._block_indices.end()) {
                    LUISA_ASSERT(incoming.block->parent_function() == function,
                                 "SPIR-V Phi incoming block belongs to another function.");
                    continue;// incoming edge from an omitted true orphan
                }
                LUISA_ASSERT(std::find(predecessors[block_index].begin(), predecessors[block_index].end(),
                                       predecessor_iter->second) != predecessors[block_index].end(),
                             "SPIR-V Phi incoming block is not a logical predecessor.");
                LUISA_ASSERT(incoming_blocks.emplace(incoming.block).second,
                             "SPIR-V Phi contains duplicate incoming blocks.");
                PhiIncomingPlan incoming_plan{
                    .value = incoming.value,
                    .predecessor = incoming.block};
                const IfRegion *loop_boundary_guard = nullptr;
                for (auto &if_region : plan._if_regions) {
                    if (!if_region.emit_selection_merge &&
                        if_region.loop_boundary_exit_target ==
                            block &&
                        if_region
                                .loop_boundary_exit_predecessor ==
                            incoming.block) {
                        LUISA_ASSERT(
                            loop_boundary_guard == nullptr,
                            "SPIR-V Phi incoming is bypassed by multiple "
                            "loop-boundary guards.");
                        loop_boundary_guard = &if_region;
                    }
                }
                if (loop_boundary_guard != nullptr) {
                    // The empty proxy chain is physically dead. Its incoming
                    // value is already available at the guard header (the
                    // chain contains no value-producing instructions), and
                    // the header is the real OpPhi predecessor.
                    incoming_plan.predecessor =
                        loop_boundary_guard->header;
                }
                if (incoming.block->terminator()->isa<xir::SwitchInst>()) {
                    auto *switch_inst = static_cast<const xir::SwitchInst *>(
                        incoming.block->terminator());
                    auto &switch_region = plan._switch_regions.at(
                        plan._switch_indices.at(switch_inst));
                    if (switch_region.loop_wrapped) {
                        // The logical Switch header physically branches to the
                        // synthetic dispatch block before taking the selected
                        // case edge. Preserve that real predecessor boundary
                        // with an auxiliary OpPhi in the dispatch block.
                        incoming_plan.forwarding_synthetic_indices.emplace_back(
                            switch_region.dispatch_synthetic_index);
                    }
                }
                auto target = [&] {
                    if (loop_boundary_guard == nullptr) {
                        return resolved_edge_target(
                            incoming.block, block);
                    }
                    auto *inst =
                        loop_boundary_guard->instruction;
                    auto exit_is_true =
                        inst->true_block() !=
                        inst->merge_block();
                    return exit_is_true ?
                               loop_boundary_guard->true_target :
                               loop_boundary_guard->false_target;
                }();
                if (!plan._planning_diagnostic.empty()) { return plan; }
                auto forwarding_guard = size_t{0u};
                while (!(target == result_target)) {
                    if (target.kind != Target::Kind::SYNTHETIC_BLOCK ||
                        target.synthetic_index >= plan._synthetic_blocks.size()) {
                        if (reject_planning_precondition(
                                "SPIR-V Phi incoming edge does not reach its "
                                "planned result block.")) {
                            return plan;
                        }
                    }
                    auto &synthetic = plan._synthetic_blocks[target.synthetic_index];
                    if (synthetic.kind == SyntheticBlockKind::SIMPLE_LOOP_HEADER) {
                        if (reject_planning_precondition(
                                "SPIR-V Phi incoming crossed an unrelated "
                                "SimpleLoop header.")) {
                            return plan;
                        }
                    }
                    incoming_plan.forwarding_synthetic_indices.emplace_back(target.synthetic_index);
                    target = synthetic.continuation;
                    if (++forwarding_guard > plan._synthetic_blocks.size()) {
                        if (reject_planning_precondition(
                                "SPIR-V Phi forwarding path contains a "
                                "synthetic cycle.")) {
                            return plan;
                        }
                    }
                }
                phi_plan.incomings.emplace_back(std::move(incoming_plan));
            }
            LUISA_ASSERT(phi_plan.incomings.size() == predecessors[block_index].size(),
                         "SPIR-V Phi incomings do not cover every planned logical predecessor.");
            LUISA_ASSERT(!phi_plan.incomings.empty(),
                         "SPIR-V Phi has no predecessor in the executable/structural closure.");
            auto phi_index = plan._phi_plans.size();
            auto [_, inserted] = plan._phi_indices.emplace(phi, phi_index);
            LUISA_ASSERT(inserted, "SPIR-V Phi was planned more than once.");
            plan._phi_plans.emplace_back(std::move(phi_plan));
        }
    }
    return plan;
}

ControlFlowPlan::FunctionPhysicalLoopBoundaryValidation
ControlFlowPlan::validate_function_physical_loop_boundaries(
    const xir::FunctionDefinition *function) noexcept {
    auto plan = _create(function, false);
    FunctionPhysicalLoopBoundaryValidation validation;
    validation.planning_diagnostic =
        std::move(plan._planning_diagnostic);
    if (!validation.planning_succeeded()) { return validation; }
    validation.loops.reserve(
        plan._loop_regions.size() + plan._simple_loop_regions.size() +
        plan._switch_regions.size());
    for (auto &&loop : plan._loop_regions) {
        validation.loops.emplace_back(loop.physical_boundary);
    }
    for (auto &&loop : plan._simple_loop_regions) {
        validation.loops.emplace_back(loop.physical_boundary);
    }
    for (auto &&region : plan._switch_regions) {
        if (region.loop_wrapped) {
            validation.loops.emplace_back(region.physical_boundary);
        }
    }
    return validation;
}

ControlFlowPlan ControlFlowPlan::create(
    const xir::FunctionDefinition *function) noexcept {
    return _create(function, true);
}

const ControlFlowPlan::BlockPlan &ControlFlowPlan::block(const xir::BasicBlock *block) const noexcept {
    auto iter = _block_indices.find(block);
    LUISA_ASSERT(iter != _block_indices.end(), "SPIR-V control-flow plan has no entry for XIR block.");
    return _blocks[iter->second];
}

const ControlFlowPlan::IfRegion &ControlFlowPlan::if_region(const xir::IfInst *instruction) const noexcept {
    auto iter = _if_indices.find(instruction);
    LUISA_ASSERT(iter != _if_indices.end(), "SPIR-V control-flow plan has no If region.");
    return _if_regions[iter->second];
}

const ControlFlowPlan::LoopRegion &ControlFlowPlan::loop_region(const xir::LoopInst *instruction) const noexcept {
    auto iter = _loop_indices.find(instruction);
    LUISA_ASSERT(iter != _loop_indices.end(), "SPIR-V control-flow plan has no Loop region.");
    return _loop_regions[iter->second];
}

const ControlFlowPlan::SimpleLoopRegion &ControlFlowPlan::simple_loop_region(const xir::SimpleLoopInst *instruction) const noexcept {
    auto iter = _simple_loop_indices.find(instruction);
    LUISA_ASSERT(iter != _simple_loop_indices.end(), "SPIR-V control-flow plan has no SimpleLoop region.");
    return _simple_loop_regions[iter->second];
}

const ControlFlowPlan::SwitchRegion &ControlFlowPlan::switch_region(const xir::SwitchInst *instruction) const noexcept {
    auto iter = _switch_indices.find(instruction);
    LUISA_ASSERT(iter != _switch_indices.end(), "SPIR-V control-flow plan has no Switch region.");
    return _switch_regions[iter->second];
}

const ControlFlowPlan::PhiPlan &ControlFlowPlan::phi_plan(const xir::PhiInst *instruction) const noexcept {
    auto iter = _phi_indices.find(instruction);
    LUISA_ASSERT(iter != _phi_indices.end(), "SPIR-V control-flow plan has no Phi plan.");
    return _phi_plans[iter->second];
}

const ControlFlowPlan::LoopRegion *ControlFlowPlan::loop_with_prepare(const xir::BasicBlock *prepare) const noexcept {
    if (auto iter = _loop_prepare_indices.find(prepare); iter != _loop_prepare_indices.end()) {
        return &_loop_regions[iter->second];
    }
    return nullptr;
}

ControlFlowPlan::Target ControlFlowPlan::edge_target(const xir::Instruction *instruction) const noexcept {
    auto iter = _edge_targets.find(instruction);
    LUISA_ASSERT(iter != _edge_targets.end(),
                 "SPIR-V control-flow plan has no executable edge for {}.",
                 xir::to_string(instruction->derived_instruction_tag()));
    return iter->second;
}

const std::array<ControlFlowPlan::Target, 2u> &
ControlFlowPlan::conditional_branch_targets(
    const xir::ConditionalBranchInst *instruction) const noexcept {
    auto iter = _conditional_branch_targets.find(instruction);
    LUISA_ASSERT(
        iter != _conditional_branch_targets.end(),
        "SPIR-V control-flow plan has no conditional targets.");
    return iter->second;
}

}// namespace lc::spirv
