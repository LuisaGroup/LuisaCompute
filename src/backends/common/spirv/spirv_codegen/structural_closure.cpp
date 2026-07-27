#include "structural_closure.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/autodiff.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/indexed_branch.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/switch.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

SpirvCodegenStructuralClosure
plan_spirv_codegen_structural_closure(
    const xir::FunctionDefinition *function) noexcept {
    SpirvCodegenStructuralClosure result;
    if (function == nullptr) {
        result.status =
            SpirvCodegenStructuralClosureStatus::NULL_FUNCTION;
        return result;
    }
    if (function->body_block() == nullptr) {
        result.status = SpirvCodegenStructuralClosureStatus::MISSING_BODY;
        return result;
    }
    luisa::unordered_set<const xir::BasicBlock *> owned;
    for (auto *block : function->basic_blocks()) { owned.emplace(block); }
    auto validate_block = [&](
                              const xir::BasicBlock *block,
                              const xir::Instruction *owner = nullptr,
                              luisa::string_view role = {}) noexcept {
        if (block != nullptr && owned.contains(block) &&
            block->parent_function() == function) {
            return true;
        }
        result.status = SpirvCodegenStructuralClosureStatus::UNOWNED_BLOCK;
        result.invalid_block = block;
        result.invalid_instruction = owner;
        result.invalid_role = role;
        return false;
    };
    if (!validate_block(function->body_block(), nullptr, "body entry")) {
        return result;
    }

    auto block_operand = [](const xir::Instruction *instruction,
                            size_t index) noexcept
        -> const xir::BasicBlock * {
        if (index >= instruction->operand_count()) { return nullptr; }
        auto *value = instruction->operand(index);
        return value != nullptr && value->isa<xir::BasicBlock>() ?
                   static_cast<const xir::BasicBlock *>(value) :
                   nullptr;
    };
    auto traverse_encoded_successors = [&]<typename Visit>(
                                           const xir::Instruction *terminator,
                                           Visit &&visit) noexcept -> bool {
        auto visit_operand = [&](size_t index,
                                 luisa::string_view role) noexcept {
            auto *target = block_operand(terminator, index);
            if (!validate_block(target, terminator, role)) { return false; }
            return visit(target);
        };
        switch (terminator->derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF:
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
                return visit_operand(
                           xir::ConditionalBranchTerminatorInstruction::
                               operand_index_true_target,
                           "true target") &&
                       visit_operand(
                           xir::ConditionalBranchTerminatorInstruction::
                               operand_index_false_target,
                           "false target");
            case xir::DerivedInstructionTag::SWITCH:
            case xir::DerivedInstructionTag::INDEXED_BRANCH: {
                auto *indexed_branch = static_cast<
                    const xir::IndexedBranchTerminatorInstruction *>(
                    terminator);
                if (!visit_operand(
                        xir::IndexedBranchTerminatorInstruction::
                            operand_index_default_block,
                        "default target")) {
                    return false;
                }
                for (auto i = size_t{0u};
                     i < indexed_branch->case_count(); ++i) {
                    if (!visit_operand(
                            xir::IndexedBranchTerminatorInstruction::
                                    operand_index_case_block_offset +
                                i,
                            "case target")) {
                        return false;
                    }
                }
                return true;
            }
            case xir::DerivedInstructionTag::LOOP:
                return visit_operand(
                    xir::LoopInst::operand_index_prepare_block,
                    "prepare");
            case xir::DerivedInstructionTag::SIMPLE_LOOP:
                return visit_operand(
                    xir::SimpleLoopInst::operand_index_body_block,
                    "body");
            case xir::DerivedInstructionTag::BRANCH:
            case xir::DerivedInstructionTag::BREAK:
            case xir::DerivedInstructionTag::CONTINUE:
            case xir::DerivedInstructionTag::OUTLINE:
                return visit_operand(
                    xir::BranchTerminatorInstruction::operand_index_target,
                    "target");
            case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                return visit_operand(
                    xir::RayQueryLoopInst::operand_index_dispatch_block,
                    "dispatch");
            case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                return visit_operand(
                           xir::RayQueryDispatchInst::operand_index_exit_block,
                           "exit target") &&
                       visit_operand(
                           xir::RayQueryDispatchInst::
                               operand_index_on_surface_candidate_block,
                           "surface-candidate target") &&
                       visit_operand(
                           xir::RayQueryDispatchInst::
                               operand_index_on_procedural_candidate_block,
                           "procedural-candidate target");
            case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
                return visit_operand(
                    xir::AutodiffScopeInst::operand_index_entry_block,
                    "entry");
            default: return true;
        }
    };

    luisa::vector<const xir::BasicBlock *> post_order;
    luisa::unordered_set<const xir::BasicBlock *> visited;
    auto visit_ordinary = [&](auto &&self,
                              const xir::BasicBlock *block) noexcept -> bool {
        if (!validate_block(block)) { return false; }
        if (!visited.emplace(block).second) { return true; }
        if (block->is_terminated() &&
            !traverse_encoded_successors(
                block->terminator(),
                [&](const xir::BasicBlock *successor) noexcept {
                    return self(self, successor);
                })) {
            return false;
        }
        post_order.emplace_back(block);
        return true;
    };
    if (!visit_ordinary(visit_ordinary, function->body_block())) {
        return result;
    }
    result.blocks.assign(post_order.rbegin(), post_order.rend());
    result.ordinary_block_count = result.blocks.size();

    auto append = [&](const xir::BasicBlock *block) noexcept {
        if (visited.emplace(block).second) {
            result.blocks.emplace_back(block);
        }
        return true;
    };
    auto append_role = [&](const xir::BasicBlock *block,
                           const xir::Instruction *owner,
                           luisa::string_view role) noexcept {
        return validate_block(block, owner, role) && append(block);
    };

    // Normal traversal follows the encoded CFG successors above. Loop
    // body/update and structured merges are raw role pointers in parts of XIR,
    // so close over them explicitly and recursively. Required encoded roles are
    // checked again for blocks introduced only by a raw role.
    for (size_t index = 0u; index < result.blocks.size(); ++index) {
        auto *block = result.blocks[index];
        if (!block->is_terminated()) { continue; }
        auto *terminator = block->terminator();
        if (!traverse_encoded_successors(
                terminator,
                [&](const xir::BasicBlock *successor) noexcept {
                    return append(successor);
                })) {
            return result;
        }
        if (auto *merge = terminator->control_flow_merge();
            merge != nullptr &&
            !append_role(merge->merge_block(), terminator, "merge")) {
            return result;
        }
        if (terminator->isa<xir::LoopInst>()) {
            auto *loop = static_cast<const xir::LoopInst *>(terminator);
            if (!append_role(loop->body_block(), terminator, "body") ||
                !append_role(loop->update_block(), terminator, "update")) {
                return result;
            }
        }
    }
    return result;
}

luisa::vector<const xir::BasicBlock *>
collect_spirv_codegen_structural_closure(
    const xir::FunctionDefinition *function) noexcept {
    auto result = plan_spirv_codegen_structural_closure(function);
    LUISA_ASSERT(
        result.succeeded(),
        "SPIR-V structural closure rejected function {} with status {} and block {}.",
        static_cast<const void *>(function),
        static_cast<uint32_t>(result.status),
        static_cast<const void *>(result.invalid_block));
    return std::move(result.blocks);
}

}// namespace lc::spirv
