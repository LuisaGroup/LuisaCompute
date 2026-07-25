#include "call_graph_validation.h"
#include "structural_closure.h"

#include <cstdint>
#include <utility>

#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/module.h>

namespace lc::spirv {

using namespace luisa;
using namespace luisa::compute;

SpirvReachableCallGraphValidationResult
validate_spirv_reachable_call_graph(
    const xir::Module *module) noexcept {
    SpirvReachableCallGraphValidationResult result;
    if (module == nullptr) { return result; }

    enum class VisitState : uint8_t {
        ACTIVE,
        COMPLETE,
    };
    luisa::unordered_map<const xir::Function *, VisitState> states;
    states.reserve(64u);

    auto visit = [&](auto &&self,
                     const xir::Function *function) noexcept -> void {
        if (function == nullptr || !function->is_definition()) { return; }
        if (auto iter = states.find(function); iter != states.end()) {
            if (iter->second == VisitState::COMPLETE) { return; }
            // Active callees are diagnosed at the call edge below, where the
            // source block and instruction are available.
            return;
        }
        states.emplace(function, VisitState::ACTIVE);
        auto *definition = function->definition();
        auto closure =
            plan_spirv_codegen_structural_closure(definition);
        if (!closure.succeeded()) {
            auto message = [&]() noexcept -> luisa::string {
                switch (closure.status) {
                    case SpirvCodegenStructuralClosureStatus::NULL_FUNCTION:
                        return "Native XIR-to-SPIR-V reachable call-graph planning cannot inspect a null function definition.";
                    case SpirvCodegenStructuralClosureStatus::MISSING_BODY:
                        return luisa::format(
                            "Native XIR-to-SPIR-V reachable call-graph planning requires function '{}' to have a body block.",
                            function->name().value_or("<unnamed>"));
                    case SpirvCodegenStructuralClosureStatus::UNOWNED_BLOCK:
                        if (closure.invalid_instruction != nullptr &&
                            !closure.invalid_role.empty()) {
                            auto construct = [&]() noexcept
                                -> luisa::string_view {
                                switch (closure.invalid_instruction
                                            ->derived_instruction_tag()) {
                                    case xir::DerivedInstructionTag::IF:
                                        return "If";
                                    case xir::DerivedInstructionTag::SWITCH:
                                        return "Switch";
                                    case xir::DerivedInstructionTag::
                                        INDEXED_BRANCH:
                                        return "IndexedBranch";
                                    case xir::DerivedInstructionTag::LOOP:
                                        return "Loop";
                                    case xir::DerivedInstructionTag::SIMPLE_LOOP:
                                        return "SimpleLoop";
                                    case xir::DerivedInstructionTag::BRANCH:
                                        return "Branch";
                                    case xir::DerivedInstructionTag::CONDITIONAL_BRANCH:
                                        return "ConditionalBranch";
                                    case xir::DerivedInstructionTag::BREAK:
                                        return "Break";
                                    case xir::DerivedInstructionTag::CONTINUE:
                                        return "Continue";
                                    case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                                        return "RayQueryLoop";
                                    case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                                        return "RayQueryDispatch";
                                    case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
                                        return "AutodiffScope";
                                    case xir::DerivedInstructionTag::OUTLINE:
                                        return "Outline";
                                    default: return "structured terminator";
                                }
                            }();
                            return luisa::format(
                                "Native XIR-to-SPIR-V reachable call-graph planning requires {} to have a non-null {} block owned by function '{}'; found a null, non-block, or foreign structural block.",
                                construct, closure.invalid_role,
                                function->name().value_or("<unnamed>"));
                        }
                        return luisa::format(
                            "Native XIR-to-SPIR-V reachable call-graph planning found a null or foreign structural block in function '{}'.",
                            function->name().value_or("<unnamed>"));
                    case SpirvCodegenStructuralClosureStatus::SUCCESS:
                        break;
                }
                return "Native XIR-to-SPIR-V reachable call-graph planning failed without a structural-closure diagnostic.";
            }();
            result.diagnostics.emplace_back(
                SpirvReachableCallGraphDiagnostic{
                    .function = function,
                    .block = closure.invalid_instruction == nullptr ?
                                 closure.invalid_block :
                                 closure.invalid_instruction->parent_block(),
                    .instruction = closure.invalid_instruction,
                    .message = std::move(message),
                });
            states[function] = VisitState::COMPLETE;
            return;
        }
        for (auto *block : closure.blocks) {
            block->traverse_instructions(
                [&](const xir::Instruction *instruction) noexcept {
                    for (auto *operand_use :
                         instruction->operand_uses()) {
                        auto *operand = operand_use->value();
                        if (operand == nullptr ||
                            !operand->isa<xir::Function>()) {
                            continue;
                        }
                        auto *callee = static_cast<const xir::Function *>(
                            operand);
                        if (!callee->is_definition()) { continue; }
                        if (auto iter = states.find(callee);
                            iter != states.end() &&
                            iter->second == VisitState::ACTIVE) {
                            result.diagnostics.emplace_back(
                                SpirvReachableCallGraphDiagnostic{
                                    .function = function,
                                    .block = block,
                                    .instruction = instruction,
                                    .message = luisa::format(
                                        "Native XIR-to-SPIR-V rejected a reachable recursive callable cycle: function '{}' calls the still-active function '{}'.",
                                        function->name().value_or(
                                            "<unnamed>"),
                                        callee->name().value_or(
                                            "<unnamed>")),
                                });
                            continue;
                        }
                        self(self, callee);
                    }
                });
        }
        states[function] = VisitState::COMPLETE;
        result.functions_post_order.emplace_back(function);
    };

    // The native emitter starts usage analysis at the kernel. Validate every
    // kernel root here so this helper also remains meaningful for a malformed
    // standalone module; the AST/XIR ABI boundary separately requires exactly
    // one kernel before production codegen.
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() ==
            xir::DerivedFunctionTag::KERNEL) {
            visit(visit, function);
        }
    }
    if (!result.succeeded()) {
        result.functions_post_order.clear();
    }
    return result;
}

}// namespace lc::spirv
