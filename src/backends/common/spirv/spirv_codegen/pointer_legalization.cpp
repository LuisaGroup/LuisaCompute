#include "pointer_legalization.h"
#include "argument_usage.h"
#include "structural_closure.h"

#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/atomic.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/passes/destructure_cfg.h>

namespace lc::spirv {

namespace xir = luisa::compute::xir;
using luisa::compute::Type;
using luisa::compute::Usage;

namespace {

struct PointerCall {
    xir::CallInst *call;
    xir::Function *callee;
};

struct StructuredInventory {
    size_t switch_count{0u};
    size_t unsupported_count{0u};

    [[nodiscard]] bool contains_any() const noexcept {
        return switch_count != 0u || unsupported_count != 0u;
    }
};

[[nodiscard]] bool has_single_owned_block(
    const xir::FunctionDefinition *definition) noexcept {
    if (definition == nullptr || definition->body_block() == nullptr) {
        return false;
    }
    auto closure = plan_spirv_codegen_structural_closure(definition);
    return closure.succeeded() && closure.blocks.size() == 1u;
}

[[nodiscard]] bool is_indirect_dispatch_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() == "LC_IndirectDispatchBuffer";
}

[[nodiscard]] bool usage_contains(Usage usage, Usage expected) noexcept {
    return (luisa::to_underlying(usage) &
            luisa::to_underlying(expected)) != 0u;
}

[[nodiscard]] bool call_requires_specialization(
    const xir::CallInst *call, const xir::Function *callee,
    const SpirvFunctionArgumentAnalysisMap &usage,
    const SpirvReadonlyResourceOriginMap
        &readonly_resource_origins) noexcept {
    if (call == nullptr || callee == nullptr ||
        call->argument_count() != callee->arguments().count_size()) {
        return true;
    }
    auto index = size_t{0u};
    for (auto *formal : callee->arguments()) {
        auto *actual = call->argument(index++);
        if (formal == nullptr || formal->type() == nullptr ||
            actual == nullptr) {
            return true;
        }
        auto *type = formal->type();
        if (is_indirect_dispatch_type(type)) { return true; }
        if (formal->is_resource()) {
            auto argument_usage = spirv_function_argument_usage_of(
                usage, callee, formal);
            if (argument_usage == Usage::NONE) { continue; }
            if (type->is_buffer() || type->is_bindless_array()) {
                if (!readonly_resource_origins.contains(formal)) {
                    return true;
                }
                continue;
            }
            if (type->is_accel() &&
                (usage_contains(argument_usage, Usage::WRITE) ||
                 spirv_function_argument_requires_accel_instance_buffer(
                     usage, callee, formal))) {
                return true;
            }
            if (type->is_texture() &&
                usage_contains(argument_usage, Usage::READ) &&
                usage_contains(argument_usage, Usage::WRITE)) {
                return true;
            }
        }
        if (formal->is_reference() &&
            !validate_spirv_callable_reference_actual(actual)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] luisa::vector<PointerCall> collect_pointer_calls(
    xir::Module *module,
    const SpirvFunctionArgumentAnalysisMap &usage,
    const SpirvReadonlyResourceOriginMap
        &readonly_resource_origins) noexcept {
    luisa::vector<PointerCall> calls;
    for (auto *function : module->function_list()) {
        auto *definition = function->definition();
        if (definition == nullptr) { continue; }
        auto closure = plan_spirv_codegen_structural_closure(definition);
        if (!closure.succeeded()) { continue; }
        for (auto *const_block : closure.blocks) {
            auto *block = const_cast<xir::BasicBlock *>(const_block);
            for (auto *instruction : block->instructions()) {
                if (!instruction->isa<xir::CallInst>()) { continue; }
                auto *call = static_cast<xir::CallInst *>(instruction);
                auto *callee = call->callee();
                if (callee == nullptr || callee->definition() == nullptr ||
                    callee->derived_function_tag() !=
                        xir::DerivedFunctionTag::CALLABLE) {
                    continue;
                }
                if (call_requires_specialization(
                        call, callee, usage,
                        readonly_resource_origins)) {
                    calls.emplace_back(PointerCall{call, callee});
                }
            }
        }
    }
    return calls;
}

[[nodiscard]] StructuredInventory inspect_structured_control_flow(
    const xir::FunctionDefinition *definition) noexcept {
    StructuredInventory inventory;
    if (definition == nullptr) { return inventory; }
    auto closure = plan_spirv_codegen_structural_closure(definition);
    if (!closure.succeeded()) {
        inventory.unsupported_count++;
        return inventory;
    }
    for (auto *block : closure.blocks) {
        for (auto *instruction : block->instructions()) {
            switch (instruction->derived_instruction_tag()) {
                case xir::DerivedInstructionTag::SWITCH:
                    inventory.switch_count++;
                    break;
                case xir::DerivedInstructionTag::IF:
                case xir::DerivedInstructionTag::LOOP:
                case xir::DerivedInstructionTag::SIMPLE_LOOP:
                case xir::DerivedInstructionTag::BREAK:
                case xir::DerivedInstructionTag::CONTINUE:
                case xir::DerivedInstructionTag::RAY_QUERY_LOOP:
                case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH:
                case xir::DerivedInstructionTag::AUTODIFF_SCOPE:
                case xir::DerivedInstructionTag::OUTLINE:
                    inventory.unsupported_count++;
                    break;
                default: break;
            }
        }
    }
    return inventory;
}

[[nodiscard]] bool destructure_boundary_is_valid(
    xir::Function *function) noexcept {
    return function != nullptr && function->definition() != nullptr &&
           xir::destructure_cfg_pass_preflight_function(function)
               .succeeded();
}

[[nodiscard]] bool call_shape_is_valid(
    const xir::CallInst *call, const xir::Function *callee) noexcept {
    if (call == nullptr || callee == nullptr ||
        call->type() != callee->type() ||
        call->argument_count() != callee->arguments().count_size() ||
        callee->definition() == nullptr) {
        return false;
    }
    auto index = size_t{0u};
    for (auto *formal : callee->arguments()) {
        auto *actual = call->argument(index++);
        if (formal == nullptr || actual == nullptr ||
            formal->type() != actual->type()) {
            return false;
        }
        if (formal->is_resource()) {
            if (!actual->isa<xir::ResourceArgument>() ||
                actual->is_lvalue()) {
                return false;
            }
        } else if (formal->is_reference()) {
            if (!actual->is_lvalue() || actual->isa<xir::BasicBlock>() ||
                actual->isa<xir::Function>()) {
                return false;
            }
        } else if (actual->is_lvalue() || actual->type()->is_resource() ||
                   actual->isa<xir::BasicBlock>() ||
                   actual->isa<xir::Function>()) {
            return false;
        }
    }
    auto return_count = size_t{0u};
    auto closure = plan_spirv_codegen_structural_closure(
        callee->definition());
    if (!closure.succeeded()) { return false; }
    for (auto *block : closure.blocks) {
        for (auto *instruction : block->instructions()) {
            if (!instruction->isa<xir::ReturnInst>()) { continue; }
            auto *return_inst =
                static_cast<const xir::ReturnInst *>(instruction);
            auto *value = return_inst->return_value();
            if ((call->type() == nullptr) != (value == nullptr) ||
                (value != nullptr && value->type() != call->type())) {
                return false;
            }
            return_count++;
        }
    }
    return call->type() == nullptr || return_count != 0u;
}

[[nodiscard]] luisa::unordered_set<xir::Function *>
find_recursive_callables(xir::Module *module) noexcept {
    luisa::vector<xir::Function *> callables;
    luisa::unordered_set<xir::Function *> callable_set;
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() ==
            xir::DerivedFunctionTag::CALLABLE) {
            callables.emplace_back(function);
            callable_set.emplace(function);
        }
    }
    luisa::unordered_map<xir::Function *, luisa::vector<xir::Function *>> edges;
    for (auto *function : callables) {
        auto closure = plan_spirv_codegen_structural_closure(
            function->definition());
        if (!closure.succeeded()) { continue; }
        for (auto *block : closure.blocks) {
            block->traverse_instructions(
                [&](const xir::Instruction *instruction) noexcept {
                    if (!instruction->isa<xir::CallInst>()) { return; }
                    auto *callee = const_cast<xir::Function *>(
                        static_cast<const xir::CallInst *>(instruction)->callee());
                    if (callable_set.contains(callee)) {
                        edges[function].emplace_back(callee);
                    }
                });
        }
    }
    luisa::unordered_set<xir::Function *> recursive;
    for (auto *start : callables) {
        luisa::unordered_set<xir::Function *> visited;
        luisa::vector<xir::Function *> worklist{start};
        while (!worklist.empty()) {
            auto *function = worklist.back();
            worklist.pop_back();
            if (!visited.emplace(function).second) { continue; }
            for (auto *callee : edges[function]) {
                if (callee == start) {
                    recursive.emplace(start);
                    worklist.clear();
                    break;
                }
                worklist.emplace_back(callee);
            }
        }
    }
    return recursive;
}

void accumulate_inline_info(
    xir::InlineInfo &total, const xir::InlineInfo &increment) noexcept {
    total.inlined_call_count += increment.inlined_call_count;
    total.removed_callable_count += increment.removed_callable_count;
    total.skipped_recursive_callable_count +=
        increment.skipped_recursive_callable_count;
    total.skipped_structured_call_count +=
        increment.skipped_structured_call_count;
    total.rejected_malformed_call_count +=
        increment.rejected_malformed_call_count;
}

}// namespace

SpirvCallableReferenceActualValidation
validate_spirv_callable_reference_actual(
    const xir::Value *actual) noexcept {
    using Status = SpirvCallableReferenceActualStatus;
    if (actual == nullptr) {
        return {
            .status = Status::NULL_ACTUAL,
            .diagnostic = "the reference actual is null"};
    }
    if (actual->isa<xir::ReferenceArgument>()) {
        return {};
    }
    if (actual->isa<xir::AllocaInst>()) {
        auto *alloca = static_cast<const xir::AllocaInst *>(actual);
        if (alloca->is_local()) { return {}; }
        if (alloca->is_shared()) {
            return {
                .status = Status::SHARED_ALLOCATION,
                .diagnostic =
                    "a shared/workgroup allocation cannot be passed through "
                    "a Function-storage callable parameter"};
        }
    }
    if (actual->isa<xir::GEPInst>()) {
        return {
            .status = Status::DERIVED_POINTER,
            .diagnostic =
                "a GEP-derived pointer cannot be passed through a callable "
                "parameter without VariablePointers"};
    }
    return {
        .status = Status::UNSUPPORTED_VALUE,
        .diagnostic =
            "the actual must be a forwarded reference argument or a direct "
            "function-local allocation"};
}

SpirvPointerLegalizationInfo
legalize_spirv_pointer_arguments(xir::Module *module) noexcept {
    SpirvPointerLegalizationInfo result;
    if (module == nullptr) {
        result.status =
            SpirvPointerLegalizationStatus::INLINE_RETRY_FAILED;
        result.diagnostic =
            "SPIR-V pointer-argument legalization received a null module.";
        return result;
    }

    auto analyze_argument_usage = [&]() noexcept {
        SpirvFunctionArgumentAnalysisStatistics statistics;
        auto usage = analyze_spirv_function_argument_usage(
            module, &statistics);
        ++result.argument_usage_analysis_count;
        result.argument_usage_structural_closure_count +=
            statistics.structural_closure_count;
        result.argument_usage_instruction_scan_count +=
            statistics.instruction_scan_count;
        result.argument_usage_call_dependency_count +=
            statistics.call_dependency_count;
        result.argument_usage_worklist_pop_count +=
            statistics.worklist_pop_count;
        result.argument_usage_dependency_visit_count +=
            statistics.dependency_visit_count;
        return usage;
    };

    luisa::unordered_set<xir::Function *> blocking_functions_seen;
    for (;;) {
        auto usage = analyze_argument_usage();
        auto readonly_resource_origins =
            analyze_spirv_readonly_resource_origins(
                module, usage);
        auto pointer_calls = collect_pointer_calls(
            module, usage, readonly_resource_origins);
        if (pointer_calls.empty()) { break; }
        result.planned_pointer_call_count += pointer_calls.size();

        auto recursive = find_recursive_callables(module);
        luisa::unordered_set<xir::Function *> recursive_callees;
        auto malformed_count = size_t{0u};
        for (auto &&pointer_call : pointer_calls) {
            malformed_count +=
                call_shape_is_valid(pointer_call.call,
                                    pointer_call.callee) ?
                    0u :
                    1u;
            if (recursive.contains(pointer_call.callee)) {
                recursive_callees.emplace(pointer_call.callee);
            }
        }
        if (malformed_count != 0u || !recursive_callees.empty()) {
            result.inline_info.rejected_malformed_call_count +=
                malformed_count;
            result.inline_info.skipped_recursive_callable_count +=
                recursive_callees.size();
            result.remaining_pointer_call_count = pointer_calls.size();
            result.status =
                SpirvPointerLegalizationStatus::INLINE_RETRY_FAILED;
            result.diagnostic = luisa::format(
                "SPIR-V pointer-argument inline retry failed "
                "(remaining={}, structured=0, malformed={}, recursive={}).",
                result.remaining_pointer_call_count, malformed_count,
                recursive_callees.size());
            return result;
        }

        luisa::unordered_set<xir::Function *> blocking_set;
        for (auto &&pointer_call : pointer_calls) {
            auto *callee_definition = pointer_call.callee->definition();
            if (has_single_owned_block(callee_definition)) { continue; }
            auto *caller = pointer_call.call->parent_function();
            if (inspect_structured_control_flow(
                    caller == nullptr ? nullptr : caller->definition())
                    .contains_any()) {
                blocking_set.emplace(caller);
            }
            if (inspect_structured_control_flow(callee_definition)
                    .contains_any()) {
                blocking_set.emplace(pointer_call.callee);
            }
        }

        luisa::vector<xir::Function *> blocking_functions;
        auto unsupported_count = size_t{0u};
        auto rejected_destructure_count = size_t{0u};
        for (auto *function : module->function_list()) {
            if (!blocking_set.contains(function)) { continue; }
            blocking_functions.emplace_back(function);
            blocking_functions_seen.emplace(function);
            auto inventory =
                inspect_structured_control_flow(function->definition());
            unsupported_count += inventory.unsupported_count;
            rejected_destructure_count +=
                destructure_boundary_is_valid(function) ? 0u : 1u;
        }
        result.blocking_function_count = blocking_functions_seen.size();
        if (unsupported_count != 0u) {
            result.status = SpirvPointerLegalizationStatus::
                UNSUPPORTED_STRUCTURED_CONTROL_FLOW;
            result.remaining_pointer_call_count = pointer_calls.size();
            result.diagnostic = luisa::format(
                "SPIR-V pointer-argument fallback expected only SwitchInst "
                "after destructure_cfg, but found {} other structured "
                "instruction(s) across {} blocking function(s).",
                unsupported_count, blocking_functions.size());
            return result;
        }
        if (rejected_destructure_count != 0u) {
            result.status =
                SpirvPointerLegalizationStatus::DESTRUCTURE_FAILED;
            result.remaining_pointer_call_count = pointer_calls.size();
            result.diagnostic = luisa::format(
                "SPIR-V pointer-argument fallback rejected {} blocking "
                "function(s) during atomic destructure preflight; the module "
                "was left unchanged.",
                rejected_destructure_count);
            return result;
        }

        for (auto *function : blocking_functions) {
            auto destructured =
                xir::destructure_cfg_pass_run_on_function(function);
            result.destructured_switch_count +=
                destructured.destructured_switch_count;
            if (!destructured.succeeded()) {
                result.status =
                    SpirvPointerLegalizationStatus::DESTRUCTURE_FAILED;
                auto remaining_usage = analyze_argument_usage();
                auto remaining_readonly_resource_origins =
                    analyze_spirv_readonly_resource_origins(
                        module, remaining_usage);
                result.remaining_pointer_call_count =
                    collect_pointer_calls(
                        module, remaining_usage,
                        remaining_readonly_resource_origins)
                        .size();
                result.diagnostic = luisa::format(
                    "SPIR-V pointer-argument fallback could not destructure a "
                    "blocking function (errors={}, leaked_blocks={}).",
                    destructured.error_count,
                    destructured.leaked_block_count);
                return result;
            }
            result.destructured_blocking_function_count++;
        }

        luisa::vector<xir::CallInst *> call_sites;
        call_sites.reserve(pointer_calls.size());
        for (auto &&pointer_call : pointer_calls) {
            call_sites.emplace_back(pointer_call.call);
        }
        auto inline_info = xir::inline_call_sites_pass_run_on_module(
            module, luisa::span{call_sites});
        accumulate_inline_info(result.inline_info, inline_info);
        if (inline_info.inlined_call_count == 0u ||
            inline_info.skipped_structured_call_count != 0u ||
            inline_info.rejected_malformed_call_count != 0u ||
            inline_info.skipped_recursive_callable_count != 0u) {
            auto remaining_usage = analyze_argument_usage();
            auto remaining_readonly_resource_origins =
                analyze_spirv_readonly_resource_origins(
                    module, remaining_usage);
            result.remaining_pointer_call_count =
                collect_pointer_calls(
                    module, remaining_usage,
                    remaining_readonly_resource_origins)
                    .size();
            result.status =
                SpirvPointerLegalizationStatus::INLINE_RETRY_FAILED;
            result.diagnostic = luisa::format(
                "SPIR-V pointer-argument inline retry failed "
                "(remaining={}, structured={}, malformed={}, recursive={}).",
                result.remaining_pointer_call_count,
                result.inline_info.skipped_structured_call_count,
                result.inline_info.rejected_malformed_call_count,
                result.inline_info.skipped_recursive_callable_count);
            return result;
        }
    }
    result.remaining_pointer_call_count = 0u;
    return result;
}

}// namespace lc::spirv
