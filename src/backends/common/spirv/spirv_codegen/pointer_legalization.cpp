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
#include <luisa/xir/passes/unused_callable_removal.h>

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
    const SpirvUniqueResourceOriginMap
        &unique_resource_origins) noexcept {
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
            // A complete equal-origin proof specializes every descriptor and
            // resource-specific side channel to the same kernel binding. The
            // resource therefore does not cross the callable ABI at all.
            if (unique_resource_origins.contains(formal)) { continue; }
            if (type->is_buffer() || type->is_bindless_array()) {
                return true;
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
    luisa::span<const xir::CallInst *const> call_sites,
    const SpirvFunctionArgumentAnalysisMap &usage,
    const SpirvUniqueResourceOriginMap
        &unique_resource_origins) noexcept {
    luisa::vector<PointerCall> calls;
    calls.reserve(call_sites.size());
    for (auto *const_call : call_sites) {
        auto *call = const_cast<xir::CallInst *>(const_call);
        if (call == nullptr ||
            !usage.contains(call->parent_function())) {
            continue;
        }
        auto *callee = call->callee();
        if (callee == nullptr || !usage.contains(callee) ||
            callee->definition() == nullptr ||
            callee->derived_function_tag() !=
                xir::DerivedFunctionTag::CALLABLE) {
            continue;
        }
        if (call_requires_specialization(
                call, callee, usage,
                unique_resource_origins)) {
            calls.emplace_back(PointerCall{call, callee});
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
find_recursive_callables(
    xir::Module *module,
    luisa::span<const xir::CallInst *const> call_sites) noexcept {
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
    for (auto *call : call_sites) {
        if (call == nullptr) { continue; }
        auto *caller = const_cast<xir::Function *>(
            call->parent_function());
        auto *callee = const_cast<xir::Function *>(call->callee());
        if (callable_set.contains(caller) &&
            callable_set.contains(callee)) {
            edges[caller].emplace_back(callee);
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
    total.skipped_constrained_call_count +=
        increment.skipped_constrained_call_count;
    total.skipped_noinline_call_count +=
        increment.skipped_noinline_call_count;
    total.skipped_metadata_call_count +=
        increment.skipped_metadata_call_count;
    total.consumed_call_site_diagnostic_metadata_count +=
        increment.consumed_call_site_diagnostic_metadata_count;
    total.skipped_declaration_call_count +=
        increment.skipped_declaration_call_count;
    total.rejected_malformed_call_count +=
        increment.rejected_malformed_call_count;
    total.skipped_costly_callable_count +=
        increment.skipped_costly_callable_count;
    total.call_site_summary_function_count +=
        increment.call_site_summary_function_count;
    total.call_site_summary_instruction_scan_count +=
        increment.call_site_summary_instruction_scan_count;
    total.call_site_cached_apply_count +=
        increment.call_site_cached_apply_count;
    total.call_site_revalidated_apply_count +=
        increment.call_site_revalidated_apply_count;
    total.call_site_clone_layout_function_count +=
        increment.call_site_clone_layout_function_count;
    total.call_site_clone_layout_value_count +=
        increment.call_site_clone_layout_value_count;
    total.call_site_dense_resolver_apply_count +=
        increment.call_site_dense_resolver_apply_count;
    total.call_site_dense_resolver_fallback_count +=
        increment.call_site_dense_resolver_fallback_count;
    total.inline_pass_summary_function_count +=
        increment.inline_pass_summary_function_count;
    total.inline_pass_summary_instruction_scan_count +=
        increment.inline_pass_summary_instruction_scan_count;
    total.inline_pass_clone_layout_function_count +=
        increment.inline_pass_clone_layout_function_count;
    total.inline_pass_clone_layout_value_count +=
        increment.inline_pass_clone_layout_value_count;
    total.inline_pass_dense_resolver_apply_count +=
        increment.inline_pass_dense_resolver_apply_count;
    total.inline_pass_dense_resolver_fallback_count +=
        increment.inline_pass_dense_resolver_fallback_count;
    total.inline_pass_caller_barrier_function_count +=
        increment.inline_pass_caller_barrier_function_count;
    total.inline_pass_caller_barrier_instruction_scan_count +=
        increment.inline_pass_caller_barrier_instruction_scan_count;
    total.inline_pass_caller_barrier_cache_hit_count +=
        increment.inline_pass_caller_barrier_cache_hit_count;
    total.recursion_analysis_function_count +=
        increment.recursion_analysis_function_count;
    total.recursion_analysis_call_use_visit_count +=
        increment.recursion_analysis_call_use_visit_count;
    total.recursion_analysis_edge_count +=
        increment.recursion_analysis_edge_count;
    total.recursion_analysis_vertex_visit_count +=
        increment.recursion_analysis_vertex_visit_count;
    total.recursion_analysis_edge_visit_count +=
        increment.recursion_analysis_edge_visit_count;
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

    struct AnalysisSnapshot {
        SpirvFunctionArgumentAnalysisMap usage;
        SpirvUniqueResourceOriginMap unique_resource_origins;
        SpirvFunctionCallSiteList call_sites;
    };
    auto analyze_argument_flow = [&]() noexcept {
        SpirvFunctionArgumentAnalysisStatistics statistics;
        AnalysisSnapshot snapshot;
        snapshot.usage = analyze_spirv_function_argument_usage(
            module, &statistics,
            {.kernel_reachable_only = true},
            &snapshot.call_sites);
        snapshot.unique_resource_origins =
            analyze_spirv_unique_resource_origins_from_call_sites(
                snapshot.usage,
                luisa::span{snapshot.call_sites});
        ++result.argument_usage_analysis_count;
        result.indexed_call_site_count +=
            snapshot.call_sites.size();
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
        return snapshot;
    };

    luisa::unordered_set<xir::Function *> blocking_functions_seen;
    for (;;) {
        auto analysis = analyze_argument_flow();
        auto pointer_calls = collect_pointer_calls(
            luisa::span{analysis.call_sites},
            analysis.usage,
            analysis.unique_resource_origins);
        if (pointer_calls.empty()) { break; }
        result.planned_pointer_call_count += pointer_calls.size();

        auto recursive = find_recursive_callables(
            module, luisa::span{analysis.call_sites});
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
                auto remaining = analyze_argument_flow();
                result.remaining_pointer_call_count =
                    collect_pointer_calls(
                        luisa::span{remaining.call_sites},
                        remaining.usage,
                        remaining.unique_resource_origins)
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
            module, luisa::span{call_sites},
            {.consume_call_site_diagnostic_metadata = true,
             .override_noinline = true});
        accumulate_inline_info(result.inline_info, inline_info);
        if (inline_info.inlined_call_count == 0u ||
            inline_info.skipped_structured_call_count != 0u ||
            inline_info.skipped_constrained_call_count != 0u ||
            inline_info.skipped_noinline_call_count != 0u ||
            inline_info.skipped_metadata_call_count != 0u ||
            inline_info.skipped_declaration_call_count != 0u ||
            inline_info.rejected_malformed_call_count != 0u ||
            inline_info.skipped_recursive_callable_count != 0u) {
            auto remaining = analyze_argument_flow();
            result.remaining_pointer_call_count =
                collect_pointer_calls(
                    luisa::span{remaining.call_sites},
                    remaining.usage,
                    remaining.unique_resource_origins)
                    .size();
            result.status =
                SpirvPointerLegalizationStatus::INLINE_RETRY_FAILED;
            result.diagnostic = luisa::format(
                "SPIR-V pointer-argument inline retry failed "
                "(remaining={}, structured={}, malformed={}, recursive={}, "
                "constrained={}, noinline={}, metadata={}, declaration={}).",
                result.remaining_pointer_call_count,
                result.inline_info.skipped_structured_call_count,
                result.inline_info.rejected_malformed_call_count,
                result.inline_info.skipped_recursive_callable_count,
                result.inline_info.skipped_constrained_call_count,
                result.inline_info.skipped_noinline_call_count,
                result.inline_info.skipped_metadata_call_count,
                result.inline_info.skipped_declaration_call_count);
            return result;
        }

        // Pointer legalization is a fixed point over the semantic call graph
        // reachable from kernel roots. The argument/resource analyses above
        // explicitly project onto that domain, including when an orphan block
        // still owns a physical function operand. Inlining a wrapper can then
        // drop the last physical use of a callable that has left the semantic
        // domain. Remove such definitions at the mutation boundary so later
        // whole-module passes and SPIR-V emission observe the same domain, and
        // so subsequent fixed-point iterations need not rescan dead bodies.
        auto pruned =
            xir::unused_callable_removal_pass_run_on_module(module);
        result.pruned_unreachable_callable_count +=
            pruned.removed_callable_count;
    }
    result.remaining_pointer_call_count = 0u;
    return result;
}

}// namespace lc::spirv
