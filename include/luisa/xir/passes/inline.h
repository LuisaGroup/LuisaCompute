#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>

namespace luisa::compute::xir {

class PassReport;
class CallInst;

struct InlineInfo {
    size_t inlined_call_count{0u};
    size_t removed_callable_count{0u};
    size_t skipped_recursive_callable_count{0u};
    size_t skipped_structured_call_count{0u};
    size_t skipped_constrained_call_count{0u};
    size_t skipped_metadata_call_count{0u};
    size_t skipped_declaration_call_count{0u};
    size_t rejected_malformed_call_count{0u};
    // Diagnostic work counts for call-site preflight and application.
    // These do not participate in changed().
    size_t call_site_summary_function_count{0u};
    size_t call_site_summary_instruction_scan_count{0u};
    size_t call_site_cached_apply_count{0u};
    size_t call_site_revalidated_apply_count{0u};
    size_t call_site_clone_layout_function_count{0u};
    size_t call_site_clone_layout_value_count{0u};
    size_t call_site_dense_resolver_apply_count{0u};
    size_t call_site_dense_resolver_fallback_count{0u};
    size_t inline_pass_summary_function_count{0u};
    size_t inline_pass_summary_instruction_scan_count{0u};
    size_t inline_pass_clone_layout_function_count{0u};
    size_t inline_pass_clone_layout_value_count{0u};
    size_t inline_pass_dense_resolver_apply_count{0u};
    size_t inline_pass_dense_resolver_fallback_count{0u};
    size_t inline_pass_caller_barrier_function_count{0u};
    size_t inline_pass_caller_barrier_instruction_scan_count{0u};
    size_t inline_pass_caller_barrier_cache_hit_count{0u};
    // The recursion solver visits every callable and call edge once in each
    // direction. These counters expose that linear SCC complexity contract.
    size_t recursion_analysis_function_count{0u};
    size_t recursion_analysis_call_use_visit_count{0u};
    size_t recursion_analysis_edge_count{0u};
    size_t recursion_analysis_vertex_visit_count{0u};
    size_t recursion_analysis_edge_visit_count{0u};
    [[nodiscard]] bool changed() const noexcept {
        return inlined_call_count != 0u ||
               removed_callable_count != 0u;
    }
};

struct InlineOptions {
    bool allow_autodiff_scope_in_caller{false};
};

// Single-block callees can be inlined into structured callers without changing
// their CFG. By default, multi-block inlining is unstructured-CFG-only. The
// opt-in option permits only a retained caller-side autodiff scope after the
// caller and callee's ordinary structured CFG has already been destructured.
// Signature-constrained callees and calls whose metadata cannot be assigned to
// one replacement owner are rejected without mutation. Bodyless callable
// declarations are valid references but are never inline candidates. Callee
// instruction and basic-block metadata is cloned one-to-one;
// function/argument names are debug declarations and are not materialized into
// the inline region.
[[nodiscard]] LUISA_XIR_API InlineInfo inline_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API InlineInfo inline_all_pass_run_on_module(Module *module, PassReport *report = nullptr) noexcept;
[[nodiscard]] LUISA_XIR_API InlineInfo inline_all_pass_run_on_module(Module *module, InlineOptions options, PassReport *report = nullptr) noexcept;

[[nodiscard]] LUISA_XIR_API InlineInfo
inline_call_sites_pass_run_on_module(
    Module *module, luisa::span<CallInst *const> call_sites,
    InlineOptions options = {}, PassReport *report = nullptr) noexcept;

}// namespace luisa::compute::xir
