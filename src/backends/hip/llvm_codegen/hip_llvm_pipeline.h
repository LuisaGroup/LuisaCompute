//
// HIP-specific transformations of LLVM pass-pipeline descriptions.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace luisa::compute::hip {

// A generated callable larger than this remains a real function boundary.
// This caps the structural complexity contributed by any single callee to an
// AMDGPU kernel while leaving ordinary small-callable inlining to LLVM.
inline constexpr size_t generated_callable_inline_instruction_budget =
    500000u;

[[nodiscard]] bool preserve_generated_callable_boundary(
    size_t instruction_count) noexcept;

// A compact, backend-independent model of the generated-callable call graph.
// `callees` contains one entry per static direct call site. Each entry in an
// alternative group is an index into `callees`; the corresponding call sites
// are guaranteed by the LLVM adapter to lie in mutually exclusive CFG
// successors. Repeated references to one shared callee therefore need only one
// outlined function body, although the residual expansion recurrence still
// counts every static call site outside that frontier.
struct GeneratedCallableInlineGraphNode {
    size_t instruction_count{};
    std::vector<size_t> callees;
    std::vector<std::vector<size_t>> alternative_call_groups;
};

// Selects the generated callables that must remain real function boundaries.
// For an acyclic node v its residual fully-inlined size is modeled as
//
//   R(v) = I(v) + sum_site(max(R(callee(site)), 1) - 1),
//
// except that a preserved callee contributes only its already-counted call
// instruction. When R(v) exceeds the budget, the selector preserves every
// callee on one proven mutually-exclusive dispatch frontier. Keeping the
// frontier whole is intentional: partially inlining a polymorphic frontier
// retains most of the caller's code and register pressure while adding an ABI
// boundary to only some cases. Among available frontiers the one with greatest
// modeled expansion is selected. Size arithmetic saturates only at size_t's
// representable maximum, so candidates above the budget remain ordered.
// Residuals and choices are recomputed to a monotone fixed point. Recursive
// SCCs are boundaries by construction because their fully-inlined expansion
// is not finite.
[[nodiscard]] std::vector<std::uint8_t>
select_generated_callable_boundaries(
    std::span<const GeneratedCallableInlineGraphNode> graph,
    size_t instruction_budget =
        generated_callable_inline_instruction_budget) noexcept;

// Replaces complete `no-keep-loops` pass-option tokens while preserving every
// other byte of the serialized pipeline. The caller owns the version-specific
// cardinality invariant for the pipeline it supplied.
[[nodiscard]] size_t preserve_hardware_ray_query_loop_form(
    std::string &pipeline) noexcept;

}// namespace luisa::compute::hip
