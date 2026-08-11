//
// HIP-specific transformations of LLVM pass-pipeline descriptions.
//

#pragma once

#include <cstddef>
#include <string>

namespace luisa::compute::hip {

// A generated callable larger than this remains a real function boundary.
// This caps the structural complexity contributed by any single callee to an
// AMDGPU kernel while leaving ordinary small-callable inlining to LLVM.
inline constexpr size_t generated_callable_inline_instruction_budget =
    500000u;

[[nodiscard]] bool preserve_generated_callable_boundary(
    size_t instruction_count) noexcept;

// Replaces complete `no-keep-loops` pass-option tokens while preserving every
// other byte of the serialized pipeline. The caller owns the version-specific
// cardinality invariant for the pipeline it supplied.
[[nodiscard]] size_t preserve_hardware_ray_query_loop_form(
    std::string &pipeline) noexcept;

}// namespace luisa::compute::hip
