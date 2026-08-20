//
// HIP-specific transformations of LLVM pass-pipeline descriptions.
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace luisa::compute::hip {

// Attribute cleanup preserves only explicitly modeled backend boundaries.
// Ordinary generated callables are absent from this predicate by design and
// therefore retain neither inline directive after IPO.
[[nodiscard]] bool preserve_hip_backend_noinline_boundary(
    std::string_view function_name,
    bool has_noinline_attribute) noexcept;

// Replaces complete `no-keep-loops` pass-option tokens while preserving every
// other byte of the serialized pipeline. The caller owns the version-specific
// cardinality invariant for the pipeline it supplied.
[[nodiscard]] size_t preserve_hardware_ray_query_loop_form(
    std::string &pipeline) noexcept;

}// namespace luisa::compute::hip
