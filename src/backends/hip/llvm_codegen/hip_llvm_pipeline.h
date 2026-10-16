//
// HIP-specific transformations of LLVM pass-pipeline descriptions.
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace luisa::compute::hip {

// Replaces complete `no-keep-loops` pass-option tokens while preserving every
// other byte of the serialized pipeline. The caller owns the version-specific
// cardinality invariant for the pipeline it supplied.
[[nodiscard]] size_t preserve_hardware_ray_query_loop_form(
    std::string &pipeline) noexcept;

}// namespace luisa::compute::hip
