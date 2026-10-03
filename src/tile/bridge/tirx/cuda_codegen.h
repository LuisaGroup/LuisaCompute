#pragma once

#include <luisa/core/stl/string.h>

namespace luisa::compute::tile::bridge::tirx::detail {

// CUDA-source-only helpers for an explicitly selected subgroup realization.
// The caller prepends these only when such a plan was actually emitted.
// They do not register TVM callbacks or affect the ordinary/Metal paths.
[[nodiscard]] luisa::string_view native_cuda_subgroup_helpers() noexcept;

// Supply only absent TVM CUDA callbacks for native C++ clients. Existing Python
// or application registrations remain authoritative and are never replaced.
void initialize_native_cuda_codegen();

}// namespace luisa::compute::tile::bridge::tirx::detail
