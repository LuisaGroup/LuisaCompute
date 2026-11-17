#pragma once

// Host API only: do not include these headers in the embedded NVRTC/device library.
// The device library deliberately keeps its lightweight intrinsic declarations.
#include <cstddef>
#include <optix.h>
#include <optix_function_table.h>

// The selected official headers supply their own function table ABI. Keep the
// shared host/device layouts checked below instead of pinning one SDK release.
static_assert(OPTIX_VERSION >= 90000, "OptiX 9.0 or newer headers are required.");

// These layouts are also consumed by the lightweight device declarations and
// LLVM code generation. An SDK update must review both sides of this boundary.
static_assert(sizeof(OptixTraversableHandle) == 8u);
static_assert(sizeof(OptixInstance) == 80u);
static_assert(offsetof(OptixInstance, instanceId) == 48u);
static_assert(offsetof(OptixInstance, traversableHandle) == 64u);
static_assert(sizeof(OptixMotionOptions) == 12u);
static_assert(offsetof(OptixMatrixMotionTransform, transform) == 32u);
static_assert(offsetof(OptixSRTMotionTransform, srtData) == 32u);
static_assert(sizeof(OptixSRTData) == 64u);

namespace luisa::compute::optix {

[[nodiscard]] const OptixFunctionTable &api() noexcept;

// Cached, non-fatal capability check. Missing/incompatible drivers select the
// software ray-tracing fallback; api() diagnoses an explicitly required runtime.
[[nodiscard]] bool available() noexcept;

}// namespace luisa::compute::optix
