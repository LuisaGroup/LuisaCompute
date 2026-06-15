#pragma once
#include <luisa/vstl/common.h>

namespace lc::vk {

// Post-processes SPIR-V binary to enable NV ray tracing motion blur.
//
// DXC cannot emit SPV_NV_ray_tracing_motion_blur instructions directly,
// so we compile with standard TraceRay (OpTraceRayKHR) and then patch
// the SPIR-V binary to:
//   1. Add OpCapability RayTracingMotionBlurNV (5341)
//   2. Add OpExtension "SPV_NV_ray_tracing_motion_blur"
//   3. Replace OpTraceRayKHR (4445) with OpTraceRayMotionNV (5339),
//      inserting the time operand between rayTmax and payload.
//
// The HLSL code stores the time value as field 4 of _MotionPayload.
// The patcher recognizes either a field store through OpAccessChain or
// a whole-payload store built from OpCompositeConstruct, and inserts
// the recovered value as the Time operand of OpTraceRayMotionNV.
//
// Returns the patched SPIR-V binary. If no OpTraceRayKHR is found,
// returns the input unchanged.
vstd::vector<uint32_t> patch_spirv_for_motion_blur(vstd::span<uint32_t const> spirv);

}// namespace lc::vk
