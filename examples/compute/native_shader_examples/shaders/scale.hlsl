// Sample native compute shader source (DirectX / Vulkan, HLSL).
//
// `dst[i] = src[i] * k + c` with a structured buffer pair and a uniform block,
// fed by the launcher's `add_uniform` values:
//   * dx: the `cbuffer Uniforms : register(b0)` block is the root 32-bit
//     constant block (`push_constant_size = 8`).
//   * vk: the same block, bound as the push-constant range.
//
// The body comes from `native_shader_math.h`, found through the dispatch
// document's `config.include_dirs` (and through this file's own directory when
// the source is compiled from a path).
#include "native_shader_math.h"

StructuredBuffer<float> src : register(t0);
RWStructuredBuffer<float> dst : register(u0);
cbuffer Uniforms : register(b0) { float k; float c; };

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    dst[tid.x] = luisa_native_shader_transform(src[tid.x], k, c);
}
