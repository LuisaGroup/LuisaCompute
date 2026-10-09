// Sample native compute shader source (Vulkan, GLSL 450).
//
// The same `dst[i] = src[i] * k + c` as the HLSL sample; the (set, binding)
// pairs are spelled out, and the `layout(push_constant)` block is fed by the
// launcher's `add_uniform` values. `GL_GOOGLE_include_directive` must be
// requested after `#version` and before any `#include`.
#version 450
#extension GL_GOOGLE_include_directive : require
#include "native_shader_math.h"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer A { float a[]; } src;
layout(set = 0, binding = 1) buffer B { float b[]; } dst;
layout(push_constant) uniform Push { float k; float c; } uniforms;

void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.b[i] = luisa_native_shader_transform(src.a[i], uniforms.k, uniforms.c);
}
