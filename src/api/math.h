//
// Created by Mike on 2022/1/7.
//

#pragma once

#include <core/platform.h>
        
LUISA_EXPORT_API void *luisa_compute_int2_create(int v0, int v1) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_int2_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_int3_create(int v0, int v1, int v2) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_int3_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_int4_create(int v0, int v1, int v2, int v3) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_int4_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_uint2_create(uint32_t v0, uint32_t v1) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_uint2_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_uint3_create(uint32_t v0, uint32_t v1, uint32_t v2) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_uint3_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_uint4_create(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_uint4_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_float2_create(float v0, float v1) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_float2_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_float3_create(float v0, float v1, float v2) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_float3_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_float4_create(float v0, float v1, float v2, float v3) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_float4_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_bool2_create(int v0, int v1) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bool2_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_bool3_create(int v0, int v1, int v2) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bool3_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_bool4_create(int v0, int v1, int v2, int v3) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_bool4_destroy(void *v) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_float2x2_create(
    float m00, float m01,
    float m10, float m11) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_float2x2_destroy(void *m) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_float3x3_create(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_float3x3_destroy(void *m) LUISA_NOEXCEPT;
LUISA_EXPORT_API void *luisa_compute_float4x4_create(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33) LUISA_NOEXCEPT;
LUISA_EXPORT_API void luisa_compute_float4x4_destroy(void *m) LUISA_NOEXCEPT;
