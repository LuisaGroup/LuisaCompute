#pragma once
#include <stdint.h>
#include <math.h>

#pragma once
#include "detail/c11.inc"

// Vector
typedef struct alignas(8) {
    float x, y;
} float2;
typedef struct alignas(16) {
    float x, y, z;
} float3;
typedef struct alignas(16) {
    float x, y, z, w;
} float4;
typedef struct alignas(16) {
    double x, y;
} double2;
typedef struct alignas(16) {
    double x, y, z;
} double3;
typedef struct alignas(16) {
    double x, y, z, w;
} double4;
typedef struct alignas(16) {
    uint64_t x, y;
} uint64_t2;
typedef struct alignas(16) {
    uint64_t x, y, z;
} uint64_t3;
typedef struct alignas(16) {
    uint64_t x, y, z, w;
} uint64_t4;
typedef struct alignas(16) {
    int64_t x, y;
} int64_t2;
typedef struct alignas(16) {
    int64_t x, y, z;
} int64_t3;
typedef struct alignas(16) {
    int64_t x, y, z, w;
} int64_t4;
typedef struct alignas(8) {
    uint32_t x, y;
} uint32_t2;
typedef struct alignas(16) {
    uint32_t x, y, z;
} uint32_t3;
typedef struct alignas(16) {
    uint32_t x, y, z, w;
} uint32_t4;
typedef struct alignas(8) {
    int32_t x, y;
} int32_t2;
typedef struct alignas(16) {
    int32_t x, y, z;
} int32_t3;
typedef struct alignas(16) {
    int32_t x, y, z, w;
} int32_t4;
typedef struct alignas(4) {
    uint16_t x, y;
} uint16_t2;
typedef struct alignas(8) {
    uint16_t x, y, z;
} uint16_t3;
typedef struct alignas(8) {
    uint16_t x, y, z, w;
} uint16_t4;
typedef struct alignas(4) {
    int16_t x, y;
} int16_t2;
typedef struct alignas(8) {
    int16_t x, y, z;
} int16_t3;
typedef struct alignas(8) {
    int16_t x, y, z, w;
} int16_t4;
typedef struct alignas(2) {
    uint8_t x, y;
} uint8_t2;
typedef struct alignas(4) {
    uint8_t x, y, z;
} uint8_t3;
typedef struct alignas(4) {
    uint8_t x, y, z, w;
} uint8_t4;
typedef struct alignas(2) {
    int8_t x, y;
} int8_t2;
typedef struct alignas(4) {
    int8_t x, y, z;
} int8_t3;
typedef struct alignas(4) {
    int8_t x, y, z, w;
} int8_t4;
typedef struct alignas(2) {
    bool x, y;
} bool2;
typedef struct alignas(4) {
    bool x, y, z;
} bool3;
typedef struct alignas(4) {
    bool x, y, z, w;
} bool4;
// Matrix
typedef struct alignas(8) {
    float2 c0, c1;
} float2x2;
typedef struct alignas(16) {
    float3 c0, c1, c2;
} float3x3;
typedef struct alignas(16) {
    float4 c0, c1, c2, c3;
} float4x4;

typedef struct {
    uint64_t ptr;
    uint64_t len;
} buffer_type;

inline float2x2 make_float2x2_0(float2 a, float2 b) {
    return (float2x2){a, b};
}
inline float2x2 make_float2x2_1(float a0, float a1, float a2, float a3) {
    float2x2 f;
    f.c0 = (float2){a0, a1};
    f.c1 = (float2){a2, a3};
    return f;
}
inline float3x3 make_float3x3_0(float3 a, float3 b, float3 c) {
    return (float3x3){a, b, c};
}
inline float3x3 make_float3x3_1(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
    float3x3 f;
    f.c0 = (float3){a0, a1, a2};
    f.c1 = (float3){a3, a4, a5};
    f.c2 = (float3){a6, a7, a8};
    return f;
}
inline float4x4 make_float4x4_0(float4 a, float4 b, float4 c, float4 d) {
    return (float4x4){a, b, c, d};
}
inline float4x4 make_float4x4_1(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15) {
    float4x4 f;
    f.c0 = (float4){a0, a1, a2, a3};
    f.c1 = (float4){a4, a5, a6, a7};
    f.c2 = (float4){a8, a9, a10, a11};
    f.c3 = (float4){a12, a13, a14, a15};
    return f;
}
inline float2 mul_float2x2_float2(float2x2 m, float2 v) {
    float2 r;
    r.x = v.x * m.c0.x + v.y * m.c1.x;
    r.y = v.x * m.c0.y + v.y * m.c1.y;
    return r;
}
inline float3 mul_float3x3_float3(float3x3 m, float3 v) {
    float3 r;
    r.x = v.x * m.c0.x + v.y * m.c1.x + v.z * m.c2.x;
    r.y = v.x * m.c0.y + v.y * m.c1.y + v.z * m.c2.y;
    r.z = v.x * m.c0.z + v.y * m.c1.z + v.z * m.c2.z;
    return r;
}
inline float4 mul_float4x4_float4(float4x4 m, float4 v) {
    float4 r;
    r.x = v.x * m.c0.x + v.y * m.c1.x + v.z * m.c2.x + v.w * m.c3.x;
    r.y = v.x * m.c0.y + v.y * m.c1.y + v.z * m.c2.y + v.w * m.c3.y;
    r.z = v.x * m.c0.z + v.y * m.c1.z + v.z * m.c2.z + v.w * m.c3.z;
    r.w = v.x * m.c0.w + v.y * m.c1.w + v.z * m.c2.w + v.w * m.c3.w;
    return r;
}
inline float2x2 mul_float2x2_float2x2(float2x2 a, float2x2 b) {
    float2x2 r;
    r.c0 = mul_float2x2_float2(a, b.c0);
    r.c1 = mul_float2x2_float2(a, b.c1);
    return r;
}
inline float3x3 mul_float3x3_float3x3(float3x3 a, float3x3 b) {
    float3x3 r;
    r.c0 = mul_float3x3_float3(a, b.c0);
    r.c1 = mul_float3x3_float3(a, b.c1);
    r.c2 = mul_float3x3_float3(a, b.c2);
    return r;
}
inline float4x4 mul_float4x4_float4x4(float4x4 a, float4x4 b) {
    float4x4 r;
    r.c0 = mul_float4x4_float4(a, b.c0);
    r.c1 = mul_float4x4_float4(a, b.c1);
    r.c2 = mul_float4x4_float4(a, b.c2);
    r.c3 = mul_float4x4_float4(a, b.c3);
    return r;
}
inline buffer_type to_buffer(uint64_t ptr, uint64_t len) {
    return (buffer_type){ptr, len};
}
#define GET(ELE, value, index) (((ELE *)&(value))[index])
#define ADDR_OF(value) ((uint64_t) & (value))
#define CAST_BF(x) (x)

inline float determinant_float2x2(float2x2 m) {
    return m.c0.x * m.c1.y - m.c1.x * m.c0.y;
}
inline float determinant_float3x3(float3x3 m) {
    return m.c0.x * (m.c1.y * m.c2.z - m.c2.y * m.c1.z) - m.c1.x * (m.c0.y * m.c2.z - m.c2.y * m.c0.z) + m.c2.x * (m.c0.y * m.c1.z - m.c1.y * m.c0.z);
}
float determinant_float4x4(float4x4 m);
#if defined(__clang__)// Clang
#define LUISA_ASSUME(x) __builtin_assume(x)
#define LUISA_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)// MSVC
#define LUISA_ASSUME(x) __assume(x)
#define LUISA_UNREACHABLE() __assume(false)
#else// GCC
#define LUISA_UNREACHABLE() __builtin_unreachable()
#define LUISA_ASSUME(x) \
    if (!(x)) __builtin_unreachable()
#endif
void memzero(void *ptr, uint64_t size);
void memone(void *ptr, uint64_t size);
int32_t lc_memcmp(uint64_t dst, uint64_t src, uint64_t size);
void lc_memcpy(uint64_t dst, uint64_t src, uint64_t size);
void lc_memmove(uint64_t dst, uint64_t src, uint64_t size);
uint64_t persist_malloc(uint64_t size);
uint64_t temp_malloc(uint64_t size);
void persist_free(uint64_t ptr);
///////////////// implement
#include <string.h>
#include <stdlib.h>
inline float4 float4_mul(float4 a, float4 b) {
    return (float4){a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
inline float4 float4_add(float4 a, float4 b) {
    return (float4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
inline float4 float4_minus(float4 a, float4 b) {
    return (float4){a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
float determinant_float4x4(float4x4 m) {
    float coef00 = m.c2.z * m.c3.w - m.c3.z * m.c2.w;
    float coef02 = m.c1.z * m.c3.w - m.c3.z * m.c1.w;
    float coef03 = m.c1.z * m.c2.w - m.c2.z * m.c1.w;
    float coef04 = m.c2.y * m.c3.w - m.c3.y * m.c2.w;
    float coef06 = m.c1.y * m.c3.w - m.c3.y * m.c1.w;
    float coef07 = m.c1.y * m.c2.w - m.c2.y * m.c1.w;
    float coef08 = m.c2.y * m.c3.z - m.c3.y * m.c2.z;
    float coef10 = m.c1.y * m.c3.z - m.c3.y * m.c1.z;
    float coef11 = m.c1.y * m.c2.z - m.c2.y * m.c1.z;
    float coef12 = m.c2.x * m.c3.w - m.c3.x * m.c2.w;
    float coef14 = m.c1.x * m.c3.w - m.c3.x * m.c1.w;
    float coef15 = m.c1.x * m.c2.w - m.c2.x * m.c1.w;
    float coef16 = m.c2.x * m.c3.z - m.c3.x * m.c2.z;
    float coef18 = m.c1.x * m.c3.z - m.c3.x * m.c1.z;
    float coef19 = m.c1.x * m.c2.z - m.c2.x * m.c1.z;
    float coef20 = m.c2.x * m.c3.y - m.c3.x * m.c2.y;
    float coef22 = m.c1.x * m.c3.y - m.c3.x * m.c1.y;
    float coef23 = m.c1.x * m.c2.y - m.c2.x * m.c1.y;
    float4 fac0 = (float4){coef00, coef00, coef02, coef03};
    float4 fac1 = (float4){coef04, coef04, coef06, coef07};
    float4 fac2 = (float4){coef08, coef08, coef10, coef11};
    float4 fac3 = (float4){coef12, coef12, coef14, coef15};
    float4 fac4 = (float4){coef16, coef16, coef18, coef19};
    float4 fac5 = (float4){coef20, coef20, coef22, coef23};
    float4 Vec0 = (float4){m.c1.x, m.c0.x, m.c0.x, m.c0.x};
    float4 Vec1 = (float4){m.c1.y, m.c0.y, m.c0.y, m.c0.y};
    float4 Vec2 = (float4){m.c1.z, m.c0.z, m.c0.z, m.c0.z};
    float4 Vec3 = (float4){m.c1.w, m.c0.w, m.c0.w, m.c0.w};
    float4 inv0 = float4_add(float4_minus(float4_mul(Vec1, fac0), float4_mul(Vec2, fac1)), float4_mul(Vec3, fac2));
    float4 inv1 = float4_add(float4_minus(float4_mul(Vec0, fac0), float4_mul(Vec2, fac3)), float4_mul(Vec3, fac4));
    float4 inv2 = float4_add(float4_minus(float4_mul(Vec0, fac1), float4_mul(Vec1, fac3)), float4_mul(Vec3, fac5));
    float4 inv3 = float4_add(float4_minus(float4_mul(Vec0, fac2), float4_mul(Vec1, fac4)), float4_mul(Vec2, fac5));
    float4 sign_a = (float4){+1.0f, -1.0f, +1.0f, -1.0f};
    float4 sign_b = (float4){-1.0f, +1.0f, -1.0f, +1.0f};
    float4 inv_0 = float4_mul(inv0, sign_a);
    float4 inv_1 = float4_mul(inv1, sign_b);
    float4 inv_2 = float4_mul(inv2, sign_a);
    float4 inv_3 = float4_mul(inv3, sign_b);
    float4 dot0 = float4_mul(m.c0, (float4){inv_0.x, inv_1.x, inv_2.x, inv_3.x});
    return dot0.x + dot0.y + dot0.z + dot0.w;
}
void memzero(void *ptr, uint64_t size) {
    memset(ptr, 0, size);
}
void memone(void *ptr, uint64_t size) {
    memset(ptr, 1, size);
}
int32_t lc_memcmp(uint64_t dst, uint64_t src, uint64_t size) {
    return memcmp((void const *)dst, (void const *)src, size);
}
void lc_memcpy(uint64_t dst, uint64_t src, uint64_t size) {
    memcpy((void *)dst, (void const *)src, size);
}
void lc_memmove(uint64_t dst, uint64_t src, uint64_t size) {
    memmove((void *)dst, (void const *)src, size);
}
uint64_t persist_malloc(uint64_t size) {
    return (uint64_t)malloc(size);
}
uint64_t temp_malloc(uint64_t size) {
    return (uint64_t)malloc(size);
}
void persist_free(uint64_t ptr) {
    free((void *)ptr);
}