#pragma once
#include <stdint.h>
#include <math.h>
// Vector
typedef struct alignas(8) {
    float x, y
} float2;
typedef struct alignas(16) {
    float x, y, z;
} float3;
typedef struct alignas(16) {
    float x, y, z, w;
} float4;
typedef struct alignas(16) {
    double x, y
} double2;
typedef struct alignas(32) {
    double x, y, z;
} double3;
typedef struct alignas(32) {
    double x, y, z, w;
} double4;
typedef struct alignas(16) {
    uint64_t x, y
} uint64_t2;
typedef struct alignas(32) {
    uint64_t x, y, z;
} uint64_t3;
typedef struct alignas(32) {
    uint64_t x, y, z, w;
} uint64_t4;
typedef struct alignas(16) {
    int64_t x, y
} int64_t2;
typedef struct alignas(32) {
    int64_t x, y, z;
} int64_t3;
typedef struct alignas(32) {
    int64_t x, y, z, w;
} int64_t4;
typedef struct alignas(8) {
    uint32_t x, y
} uint32_t2;
typedef struct alignas(16) {
    uint32_t x, y, z;
} uint32_t3;
typedef struct alignas(16) {
    uint32_t x, y, z, w;
} uint32_t4;
typedef struct alignas(8) {
    int32_t x, y
} int32_t2;
typedef struct alignas(16) {
    int32_t x, y, z;
} int32_t3;
typedef struct alignas(16) {
    int32_t x, y, z, w;
} int32_t4;
typedef struct alignas(4) {
    uint16_t x, y
} uint16_t2;
typedef struct alignas(8) {
    uint16_t x, y, z;
} uint16_t3;
typedef struct alignas(8) {
    uint16_t x, y, z, w;
} uint16_t4;
typedef struct alignas(4) {
    int16_t x, y
} int16_t2;
typedef struct alignas(8) {
    int16_t x, y, z;
} int16_t3;
typedef struct alignas(8) {
    int16_t x, y, z, w;
} int16_t4;
typedef struct alignas(2) {
    uint8_t x, y
} uint8_t2;
typedef struct alignas(4) {
    uint8_t x, y, z;
} uint8_t3;
typedef struct alignas(4) {
    uint8_t x, y, z, w;
} uint8_t4;
typedef struct alignas(2) {
    int8_t x, y
} int8_t2;
typedef struct alignas(4) {
    int8_t x, y, z;
} int8_t3;
typedef struct alignas(4) {
    int8_t x, y, z, w;
} int8_t4;
typedef struct alignas(2) {
    bool x, y
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

static float2x2 make_float2x2_0(float2 a, float2 b) {
    return (float2x2){a, b};
}
static float2x2 make_float2x2_1(float a0, float a1, float a2, float a3) {
    float2x2 f;
    f.c0 = (float2){a0, a1};
    f.c1 = (float2){a2, a3};
    return f;
}
static float3x3 make_float3x3_0(float3 a, float3 b, float3 c) {
    return (float3x3){a, b, c};
}
static float3x3 make_float3x3_1(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8) {
    float3x3 f;
    f.c0 = (float3){a0, a1, a2};
    f.c1 = (float3){a3, a4, a5};
    f.c2 = (float3){a6, a7, a8};
    return f;
}
static float4x4 make_float4x4_0(float4 a, float4 b, float4 c, float4 d) {
    return (float4x4){a, b, c, d};
}
static float4x4 make_float4x4_1(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15) {
    float4x4 f;
    f.c0 = (float4){a0, a1, a2, a3};
    f.c1 = (float4){a4, a5, a6, a7};
    f.c2 = (float4){a8, a9, a10, a11};
    f.c3 = (float4){a12, a13, a14, a15};
    return f;
}