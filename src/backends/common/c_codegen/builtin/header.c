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
