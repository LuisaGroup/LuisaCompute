#pragma once

#include <cmath>

using lc_int = int;
using lc_uint = unsigned int;
using lc_float = float;
using lc_bool = bool;

struct alignas(8) lc_int2 {
    lc_int x, y;
    explicit constexpr lc_int2(int s) noexcept
        : x{s}, y{s} {}
    constexpr lc_int2(lc_int x, lc_int y) noexcept
        : x{x}, y{y} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_int3 {
    lc_int x, y, z;
    explicit constexpr lc_int3(int s) noexcept
        : x{s}, y{s}, z{s} {}
    constexpr lc_int3(lc_int x, lc_int y, lc_int z) noexcept
        : x{x}, y{y}, z{z} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_int4 {
    lc_int x, y, z, w;
    explicit constexpr lc_int4(int s) noexcept
        : x{s}, y{s}, z{s}, w{s} {}
    constexpr lc_int4(lc_int x, lc_int y, lc_int z, lc_int w) noexcept
        : x{x}, y{y}, z{z}, w{w} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(8) lc_uint2 {
    lc_uint x, y;
    explicit constexpr lc_uint2(uint s) noexcept
        : x{s}, y{s} {}
    constexpr lc_uint2(lc_uint x, lc_uint y) noexcept
        : x{x}, y{y} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_uint3 {
    lc_uint x, y, z;
    explicit constexpr lc_uint3(uint s) noexcept
        : x{s}, y{s}, z{s} {}
    constexpr lc_uint3(lc_uint x, lc_uint y, lc_uint z) noexcept
        : x{x}, y{y}, z{z} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_uint4 {
    lc_uint x, y, z, w;
    explicit constexpr lc_uint4(uint s) noexcept
        : x{s}, y{s}, z{s}, w{s} {}
    constexpr lc_uint4(lc_uint x, lc_uint y, lc_uint z, lc_uint w) noexcept
        : x{x}, y{y}, z{z}, w{w} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(8) lc_float2 {
    lc_float x, y;
    explicit constexpr lc_float2(float s) noexcept
        : x{s}, y{s} {}
    constexpr lc_float2(lc_float x, lc_float y) noexcept
        : x{x}, y{y} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_float3 {
    lc_float x, y, z;
    explicit constexpr lc_float3(float s) noexcept
        : x{s}, y{s}, z{s} {}
    constexpr lc_float3(lc_float x, lc_float y, lc_float z) noexcept
        : x{x}, y{y}, z{z} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_float4 {
    lc_float x, y, z, w;
    explicit constexpr lc_float4(float s) noexcept
        : x{s}, y{s}, z{s}, w{s} {}
    constexpr lc_float4(lc_float x, lc_float y, lc_float z, lc_float w) noexcept
        : x{x}, y{y}, z{z}, w{w} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(8) lc_bool2 {
    lc_bool x, y;
    explicit constexpr lc_bool2(bool s) noexcept
        : x{s}, y{s} {}
    constexpr lc_bool2(lc_bool x, lc_bool y) noexcept
        : x{x}, y{y} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_bool3 {
    lc_bool x, y, z;
    explicit constexpr lc_bool3(bool s) noexcept
        : x{s}, y{s}, z{s} {}
    constexpr lc_bool3(lc_bool x, lc_bool y, lc_bool z) noexcept
        : x{x}, y{y}, z{z} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

struct alignas(16) lc_bool4 {
    lc_bool x, y, z, w;
    explicit constexpr lc_bool4(bool s) noexcept
        : x{s}, y{s}, z{s}, w{s} {}
    constexpr lc_bool4(lc_bool x, lc_bool y, lc_bool z, lc_bool w) noexcept
        : x{x}, y{y}, z{z}, w{w} {}
    constexpr auto &operator[](lc_uint i) noexcept { return (&x)[i]; }
    constexpr auto operator[](lc_uint i) const noexcept { return (&x)[i]; }
};

[[nodiscard]] constexpr auto lc_make_int2(lc_int s) noexcept { return lc_int2{s, s}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_int x, lc_int y) noexcept { return lc_int2{x, y}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_int2 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_int3 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_int4 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_uint2 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_uint3 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_uint4 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_float2 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_float3 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_float4 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_bool2 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_bool3 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int2(lc_bool4 v) noexcept { return lc_int2{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_int s) noexcept { return lc_int3{s, s, s}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_int x, lc_int y, lc_int z) noexcept { return lc_int3{x, y, z}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_int x, lc_int2 yz) noexcept { return lc_int3{x, yz.x, yz.y}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_int2 xy, lc_int z) noexcept { return lc_int3{xy.x, xy.y, z}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_int3 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_int4 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_uint3 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_uint4 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_float3 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_float4 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_bool3 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int3(lc_bool4 v) noexcept { return lc_int3{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int s) noexcept { return lc_int4{s, s, s, s}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int x, lc_int y, lc_int z, lc_int w) noexcept { return lc_int4{x, y, z, w}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int x, lc_int y, lc_int2 zw) noexcept { return lc_int4{x, y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int x, lc_int2 yz, lc_int w) noexcept { return lc_int4{x, yz.x, yz.y, w}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int2 xy, lc_int z, lc_int w) noexcept { return lc_int4{xy.x, xy.y, z, w}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int2 xy, lc_int2 zw) noexcept { return lc_int4{xy.x, xy.y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int x, lc_int3 yzw) noexcept { return lc_int4{x, yzw.x, yzw.y, yzw.z}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int3 xyz, lc_int w) noexcept { return lc_int4{xyz.x, xyz.y, xyz.z, w}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_int4 v) noexcept { return lc_int4{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z), static_cast<lc_int>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_uint4 v) noexcept { return lc_int4{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z), static_cast<lc_int>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_float4 v) noexcept { return lc_int4{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z), static_cast<lc_int>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_int4(lc_bool4 v) noexcept { return lc_int4{static_cast<lc_int>(v.x), static_cast<lc_int>(v.y), static_cast<lc_int>(v.z), static_cast<lc_int>(v.w)}; }

[[nodiscard]] constexpr auto lc_make_uint2(lc_uint s) noexcept { return lc_uint2{s, s}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_uint x, lc_uint y) noexcept { return lc_uint2{x, y}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_int2 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_int3 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_int4 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_uint2 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_uint3 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_uint4 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_float2 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_float3 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_float4 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_bool2 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_bool3 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint2(lc_bool4 v) noexcept { return lc_uint2{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_uint s) noexcept { return lc_uint3{s, s, s}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_uint x, lc_uint y, lc_uint z) noexcept { return lc_uint3{x, y, z}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_uint x, lc_uint2 yz) noexcept { return lc_uint3{x, yz.x, yz.y}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_uint2 xy, lc_uint z) noexcept { return lc_uint3{xy.x, xy.y, z}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_int3 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_int4 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_uint3 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_uint4 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_float3 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_float4 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_bool3 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint3(lc_bool4 v) noexcept { return lc_uint3{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint s) noexcept { return lc_uint4{s, s, s, s}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint x, lc_uint y, lc_uint z, lc_uint w) noexcept { return lc_uint4{x, y, z, w}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint x, lc_uint y, lc_uint2 zw) noexcept { return lc_uint4{x, y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint x, lc_uint2 yz, lc_uint w) noexcept { return lc_uint4{x, yz.x, yz.y, w}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint2 xy, lc_uint z, lc_uint w) noexcept { return lc_uint4{xy.x, xy.y, z, w}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint2 xy, lc_uint2 zw) noexcept { return lc_uint4{xy.x, xy.y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint x, lc_uint3 yzw) noexcept { return lc_uint4{x, yzw.x, yzw.y, yzw.z}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint3 xyz, lc_uint w) noexcept { return lc_uint4{xyz.x, xyz.y, xyz.z, w}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_int4 v) noexcept { return lc_uint4{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z), static_cast<lc_uint>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_uint4 v) noexcept { return lc_uint4{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z), static_cast<lc_uint>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_float4 v) noexcept { return lc_uint4{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z), static_cast<lc_uint>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_uint4(lc_bool4 v) noexcept { return lc_uint4{static_cast<lc_uint>(v.x), static_cast<lc_uint>(v.y), static_cast<lc_uint>(v.z), static_cast<lc_uint>(v.w)}; }

[[nodiscard]] constexpr auto lc_make_float2(lc_float s) noexcept { return lc_float2{s, s}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_float x, lc_float y) noexcept { return lc_float2{x, y}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_int2 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_int3 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_int4 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_uint2 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_uint3 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_uint4 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_float2 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_float3 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_float4 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_bool2 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_bool3 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float2(lc_bool4 v) noexcept { return lc_float2{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_float s) noexcept { return lc_float3{s, s, s}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_float x, lc_float y, lc_float z) noexcept { return lc_float3{x, y, z}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_float x, lc_float2 yz) noexcept { return lc_float3{x, yz.x, yz.y}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_float2 xy, lc_float z) noexcept { return lc_float3{xy.x, xy.y, z}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_int3 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_int4 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_uint3 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_uint4 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_float3 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_float4 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_bool3 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float3(lc_bool4 v) noexcept { return lc_float3{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float s) noexcept { return lc_float4{s, s, s, s}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float x, lc_float y, lc_float z, lc_float w) noexcept { return lc_float4{x, y, z, w}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float x, lc_float y, lc_float2 zw) noexcept { return lc_float4{x, y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float x, lc_float2 yz, lc_float w) noexcept { return lc_float4{x, yz.x, yz.y, w}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float2 xy, lc_float z, lc_float w) noexcept { return lc_float4{xy.x, xy.y, z, w}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float2 xy, lc_float2 zw) noexcept { return lc_float4{xy.x, xy.y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float x, lc_float3 yzw) noexcept { return lc_float4{x, yzw.x, yzw.y, yzw.z}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float3 xyz, lc_float w) noexcept { return lc_float4{xyz.x, xyz.y, xyz.z, w}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_int4 v) noexcept { return lc_float4{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z), static_cast<lc_float>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_uint4 v) noexcept { return lc_float4{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z), static_cast<lc_float>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_float4 v) noexcept { return lc_float4{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z), static_cast<lc_float>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_float4(lc_bool4 v) noexcept { return lc_float4{static_cast<lc_float>(v.x), static_cast<lc_float>(v.y), static_cast<lc_float>(v.z), static_cast<lc_float>(v.w)}; }

[[nodiscard]] constexpr auto lc_make_bool2(lc_bool s) noexcept { return lc_bool2{s, s}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_bool x, lc_bool y) noexcept { return lc_bool2{x, y}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_int2 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_int3 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_int4 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_uint2 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_uint3 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_uint4 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_float2 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_float3 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_float4 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_bool2 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_bool3 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool2(lc_bool4 v) noexcept { return lc_bool2{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_bool s) noexcept { return lc_bool3{s, s, s}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_bool x, lc_bool y, lc_bool z) noexcept { return lc_bool3{x, y, z}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_bool x, lc_bool2 yz) noexcept { return lc_bool3{x, yz.x, yz.y}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_bool2 xy, lc_bool z) noexcept { return lc_bool3{xy.x, xy.y, z}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_int3 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_int4 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_uint3 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_uint4 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_float3 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_float4 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_bool3 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool3(lc_bool4 v) noexcept { return lc_bool3{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z)}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool s) noexcept { return lc_bool4{s, s, s, s}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool x, lc_bool y, lc_bool z, lc_bool w) noexcept { return lc_bool4{x, y, z, w}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool x, lc_bool y, lc_bool2 zw) noexcept { return lc_bool4{x, y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool x, lc_bool2 yz, lc_bool w) noexcept { return lc_bool4{x, yz.x, yz.y, w}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool2 xy, lc_bool z, lc_bool w) noexcept { return lc_bool4{xy.x, xy.y, z, w}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool2 xy, lc_bool2 zw) noexcept { return lc_bool4{xy.x, xy.y, zw.x, zw.y}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool x, lc_bool3 yzw) noexcept { return lc_bool4{x, yzw.x, yzw.y, yzw.z}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool3 xyz, lc_bool w) noexcept { return lc_bool4{xyz.x, xyz.y, xyz.z, w}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_int4 v) noexcept { return lc_bool4{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z), static_cast<lc_bool>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_uint4 v) noexcept { return lc_bool4{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z), static_cast<lc_bool>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_float4 v) noexcept { return lc_bool4{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z), static_cast<lc_bool>(v.w)}; }
[[nodiscard]] constexpr auto lc_make_bool4(lc_bool4 v) noexcept { return lc_bool4{static_cast<lc_bool>(v.x), static_cast<lc_bool>(v.y), static_cast<lc_bool>(v.z), static_cast<lc_bool>(v.w)}; }

[[nodiscard]] constexpr auto operator!(lc_int2 v) noexcept { return lc_make_bool2(!v.x, !v.y); }
[[nodiscard]] constexpr auto operator+(lc_int2 v) noexcept { return lc_make_int2(+v.x, +v.y); }
[[nodiscard]] constexpr auto operator-(lc_int2 v) noexcept { return lc_make_int2(-v.x, -v.y); }
[[nodiscard]] constexpr auto operator~(lc_int2 v) noexcept { return lc_make_int2(~v.x, ~v.y); }
[[nodiscard]] constexpr auto operator!(lc_int3 v) noexcept { return lc_make_bool3(!v.x, !v.y, !v.z); }
[[nodiscard]] constexpr auto operator+(lc_int3 v) noexcept { return lc_make_int3(+v.x, +v.y, +v.z); }
[[nodiscard]] constexpr auto operator-(lc_int3 v) noexcept { return lc_make_int3(-v.x, -v.y, -v.z); }
[[nodiscard]] constexpr auto operator~(lc_int3 v) noexcept { return lc_make_int3(~v.x, ~v.y, ~v.z); }
[[nodiscard]] constexpr auto operator!(lc_int4 v) noexcept { return lc_make_bool4(!v.x, !v.y, !v.z, !v.w); }
[[nodiscard]] constexpr auto operator+(lc_int4 v) noexcept { return lc_make_int4(+v.x, +v.y, +v.z, +v.w); }
[[nodiscard]] constexpr auto operator-(lc_int4 v) noexcept { return lc_make_int4(-v.x, -v.y, -v.z, -v.w); }
[[nodiscard]] constexpr auto operator~(lc_int4 v) noexcept { return lc_make_int4(~v.x, ~v.y, ~v.z, ~v.w); }

[[nodiscard]] constexpr auto operator!(lc_uint2 v) noexcept { return lc_make_bool2(!v.x, !v.y); }
[[nodiscard]] constexpr auto operator+(lc_uint2 v) noexcept { return lc_make_uint2(+v.x, +v.y); }
[[nodiscard]] constexpr auto operator-(lc_uint2 v) noexcept { return lc_make_uint2(-v.x, -v.y); }
[[nodiscard]] constexpr auto operator~(lc_uint2 v) noexcept { return lc_make_uint2(~v.x, ~v.y); }
[[nodiscard]] constexpr auto operator!(lc_uint3 v) noexcept { return lc_make_bool3(!v.x, !v.y, !v.z); }
[[nodiscard]] constexpr auto operator+(lc_uint3 v) noexcept { return lc_make_uint3(+v.x, +v.y, +v.z); }
[[nodiscard]] constexpr auto operator-(lc_uint3 v) noexcept { return lc_make_uint3(-v.x, -v.y, -v.z); }
[[nodiscard]] constexpr auto operator~(lc_uint3 v) noexcept { return lc_make_uint3(~v.x, ~v.y, ~v.z); }
[[nodiscard]] constexpr auto operator!(lc_uint4 v) noexcept { return lc_make_bool4(!v.x, !v.y, !v.z, !v.w); }
[[nodiscard]] constexpr auto operator+(lc_uint4 v) noexcept { return lc_make_uint4(+v.x, +v.y, +v.z, +v.w); }
[[nodiscard]] constexpr auto operator-(lc_uint4 v) noexcept { return lc_make_uint4(-v.x, -v.y, -v.z, -v.w); }
[[nodiscard]] constexpr auto operator~(lc_uint4 v) noexcept { return lc_make_uint4(~v.x, ~v.y, ~v.z, ~v.w); }

[[nodiscard]] constexpr auto operator!(lc_float2 v) noexcept { return lc_make_bool2(!v.x, !v.y); }
[[nodiscard]] constexpr auto operator+(lc_float2 v) noexcept { return lc_make_float2(+v.x, +v.y); }
[[nodiscard]] constexpr auto operator-(lc_float2 v) noexcept { return lc_make_float2(-v.x, -v.y); }
[[nodiscard]] constexpr auto operator!(lc_float3 v) noexcept { return lc_make_bool3(!v.x, !v.y, !v.z); }
[[nodiscard]] constexpr auto operator+(lc_float3 v) noexcept { return lc_make_float3(+v.x, +v.y, +v.z); }
[[nodiscard]] constexpr auto operator-(lc_float3 v) noexcept { return lc_make_float3(-v.x, -v.y, -v.z); }
[[nodiscard]] constexpr auto operator!(lc_float4 v) noexcept { return lc_make_bool4(!v.x, !v.y, !v.z, !v.w); }
[[nodiscard]] constexpr auto operator+(lc_float4 v) noexcept { return lc_make_float4(+v.x, +v.y, +v.z, +v.w); }
[[nodiscard]] constexpr auto operator-(lc_float4 v) noexcept { return lc_make_float4(-v.x, -v.y, -v.z, -v.w); }

[[nodiscard]] constexpr auto operator!(lc_bool2 v) noexcept { return lc_make_bool2(!v.x, !v.y); }
[[nodiscard]] constexpr auto operator!(lc_bool3 v) noexcept { return lc_make_bool3(!v.x, !v.y, !v.z); }
[[nodiscard]] constexpr auto operator!(lc_bool4 v) noexcept { return lc_make_bool4(!v.x, !v.y, !v.z, !v.w); }

[[nodiscard]] constexpr auto operator==(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_bool2(lhs.x == rhs.x, lhs.y == rhs.y); }
[[nodiscard]] constexpr auto operator==(lc_int2 lhs, lc_int rhs) noexcept { return lhs == lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator==(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_bool3(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z); }
[[nodiscard]] constexpr auto operator==(lc_int3 lhs, lc_int rhs) noexcept { return lhs == lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator==(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_bool4(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w); }
[[nodiscard]] constexpr auto operator==(lc_int4 lhs, lc_int rhs) noexcept { return lhs == lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator==(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_bool2(lhs.x == rhs.x, lhs.y == rhs.y); }
[[nodiscard]] constexpr auto operator==(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs == lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator==(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_bool3(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z); }
[[nodiscard]] constexpr auto operator==(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs == lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator==(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_bool4(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w); }
[[nodiscard]] constexpr auto operator==(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs == lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator==(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_bool2(lhs.x == rhs.x, lhs.y == rhs.y); }
[[nodiscard]] constexpr auto operator==(lc_float2 lhs, lc_float rhs) noexcept { return lhs == lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator==(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_bool3(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z); }
[[nodiscard]] constexpr auto operator==(lc_float3 lhs, lc_float rhs) noexcept { return lhs == lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator==(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_bool4(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w); }
[[nodiscard]] constexpr auto operator==(lc_float4 lhs, lc_float rhs) noexcept { return lhs == lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator==(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x == rhs.x, lhs.y == rhs.y); }
[[nodiscard]] constexpr auto operator==(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs == lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator==(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z); }
[[nodiscard]] constexpr auto operator==(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs == lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator==(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) == rhs; }
[[nodiscard]] constexpr auto operator==(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w); }
[[nodiscard]] constexpr auto operator==(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs == lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator==(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) == rhs; }

[[nodiscard]] constexpr auto operator!=(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_bool2(lhs.x != rhs.x, lhs.y != rhs.y); }
[[nodiscard]] constexpr auto operator!=(lc_int2 lhs, lc_int rhs) noexcept { return lhs != lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_bool3(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z); }
[[nodiscard]] constexpr auto operator!=(lc_int3 lhs, lc_int rhs) noexcept { return lhs != lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_bool4(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w); }
[[nodiscard]] constexpr auto operator!=(lc_int4 lhs, lc_int rhs) noexcept { return lhs != lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_bool2(lhs.x != rhs.x, lhs.y != rhs.y); }
[[nodiscard]] constexpr auto operator!=(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs != lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_bool3(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z); }
[[nodiscard]] constexpr auto operator!=(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs != lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_bool4(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w); }
[[nodiscard]] constexpr auto operator!=(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs != lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_bool2(lhs.x != rhs.x, lhs.y != rhs.y); }
[[nodiscard]] constexpr auto operator!=(lc_float2 lhs, lc_float rhs) noexcept { return lhs != lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_bool3(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z); }
[[nodiscard]] constexpr auto operator!=(lc_float3 lhs, lc_float rhs) noexcept { return lhs != lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_bool4(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w); }
[[nodiscard]] constexpr auto operator!=(lc_float4 lhs, lc_float rhs) noexcept { return lhs != lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x != rhs.x, lhs.y != rhs.y); }
[[nodiscard]] constexpr auto operator!=(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs != lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z); }
[[nodiscard]] constexpr auto operator!=(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs != lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) != rhs; }
[[nodiscard]] constexpr auto operator!=(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w); }
[[nodiscard]] constexpr auto operator!=(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs != lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator!=(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) != rhs; }

[[nodiscard]] constexpr auto operator<(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_bool2(lhs.x < rhs.x, lhs.y < rhs.y); }
[[nodiscard]] constexpr auto operator<(lc_int2 lhs, lc_int rhs) noexcept { return lhs < lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator<(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_bool3(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z); }
[[nodiscard]] constexpr auto operator<(lc_int3 lhs, lc_int rhs) noexcept { return lhs < lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator<(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_bool4(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w); }
[[nodiscard]] constexpr auto operator<(lc_int4 lhs, lc_int rhs) noexcept { return lhs < lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator<(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_bool2(lhs.x < rhs.x, lhs.y < rhs.y); }
[[nodiscard]] constexpr auto operator<(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs < lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator<(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_bool3(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z); }
[[nodiscard]] constexpr auto operator<(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs < lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator<(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_bool4(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w); }
[[nodiscard]] constexpr auto operator<(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs < lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator<(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_bool2(lhs.x < rhs.x, lhs.y < rhs.y); }
[[nodiscard]] constexpr auto operator<(lc_float2 lhs, lc_float rhs) noexcept { return lhs < lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator<(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_bool3(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z); }
[[nodiscard]] constexpr auto operator<(lc_float3 lhs, lc_float rhs) noexcept { return lhs < lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator<(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) < rhs; }
[[nodiscard]] constexpr auto operator<(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_bool4(lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w); }
[[nodiscard]] constexpr auto operator<(lc_float4 lhs, lc_float rhs) noexcept { return lhs < lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator<(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) < rhs; }

[[nodiscard]] constexpr auto operator>(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_bool2(lhs.x > rhs.x, lhs.y > rhs.y); }
[[nodiscard]] constexpr auto operator>(lc_int2 lhs, lc_int rhs) noexcept { return lhs > lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator>(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_bool3(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z); }
[[nodiscard]] constexpr auto operator>(lc_int3 lhs, lc_int rhs) noexcept { return lhs > lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator>(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_bool4(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w); }
[[nodiscard]] constexpr auto operator>(lc_int4 lhs, lc_int rhs) noexcept { return lhs > lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator>(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_bool2(lhs.x > rhs.x, lhs.y > rhs.y); }
[[nodiscard]] constexpr auto operator>(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs > lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator>(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_bool3(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z); }
[[nodiscard]] constexpr auto operator>(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs > lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator>(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_bool4(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w); }
[[nodiscard]] constexpr auto operator>(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs > lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator>(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_bool2(lhs.x > rhs.x, lhs.y > rhs.y); }
[[nodiscard]] constexpr auto operator>(lc_float2 lhs, lc_float rhs) noexcept { return lhs > lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator>(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_bool3(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z); }
[[nodiscard]] constexpr auto operator>(lc_float3 lhs, lc_float rhs) noexcept { return lhs > lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator>(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) > rhs; }
[[nodiscard]] constexpr auto operator>(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_bool4(lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w); }
[[nodiscard]] constexpr auto operator>(lc_float4 lhs, lc_float rhs) noexcept { return lhs > lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator>(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) > rhs; }

[[nodiscard]] constexpr auto operator<=(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_bool2(lhs.x <= rhs.x, lhs.y <= rhs.y); }
[[nodiscard]] constexpr auto operator<=(lc_int2 lhs, lc_int rhs) noexcept { return lhs <= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_bool3(lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z); }
[[nodiscard]] constexpr auto operator<=(lc_int3 lhs, lc_int rhs) noexcept { return lhs <= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_bool4(lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w); }
[[nodiscard]] constexpr auto operator<=(lc_int4 lhs, lc_int rhs) noexcept { return lhs <= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_bool2(lhs.x <= rhs.x, lhs.y <= rhs.y); }
[[nodiscard]] constexpr auto operator<=(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs <= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_bool3(lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z); }
[[nodiscard]] constexpr auto operator<=(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs <= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_bool4(lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w); }
[[nodiscard]] constexpr auto operator<=(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs <= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_bool2(lhs.x <= rhs.x, lhs.y <= rhs.y); }
[[nodiscard]] constexpr auto operator<=(lc_float2 lhs, lc_float rhs) noexcept { return lhs <= lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_bool3(lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z); }
[[nodiscard]] constexpr auto operator<=(lc_float3 lhs, lc_float rhs) noexcept { return lhs <= lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) <= rhs; }
[[nodiscard]] constexpr auto operator<=(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_bool4(lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w); }
[[nodiscard]] constexpr auto operator<=(lc_float4 lhs, lc_float rhs) noexcept { return lhs <= lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator<=(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) <= rhs; }

[[nodiscard]] constexpr auto operator>=(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_bool2(lhs.x >= rhs.x, lhs.y >= rhs.y); }
[[nodiscard]] constexpr auto operator>=(lc_int2 lhs, lc_int rhs) noexcept { return lhs >= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_bool3(lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z); }
[[nodiscard]] constexpr auto operator>=(lc_int3 lhs, lc_int rhs) noexcept { return lhs >= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_bool4(lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w); }
[[nodiscard]] constexpr auto operator>=(lc_int4 lhs, lc_int rhs) noexcept { return lhs >= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_bool2(lhs.x >= rhs.x, lhs.y >= rhs.y); }
[[nodiscard]] constexpr auto operator>=(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs >= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_bool3(lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z); }
[[nodiscard]] constexpr auto operator>=(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs >= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_bool4(lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w); }
[[nodiscard]] constexpr auto operator>=(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs >= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_bool2(lhs.x >= rhs.x, lhs.y >= rhs.y); }
[[nodiscard]] constexpr auto operator>=(lc_float2 lhs, lc_float rhs) noexcept { return lhs >= lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_bool3(lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z); }
[[nodiscard]] constexpr auto operator>=(lc_float3 lhs, lc_float rhs) noexcept { return lhs >= lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) >= rhs; }
[[nodiscard]] constexpr auto operator>=(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_bool4(lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w); }
[[nodiscard]] constexpr auto operator>=(lc_float4 lhs, lc_float rhs) noexcept { return lhs >= lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator>=(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) >= rhs; }

[[nodiscard]] constexpr auto operator+(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x + rhs.x, lhs.y + rhs.y); }
[[nodiscard]] constexpr auto operator+(lc_int2 lhs, lc_int rhs) noexcept { return lhs + lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator+(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
[[nodiscard]] constexpr auto operator+(lc_int3 lhs, lc_int rhs) noexcept { return lhs + lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator+(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
[[nodiscard]] constexpr auto operator+(lc_int4 lhs, lc_int rhs) noexcept { return lhs + lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator+(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x + rhs.x, lhs.y + rhs.y); }
[[nodiscard]] constexpr auto operator+(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs + lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator+(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
[[nodiscard]] constexpr auto operator+(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs + lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator+(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
[[nodiscard]] constexpr auto operator+(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs + lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator+(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs.x + rhs.x, lhs.y + rhs.y); }
[[nodiscard]] constexpr auto operator+(lc_float2 lhs, lc_float rhs) noexcept { return lhs + lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator+(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
[[nodiscard]] constexpr auto operator+(lc_float3 lhs, lc_float rhs) noexcept { return lhs + lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator+(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) + rhs; }
[[nodiscard]] constexpr auto operator+(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
[[nodiscard]] constexpr auto operator+(lc_float4 lhs, lc_float rhs) noexcept { return lhs + lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator+(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) + rhs; }

[[nodiscard]] constexpr auto operator-(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x - rhs.x, lhs.y - rhs.y); }
[[nodiscard]] constexpr auto operator-(lc_int2 lhs, lc_int rhs) noexcept { return lhs - lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator-(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
[[nodiscard]] constexpr auto operator-(lc_int3 lhs, lc_int rhs) noexcept { return lhs - lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator-(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
[[nodiscard]] constexpr auto operator-(lc_int4 lhs, lc_int rhs) noexcept { return lhs - lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator-(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x - rhs.x, lhs.y - rhs.y); }
[[nodiscard]] constexpr auto operator-(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs - lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator-(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
[[nodiscard]] constexpr auto operator-(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs - lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator-(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
[[nodiscard]] constexpr auto operator-(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs - lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator-(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs.x - rhs.x, lhs.y - rhs.y); }
[[nodiscard]] constexpr auto operator-(lc_float2 lhs, lc_float rhs) noexcept { return lhs - lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator-(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
[[nodiscard]] constexpr auto operator-(lc_float3 lhs, lc_float rhs) noexcept { return lhs - lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator-(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) - rhs; }
[[nodiscard]] constexpr auto operator-(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
[[nodiscard]] constexpr auto operator-(lc_float4 lhs, lc_float rhs) noexcept { return lhs - lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator-(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) - rhs; }

[[nodiscard]] constexpr auto operator*(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x * rhs.x, lhs.y * rhs.y); }
[[nodiscard]] constexpr auto operator*(lc_int2 lhs, lc_int rhs) noexcept { return lhs * lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator*(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
[[nodiscard]] constexpr auto operator*(lc_int3 lhs, lc_int rhs) noexcept { return lhs * lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator*(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
[[nodiscard]] constexpr auto operator*(lc_int4 lhs, lc_int rhs) noexcept { return lhs * lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator*(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x * rhs.x, lhs.y * rhs.y); }
[[nodiscard]] constexpr auto operator*(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs * lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator*(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
[[nodiscard]] constexpr auto operator*(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs * lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator*(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
[[nodiscard]] constexpr auto operator*(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs * lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator*(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs.x * rhs.x, lhs.y * rhs.y); }
[[nodiscard]] constexpr auto operator*(lc_float2 lhs, lc_float rhs) noexcept { return lhs * lc_make_float2(rhs); }
[[nodiscard]] constexpr auto operator*(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
[[nodiscard]] constexpr auto operator*(lc_float3 lhs, lc_float rhs) noexcept { return lhs * lc_make_float3(rhs); }
[[nodiscard]] constexpr auto operator*(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) * rhs; }
[[nodiscard]] constexpr auto operator*(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
[[nodiscard]] constexpr auto operator*(lc_float4 lhs, lc_float rhs) noexcept { return lhs * lc_make_float4(rhs); }
[[nodiscard]] constexpr auto operator*(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) * rhs; }

[[nodiscard]] constexpr auto operator/(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x / rhs.x, lhs.y / rhs.y); }
[[nodiscard]] constexpr auto operator/(lc_int2 lhs, lc_int rhs) noexcept { return lhs / lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator/(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }
[[nodiscard]] constexpr auto operator/(lc_int3 lhs, lc_int rhs) noexcept { return lhs / lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator/(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }
[[nodiscard]] constexpr auto operator/(lc_int4 lhs, lc_int rhs) noexcept { return lhs / lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator/(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x / rhs.x, lhs.y / rhs.y); }
[[nodiscard]] constexpr auto operator/(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs / lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator/(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }
[[nodiscard]] constexpr auto operator/(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs / lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator/(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }
[[nodiscard]] constexpr auto operator/(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs / lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator/(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_float2 lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs.x * (1.0f / rhs.x), lhs.y * (1.0f / rhs.y)); }
[[nodiscard]] constexpr auto operator/(lc_float2 lhs, lc_float rhs) noexcept { return lhs * lc_make_float2(1.0f / rhs); }
[[nodiscard]] constexpr auto operator/(lc_float lhs, lc_float2 rhs) noexcept { return lc_make_float2(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_float3 lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs.x * (1.0f / rhs.x), lhs.y * (1.0f / rhs.y), lhs.z * (1.0f / rhs.z)); }
[[nodiscard]] constexpr auto operator/(lc_float3 lhs, lc_float rhs) noexcept { return lhs * lc_make_float3(1.0f / rhs); }
[[nodiscard]] constexpr auto operator/(lc_float lhs, lc_float3 rhs) noexcept { return lc_make_float3(lhs) / rhs; }
[[nodiscard]] constexpr auto operator/(lc_float4 lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs.x * (1.0f / rhs.x), lhs.y * (1.0f / rhs.y), lhs.z * (1.0f / rhs.z), lhs.w * (1.0f / rhs.w)); }
[[nodiscard]] constexpr auto operator/(lc_float4 lhs, lc_float rhs) noexcept { return lhs * lc_make_float4(1.0f / rhs); }
[[nodiscard]] constexpr auto operator/(lc_float lhs, lc_float4 rhs) noexcept { return lc_make_float4(lhs) / rhs; }

[[nodiscard]] constexpr auto operator%(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x % rhs.x, lhs.y % rhs.y); }
[[nodiscard]] constexpr auto operator%(lc_int2 lhs, lc_int rhs) noexcept { return lhs % lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator%(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) % rhs; }
[[nodiscard]] constexpr auto operator%(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z); }
[[nodiscard]] constexpr auto operator%(lc_int3 lhs, lc_int rhs) noexcept { return lhs % lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator%(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) % rhs; }
[[nodiscard]] constexpr auto operator%(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z, lhs.w % rhs.w); }
[[nodiscard]] constexpr auto operator%(lc_int4 lhs, lc_int rhs) noexcept { return lhs % lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator%(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) % rhs; }
[[nodiscard]] constexpr auto operator%(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x % rhs.x, lhs.y % rhs.y); }
[[nodiscard]] constexpr auto operator%(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs % lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator%(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) % rhs; }
[[nodiscard]] constexpr auto operator%(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z); }
[[nodiscard]] constexpr auto operator%(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs % lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator%(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) % rhs; }
[[nodiscard]] constexpr auto operator%(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x % rhs.x, lhs.y % rhs.y, lhs.z % rhs.z, lhs.w % rhs.w); }
[[nodiscard]] constexpr auto operator%(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs % lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator%(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) % rhs; }

[[nodiscard]] constexpr auto operator<<(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x << rhs.x, lhs.y << rhs.y); }
[[nodiscard]] constexpr auto operator<<(lc_int2 lhs, lc_int rhs) noexcept { return lhs << lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator<<(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) << rhs; }
[[nodiscard]] constexpr auto operator<<(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z); }
[[nodiscard]] constexpr auto operator<<(lc_int3 lhs, lc_int rhs) noexcept { return lhs << lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator<<(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) << rhs; }
[[nodiscard]] constexpr auto operator<<(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z, lhs.w << rhs.w); }
[[nodiscard]] constexpr auto operator<<(lc_int4 lhs, lc_int rhs) noexcept { return lhs << lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator<<(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) << rhs; }
[[nodiscard]] constexpr auto operator<<(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x << rhs.x, lhs.y << rhs.y); }
[[nodiscard]] constexpr auto operator<<(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs << lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator<<(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) << rhs; }
[[nodiscard]] constexpr auto operator<<(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z); }
[[nodiscard]] constexpr auto operator<<(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs << lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator<<(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) << rhs; }
[[nodiscard]] constexpr auto operator<<(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x << rhs.x, lhs.y << rhs.y, lhs.z << rhs.z, lhs.w << rhs.w); }
[[nodiscard]] constexpr auto operator<<(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs << lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator<<(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) << rhs; }

[[nodiscard]] constexpr auto operator>>(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x >> rhs.x, lhs.y >> rhs.y); }
[[nodiscard]] constexpr auto operator>>(lc_int2 lhs, lc_int rhs) noexcept { return lhs >> lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator>>(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) >> rhs; }
[[nodiscard]] constexpr auto operator>>(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z); }
[[nodiscard]] constexpr auto operator>>(lc_int3 lhs, lc_int rhs) noexcept { return lhs >> lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator>>(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) >> rhs; }
[[nodiscard]] constexpr auto operator>>(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z, lhs.w >> rhs.w); }
[[nodiscard]] constexpr auto operator>>(lc_int4 lhs, lc_int rhs) noexcept { return lhs >> lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator>>(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) >> rhs; }
[[nodiscard]] constexpr auto operator>>(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x >> rhs.x, lhs.y >> rhs.y); }
[[nodiscard]] constexpr auto operator>>(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs >> lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator>>(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) >> rhs; }
[[nodiscard]] constexpr auto operator>>(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z); }
[[nodiscard]] constexpr auto operator>>(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs >> lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator>>(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) >> rhs; }
[[nodiscard]] constexpr auto operator>>(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x >> rhs.x, lhs.y >> rhs.y, lhs.z >> rhs.z, lhs.w >> rhs.w); }
[[nodiscard]] constexpr auto operator>>(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs >> lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator>>(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) >> rhs; }

[[nodiscard]] constexpr auto operator|(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x | rhs.x, lhs.y | rhs.y); }
[[nodiscard]] constexpr auto operator|(lc_int2 lhs, lc_int rhs) noexcept { return lhs | lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator|(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z); }
[[nodiscard]] constexpr auto operator|(lc_int3 lhs, lc_int rhs) noexcept { return lhs | lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator|(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z, lhs.w | rhs.w); }
[[nodiscard]] constexpr auto operator|(lc_int4 lhs, lc_int rhs) noexcept { return lhs | lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator|(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x | rhs.x, lhs.y | rhs.y); }
[[nodiscard]] constexpr auto operator|(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs | lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator|(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z); }
[[nodiscard]] constexpr auto operator|(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs | lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator|(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z, lhs.w | rhs.w); }
[[nodiscard]] constexpr auto operator|(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs | lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator|(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x | rhs.x, lhs.y | rhs.y); }
[[nodiscard]] constexpr auto operator|(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs | lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator|(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z); }
[[nodiscard]] constexpr auto operator|(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs | lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator|(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) | rhs; }
[[nodiscard]] constexpr auto operator|(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x | rhs.x, lhs.y | rhs.y, lhs.z | rhs.z, lhs.w | rhs.w); }
[[nodiscard]] constexpr auto operator|(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs | lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator|(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) | rhs; }

[[nodiscard]] constexpr auto operator&(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x & rhs.x, lhs.y & rhs.y); }
[[nodiscard]] constexpr auto operator&(lc_int2 lhs, lc_int rhs) noexcept { return lhs & lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator&(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z); }
[[nodiscard]] constexpr auto operator&(lc_int3 lhs, lc_int rhs) noexcept { return lhs & lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator&(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z, lhs.w & rhs.w); }
[[nodiscard]] constexpr auto operator&(lc_int4 lhs, lc_int rhs) noexcept { return lhs & lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator&(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x & rhs.x, lhs.y & rhs.y); }
[[nodiscard]] constexpr auto operator&(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs & lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator&(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z); }
[[nodiscard]] constexpr auto operator&(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs & lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator&(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z, lhs.w & rhs.w); }
[[nodiscard]] constexpr auto operator&(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs & lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator&(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x & rhs.x, lhs.y & rhs.y); }
[[nodiscard]] constexpr auto operator&(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs & lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator&(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z); }
[[nodiscard]] constexpr auto operator&(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs & lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator&(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) & rhs; }
[[nodiscard]] constexpr auto operator&(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x & rhs.x, lhs.y & rhs.y, lhs.z & rhs.z, lhs.w & rhs.w); }
[[nodiscard]] constexpr auto operator&(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs & lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator&(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) & rhs; }

[[nodiscard]] constexpr auto operator^(lc_int2 lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs.x ^ rhs.x, lhs.y ^ rhs.y); }
[[nodiscard]] constexpr auto operator^(lc_int2 lhs, lc_int rhs) noexcept { return lhs ^ lc_make_int2(rhs); }
[[nodiscard]] constexpr auto operator^(lc_int lhs, lc_int2 rhs) noexcept { return lc_make_int2(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_int3 lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z); }
[[nodiscard]] constexpr auto operator^(lc_int3 lhs, lc_int rhs) noexcept { return lhs ^ lc_make_int3(rhs); }
[[nodiscard]] constexpr auto operator^(lc_int lhs, lc_int3 rhs) noexcept { return lc_make_int3(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_int4 lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w); }
[[nodiscard]] constexpr auto operator^(lc_int4 lhs, lc_int rhs) noexcept { return lhs ^ lc_make_int4(rhs); }
[[nodiscard]] constexpr auto operator^(lc_int lhs, lc_int4 rhs) noexcept { return lc_make_int4(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_uint2 lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs.x ^ rhs.x, lhs.y ^ rhs.y); }
[[nodiscard]] constexpr auto operator^(lc_uint2 lhs, lc_uint rhs) noexcept { return lhs ^ lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto operator^(lc_uint lhs, lc_uint2 rhs) noexcept { return lc_make_uint2(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_uint3 lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z); }
[[nodiscard]] constexpr auto operator^(lc_uint3 lhs, lc_uint rhs) noexcept { return lhs ^ lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto operator^(lc_uint lhs, lc_uint3 rhs) noexcept { return lc_make_uint3(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_uint4 lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w); }
[[nodiscard]] constexpr auto operator^(lc_uint4 lhs, lc_uint rhs) noexcept { return lhs ^ lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto operator^(lc_uint lhs, lc_uint4 rhs) noexcept { return lc_make_uint4(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x ^ rhs.x, lhs.y ^ rhs.y); }
[[nodiscard]] constexpr auto operator^(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs ^ lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator^(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z); }
[[nodiscard]] constexpr auto operator^(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs ^ lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator^(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) ^ rhs; }
[[nodiscard]] constexpr auto operator^(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x ^ rhs.x, lhs.y ^ rhs.y, lhs.z ^ rhs.z, lhs.w ^ rhs.w); }
[[nodiscard]] constexpr auto operator^(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs ^ lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator^(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) ^ rhs; }

[[nodiscard]] constexpr auto operator||(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x || rhs.x, lhs.y || rhs.y); }
[[nodiscard]] constexpr auto operator||(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs || lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator||(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) || rhs; }
[[nodiscard]] constexpr auto operator||(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z); }
[[nodiscard]] constexpr auto operator||(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs || lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator||(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) || rhs; }
[[nodiscard]] constexpr auto operator||(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z, lhs.w || rhs.w); }
[[nodiscard]] constexpr auto operator||(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs || lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator||(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) || rhs; }

[[nodiscard]] constexpr auto operator&&(lc_bool2 lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs.x && rhs.x, lhs.y && rhs.y); }
[[nodiscard]] constexpr auto operator&&(lc_bool2 lhs, lc_bool rhs) noexcept { return lhs && lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto operator&&(lc_bool lhs, lc_bool2 rhs) noexcept { return lc_make_bool2(lhs) && rhs; }
[[nodiscard]] constexpr auto operator&&(lc_bool3 lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z); }
[[nodiscard]] constexpr auto operator&&(lc_bool3 lhs, lc_bool rhs) noexcept { return lhs && lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto operator&&(lc_bool lhs, lc_bool3 rhs) noexcept { return lc_make_bool3(lhs) && rhs; }
[[nodiscard]] constexpr auto operator&&(lc_bool4 lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z, lhs.w && rhs.w); }
[[nodiscard]] constexpr auto operator&&(lc_bool4 lhs, lc_bool rhs) noexcept { return lhs && lc_make_bool4(rhs); }
[[nodiscard]] constexpr auto operator&&(lc_bool lhs, lc_bool4 rhs) noexcept { return lc_make_bool4(lhs) && rhs; }

[[nodiscard]] constexpr auto &operator+=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs += lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs += lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs += lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs += lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs += lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs += lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_float2 &lhs, lc_float2 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_float2 &lhs, lc_float rhs) noexcept { return lhs += lc_make_float2(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_float3 &lhs, lc_float3 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_float3 &lhs, lc_float rhs) noexcept { return lhs += lc_make_float3(rhs); }
[[nodiscard]] constexpr auto &operator+=(lc_float4 &lhs, lc_float4 rhs) noexcept {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator+=(lc_float4 &lhs, lc_float rhs) noexcept { return lhs += lc_make_float4(rhs); }

[[nodiscard]] constexpr auto &operator-=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs -= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs -= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs -= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs -= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs -= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs -= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_float2 &lhs, lc_float2 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_float2 &lhs, lc_float rhs) noexcept { return lhs -= lc_make_float2(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_float3 &lhs, lc_float3 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_float3 &lhs, lc_float rhs) noexcept { return lhs -= lc_make_float3(rhs); }
[[nodiscard]] constexpr auto &operator-=(lc_float4 &lhs, lc_float4 rhs) noexcept {
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator-=(lc_float4 &lhs, lc_float rhs) noexcept { return lhs -= lc_make_float4(rhs); }

[[nodiscard]] constexpr auto &operator*=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs *= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs *= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    lhs.w *= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs *= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs *= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs *= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    lhs.w *= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs *= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_float2 &lhs, lc_float2 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_float2 &lhs, lc_float rhs) noexcept { return lhs *= lc_make_float2(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_float3 &lhs, lc_float3 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_float3 &lhs, lc_float rhs) noexcept { return lhs *= lc_make_float3(rhs); }
[[nodiscard]] constexpr auto &operator*=(lc_float4 &lhs, lc_float4 rhs) noexcept {
    lhs.x *= rhs.x;
    lhs.y *= rhs.y;
    lhs.z *= rhs.z;
    lhs.w *= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator*=(lc_float4 &lhs, lc_float rhs) noexcept { return lhs *= lc_make_float4(rhs); }

[[nodiscard]] constexpr auto &operator/=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs /= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    lhs.z *= 1.0f / rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs /= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    lhs.z *= 1.0f / rhs.z;
    lhs.w *= 1.0f / rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs /= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs /= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    lhs.z *= 1.0f / rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs /= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    lhs.z *= 1.0f / rhs.z;
    lhs.w *= 1.0f / rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs /= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_float2 &lhs, lc_float2 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_float2 &lhs, lc_float rhs) noexcept { return lhs *= lc_make_float2(1.0f / rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_float3 &lhs, lc_float3 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    lhs.z *= 1.0f / rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_float3 &lhs, lc_float rhs) noexcept { return lhs *= lc_make_float3(1.0f / rhs); }
[[nodiscard]] constexpr auto &operator/=(lc_float4 &lhs, lc_float4 rhs) noexcept {
    lhs.x *= 1.0f / rhs.x;
    lhs.y *= 1.0f / rhs.y;
    lhs.z *= 1.0f / rhs.z;
    lhs.w *= 1.0f / rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator/=(lc_float4 &lhs, lc_float rhs) noexcept { return lhs *= lc_make_float4(1.0f / rhs); }

[[nodiscard]] constexpr auto &operator%=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x %= rhs.x;
    lhs.y %= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator%=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs %= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator%=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x %= rhs.x;
    lhs.y %= rhs.y;
    lhs.z %= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator%=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs %= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator%=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x %= rhs.x;
    lhs.y %= rhs.y;
    lhs.z %= rhs.z;
    lhs.w %= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator%=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs %= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator%=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x %= rhs.x;
    lhs.y %= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator%=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs %= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator%=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x %= rhs.x;
    lhs.y %= rhs.y;
    lhs.z %= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator%=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs %= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator%=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x %= rhs.x;
    lhs.y %= rhs.y;
    lhs.z %= rhs.z;
    lhs.w %= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator%=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs %= lc_make_uint4(rhs); }

[[nodiscard]] constexpr auto &operator<<=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x <<= rhs.x;
    lhs.y <<= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator<<=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs <<= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator<<=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x <<= rhs.x;
    lhs.y <<= rhs.y;
    lhs.z <<= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator<<=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs <<= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator<<=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x <<= rhs.x;
    lhs.y <<= rhs.y;
    lhs.z <<= rhs.z;
    lhs.w <<= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator<<=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs <<= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator<<=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x <<= rhs.x;
    lhs.y <<= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator<<=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs <<= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator<<=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x <<= rhs.x;
    lhs.y <<= rhs.y;
    lhs.z <<= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator<<=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs <<= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator<<=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x <<= rhs.x;
    lhs.y <<= rhs.y;
    lhs.z <<= rhs.z;
    lhs.w <<= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator<<=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs <<= lc_make_uint4(rhs); }

[[nodiscard]] constexpr auto &operator>>=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x >>= rhs.x;
    lhs.y >>= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator>>=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs >>= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator>>=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x >>= rhs.x;
    lhs.y >>= rhs.y;
    lhs.z >>= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator>>=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs >>= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator>>=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x >>= rhs.x;
    lhs.y >>= rhs.y;
    lhs.z >>= rhs.z;
    lhs.w >>= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator>>=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs >>= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator>>=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x >>= rhs.x;
    lhs.y >>= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator>>=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs >>= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator>>=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x >>= rhs.x;
    lhs.y >>= rhs.y;
    lhs.z >>= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator>>=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs >>= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator>>=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x >>= rhs.x;
    lhs.y >>= rhs.y;
    lhs.z >>= rhs.z;
    lhs.w >>= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator>>=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs >>= lc_make_uint4(rhs); }

[[nodiscard]] constexpr auto &operator|=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs |= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    lhs.z |= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs |= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    lhs.z |= rhs.z;
    lhs.w |= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs |= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs |= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    lhs.z |= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs |= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    lhs.z |= rhs.z;
    lhs.w |= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs |= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_bool2 &lhs, lc_bool2 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_bool2 &lhs, lc_bool rhs) noexcept { return lhs |= lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_bool3 &lhs, lc_bool3 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    lhs.z |= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_bool3 &lhs, lc_bool rhs) noexcept { return lhs |= lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto &operator|=(lc_bool4 &lhs, lc_bool4 rhs) noexcept {
    lhs.x |= rhs.x;
    lhs.y |= rhs.y;
    lhs.z |= rhs.z;
    lhs.w |= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator|=(lc_bool4 &lhs, lc_bool rhs) noexcept { return lhs |= lc_make_bool4(rhs); }

[[nodiscard]] constexpr auto &operator&=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs &= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    lhs.z &= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs &= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    lhs.z &= rhs.z;
    lhs.w &= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs &= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs &= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    lhs.z &= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs &= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    lhs.z &= rhs.z;
    lhs.w &= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs &= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_bool2 &lhs, lc_bool2 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_bool2 &lhs, lc_bool rhs) noexcept { return lhs &= lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_bool3 &lhs, lc_bool3 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    lhs.z &= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_bool3 &lhs, lc_bool rhs) noexcept { return lhs &= lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto &operator&=(lc_bool4 &lhs, lc_bool4 rhs) noexcept {
    lhs.x &= rhs.x;
    lhs.y &= rhs.y;
    lhs.z &= rhs.z;
    lhs.w &= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator&=(lc_bool4 &lhs, lc_bool rhs) noexcept { return lhs &= lc_make_bool4(rhs); }

[[nodiscard]] constexpr auto &operator^=(lc_int2 &lhs, lc_int2 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_int2 &lhs, lc_int rhs) noexcept { return lhs ^= lc_make_int2(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_int3 &lhs, lc_int3 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    lhs.z ^= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_int3 &lhs, lc_int rhs) noexcept { return lhs ^= lc_make_int3(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_int4 &lhs, lc_int4 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    lhs.z ^= rhs.z;
    lhs.w ^= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_int4 &lhs, lc_int rhs) noexcept { return lhs ^= lc_make_int4(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_uint2 &lhs, lc_uint2 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_uint2 &lhs, lc_uint rhs) noexcept { return lhs ^= lc_make_uint2(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_uint3 &lhs, lc_uint3 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    lhs.z ^= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_uint3 &lhs, lc_uint rhs) noexcept { return lhs ^= lc_make_uint3(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_uint4 &lhs, lc_uint4 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    lhs.z ^= rhs.z;
    lhs.w ^= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_uint4 &lhs, lc_uint rhs) noexcept { return lhs ^= lc_make_uint4(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_bool2 &lhs, lc_bool2 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_bool2 &lhs, lc_bool rhs) noexcept { return lhs ^= lc_make_bool2(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_bool3 &lhs, lc_bool3 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    lhs.z ^= rhs.z;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_bool3 &lhs, lc_bool rhs) noexcept { return lhs ^= lc_make_bool3(rhs); }
[[nodiscard]] constexpr auto &operator^=(lc_bool4 &lhs, lc_bool4 rhs) noexcept {
    lhs.x ^= rhs.x;
    lhs.y ^= rhs.y;
    lhs.z ^= rhs.z;
    lhs.w ^= rhs.w;
    return lhs;
}
[[nodiscard]] constexpr auto &operator^=(lc_bool4 &lhs, lc_bool rhs) noexcept { return lhs ^= lc_make_bool4(rhs); }

[[nodiscard]] constexpr auto lc_any(lc_bool2 v) noexcept { return v.x || v.y; }
[[nodiscard]] constexpr auto lc_any(lc_bool3 v) noexcept { return v.x || v.y || v.z; }
[[nodiscard]] constexpr auto lc_any(lc_bool4 v) noexcept { return v.x || v.y || v.z || v.w; }
[[nodiscard]] constexpr auto lc_all(lc_bool2 v) noexcept { return v.x && v.y; }
[[nodiscard]] constexpr auto lc_all(lc_bool3 v) noexcept { return v.x && v.y && v.z; }
[[nodiscard]] constexpr auto lc_all(lc_bool4 v) noexcept { return v.x && v.y && v.z && v.w; }
[[nodiscard]] constexpr auto lc_none(lc_bool2 v) noexcept { return !v.x && !v.y; }
[[nodiscard]] constexpr auto lc_none(lc_bool3 v) noexcept { return !v.x && !v.y && !v.z; }
[[nodiscard]] constexpr auto lc_none(lc_bool4 v) noexcept { return !v.x && !v.y && !v.z && !v.w; }

struct lc_float2x2 {
    lc_float2 cols[2];
    constexpr lc_float2x2(lc_float s = 1.0f) noexcept
        : cols{lc_make_float2(s, 0.0f), lc_make_float2(0.0f, s)} {}
    constexpr lc_float2x2(lc_float2 c0, lc_float2 c1) noexcept
        : cols{c0, c1} {}
    [[nodiscard]] constexpr auto &operator[](lc_uint i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr auto operator[](lc_uint i) const noexcept { return cols[i]; }
};

struct lc_float3x3 {
    lc_float3 cols[3];
    constexpr lc_float3x3(lc_float s = 1.0f) noexcept
        : cols{lc_make_float3(s, 0.0f, 0.0f), lc_make_float3(0.0f, s, 0.0f), lc_make_float3(0.0f, 0.0f, s)} {}
    constexpr lc_float3x3(lc_float3 c0, lc_float3 c1, lc_float3 c2) noexcept
        : cols{c0, c1, c2} {}
    [[nodiscard]] constexpr auto &operator[](lc_uint i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr auto operator[](lc_uint i) const noexcept { return cols[i]; }
};

struct lc_float4x4 {
    lc_float4 cols[4];
    constexpr lc_float4x4(lc_float s = 1.0f) noexcept
        : cols{lc_make_float4(s, 0.0f, 0.0f, 0.0f), lc_make_float4(0.0f, s, 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, s, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, s)} {}
    constexpr lc_float4x4(lc_float4 c0, lc_float4 c1, lc_float4 c2, lc_float4 c3) noexcept
        : cols{c0, c1, c2, c3} {}
    [[nodiscard]] constexpr auto &operator[](lc_uint i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr auto operator[](lc_uint i) const noexcept { return cols[i]; }
};

[[nodiscard]] constexpr auto operator*(const lc_float2x2 m, lc_float s) noexcept { return lc_float2x2{m[0] * s, m[1] * s}; }
[[nodiscard]] constexpr auto operator*(lc_float s, const lc_float2x2 m) noexcept { return m * s; }
[[nodiscard]] constexpr auto operator/(const lc_float2x2 m, lc_float s) noexcept { return m * (1.0f / s); }
[[nodiscard]] constexpr auto operator*(const lc_float2x2 m, const lc_float2 v) noexcept { return v.x * m[0] + v.y * m[1]; }
[[nodiscard]] constexpr auto operator*(const lc_float2x2 lhs, const lc_float2x2 rhs) noexcept { return lc_float2x2{lhs * rhs[0], lhs * rhs[1]}; }
[[nodiscard]] constexpr auto operator+(const lc_float2x2 lhs, const lc_float2x2 rhs) noexcept { return lc_float2x2{lhs[0] + rhs[0], lhs[1] + rhs[1]}; }
[[nodiscard]] constexpr auto operator-(const lc_float2x2 lhs, const lc_float2x2 rhs) noexcept { return lc_float2x2{lhs[0] - rhs[0], lhs[1] - rhs[1]}; }

[[nodiscard]] constexpr auto operator*(const lc_float3x3 m, lc_float s) noexcept { return lc_float3x3{m[0] * s, m[1] * s, m[2] * s}; }
[[nodiscard]] constexpr auto operator*(lc_float s, const lc_float3x3 m) noexcept { return m * s; }
[[nodiscard]] constexpr auto operator/(const lc_float3x3 m, lc_float s) noexcept { return m * (1.0f / s); }
[[nodiscard]] constexpr auto operator*(const lc_float3x3 m, const lc_float3 v) noexcept { return v.x * m[0] + v.y * m[1] + v.z * m[2]; }
[[nodiscard]] constexpr auto operator*(const lc_float3x3 lhs, const lc_float3x3 rhs) noexcept { return lc_float3x3{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]}; }
[[nodiscard]] constexpr auto operator+(const lc_float3x3 lhs, const lc_float3x3 rhs) noexcept { return lc_float3x3{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]}; }
[[nodiscard]] constexpr auto operator-(const lc_float3x3 lhs, const lc_float3x3 rhs) noexcept { return lc_float3x3{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]}; }

[[nodiscard]] constexpr auto operator*(const lc_float4x4 m, lc_float s) noexcept { return lc_float4x4{m[0] * s, m[1] * s, m[2] * s, m[3] * s}; }
[[nodiscard]] constexpr auto operator*(lc_float s, const lc_float4x4 m) noexcept { return m * s; }
[[nodiscard]] constexpr auto operator/(const lc_float4x4 m, lc_float s) noexcept { return m * (1.0f / s); }
[[nodiscard]] constexpr auto operator*(const lc_float4x4 m, const lc_float4 v) noexcept { return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3]; }
[[nodiscard]] constexpr auto operator*(const lc_float4x4 lhs, const lc_float4x4 rhs) noexcept { return lc_float4x4{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]}; }
[[nodiscard]] constexpr auto operator+(const lc_float4x4 lhs, const lc_float4x4 rhs) noexcept { return lc_float4x4{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]}; }
[[nodiscard]] constexpr auto operator-(const lc_float4x4 lhs, const lc_float4x4 rhs) noexcept { return lc_float4x4{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]}; }

[[nodiscard]] constexpr auto lc_make_float2x2(lc_float s = 1.0f) noexcept { return lc_float2x2{lc_make_float2(s, 0.0f), lc_make_float2(0.0f, s)}; }
[[nodiscard]] constexpr auto lc_make_float2x2(lc_float m00, lc_float m01, lc_float m10, lc_float m11) noexcept { return lc_float2x2{lc_make_float2(m00, m01), lc_make_float2(m10, m11)}; }
[[nodiscard]] constexpr auto lc_make_float2x2(lc_float2 c0, lc_float2 c1) noexcept { return lc_float2x2{c0, c1}; }
[[nodiscard]] constexpr auto lc_make_float2x2(lc_float2x2 m) noexcept { return m; }
[[nodiscard]] constexpr auto lc_make_float2x2(lc_float3x3 m) noexcept { return lc_float2x2{lc_make_float2(m[0]), lc_make_float2(m[1])}; }
[[nodiscard]] constexpr auto lc_make_float2x2(lc_float4x4 m) noexcept { return lc_float2x2{lc_make_float2(m[0]), lc_make_float2(m[1])}; }

[[nodiscard]] constexpr auto lc_make_float3x3(lc_float s = 1.0f) noexcept { return lc_float3x3{lc_make_float3(s, 0.0f, 0.0f), lc_make_float3(0.0f, s, 0.0f), lc_make_float3(0.0f, 0.0f, s)}; }
[[nodiscard]] constexpr auto lc_make_float3x3(lc_float m00, lc_float m01, lc_float m02, lc_float m10, lc_float m11, lc_float m12, lc_float m20, lc_float m21, lc_float m22) noexcept { return lc_float3x3{lc_make_float3(m00, m01, m02), lc_make_float3(m10, m11, m12), lc_make_float3(m20, m21, m22)}; }
[[nodiscard]] constexpr auto lc_make_float3x3(lc_float3 c0, lc_float3 c1, lc_float3 c2) noexcept { return lc_float3x3{c0, c1, c2}; }
[[nodiscard]] constexpr auto lc_make_float3x3(lc_float2x2 m) noexcept { return lc_float3x3{lc_make_float3(m[0], 0.0f), lc_make_float3(m[1], 0.0f), lc_make_float3(0.0f, 0.0f, 1.0f)}; }
[[nodiscard]] constexpr auto lc_make_float3x3(lc_float3x3 m) noexcept { return m; }
[[nodiscard]] constexpr auto lc_make_float3x3(lc_float4x4 m) noexcept { return lc_float3x3{lc_make_float3(m[0]), lc_make_float3(m[1]), lc_make_float3(m[2])}; }

[[nodiscard]] constexpr auto lc_make_float4x4(lc_float s = 1.0f) noexcept { return lc_float4x4{lc_make_float4(s, 0.0f, 0.0f, 0.0f), lc_make_float4(0.0f, s, 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, s, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, s)}; }
[[nodiscard]] constexpr auto lc_make_float4x4(lc_float m00, lc_float m01, lc_float m02, lc_float m03, lc_float m10, lc_float m11, lc_float m12, lc_float m13, lc_float m20, lc_float m21, lc_float m22, lc_float m23, lc_float m30, lc_float m31, lc_float m32, lc_float m33) noexcept { return lc_float4x4{lc_make_float4(m00, m01, m02, m03), lc_make_float4(m10, m11, m12, m13), lc_make_float4(m20, m21, m22, m23), lc_make_float4(m30, m31, m32, m33)}; }
[[nodiscard]] constexpr auto lc_make_float4x4(lc_float4 c0, lc_float4 c1, lc_float4 c2, lc_float4 c3) noexcept { return lc_float4x4{c0, c1, c2, c3}; }
[[nodiscard]] constexpr auto lc_make_float4x4(lc_float2x2 m) noexcept { return lc_float4x4{lc_make_float4(m[0], 0.0f, 0.0f), lc_make_float4(m[1], 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 1.0f)}; }
[[nodiscard]] constexpr auto lc_make_float4x4(lc_float3x3 m) noexcept { return lc_float4x4{lc_make_float4(m[0], 0.0f), lc_make_float4(m[1], 0.0f), lc_make_float4(m[2], 0.0f), lc_make_float4(0.0f, 0.0f, 0.0f, 1.0f)}; }
[[nodiscard]] constexpr auto lc_make_float4x4(lc_float4x4 m) noexcept { return m; }
