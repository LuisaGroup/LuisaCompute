//
// Created by Mike Smith on 2020/1/30.
//

#pragma once

#include <cmath>
#include <algorithm>

#include <core/basic_types.h>
#include <core/constants.h>

namespace luisa {

/**
 * @brief Find next 2^n of v
 * 
 * @tparam uint32 or uint64
 * @param v input number
 * @return same as v
 */
template<typename T, std::enable_if_t<std::is_unsigned_v<T> && (sizeof(T) == 4u || sizeof(T) == 8u), int> = 0>
[[nodiscard]] constexpr auto next_pow2(T v) noexcept {
    v--;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    if constexpr (sizeof(T) == 8u) { v |= v >> 32u; }
    return v + 1u;
}

// Scalar Functions
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cos;
using std::sin;
using std::tan;

using std::sqrt;

using std::ceil;
using std::floor;
using std::round;

using std::exp;
using std::log;
using std::log10;
using std::log2;
using std::pow;

using std::abs;

using std::max;
using std::min;

/// Convert degree to radian
[[nodiscard]] constexpr float radians(float deg) noexcept { return deg * constants::pi / 180.0f; }
/// Convert radian to degree
[[nodiscard]] constexpr float degrees(float rad) noexcept { return rad * constants::inv_pi * 180.0f; }

/// Make function apply to floatN element-wise
#define LUISA_MAKE_VECTOR_UNARY_FUNC(func)                                                                \
    [[nodiscard]] inline auto func(float2 v) noexcept { return float2{func(v.x), func(v.y)}; }            \
    [[nodiscard]] inline auto func(float3 v) noexcept { return float3{func(v.x), func(v.y), func(v.z)}; } \
    [[nodiscard]] inline auto func(float4 v) noexcept { return float4{func(v.x), func(v.y), func(v.z), func(v.w)}; }

LUISA_MAKE_VECTOR_UNARY_FUNC(acos)
LUISA_MAKE_VECTOR_UNARY_FUNC(asin)
LUISA_MAKE_VECTOR_UNARY_FUNC(atan)
LUISA_MAKE_VECTOR_UNARY_FUNC(cos)
LUISA_MAKE_VECTOR_UNARY_FUNC(sin)
LUISA_MAKE_VECTOR_UNARY_FUNC(tan)
LUISA_MAKE_VECTOR_UNARY_FUNC(sqrt)
LUISA_MAKE_VECTOR_UNARY_FUNC(ceil)
LUISA_MAKE_VECTOR_UNARY_FUNC(floor)
LUISA_MAKE_VECTOR_UNARY_FUNC(round)
LUISA_MAKE_VECTOR_UNARY_FUNC(exp)
LUISA_MAKE_VECTOR_UNARY_FUNC(log)
LUISA_MAKE_VECTOR_UNARY_FUNC(log10)
LUISA_MAKE_VECTOR_UNARY_FUNC(log2)
LUISA_MAKE_VECTOR_UNARY_FUNC(abs)
LUISA_MAKE_VECTOR_UNARY_FUNC(radians)
LUISA_MAKE_VECTOR_UNARY_FUNC(degrees)

#undef LUISA_MAKE_VECTOR_UNARY_FUNC

template<typename T>
[[nodiscard]] inline auto abs(Vector<T, 2> v) noexcept { return Vector<T, 2>{abs(v.x), abs(v.y)}; }
template<typename T>
[[nodiscard]] inline auto abs(Vector<T, 3> v) noexcept { return Vector<T, 3>{abs(v.x), abs(v.y), abs(v.z)}; }
template<typename T>
[[nodiscard]] inline auto abs(Vector<T, 4> v) noexcept { return Vector<T, 4>{abs(v.x), abs(v.y), abs(v.z), abs(v.w)}; }

/// Make function apply to floatN element-wise
#define LUISA_MAKE_VECTOR_BINARY_FUNC(func)                                                  \
    template<typename T>                                                                     \
    [[nodiscard]] constexpr auto func(Vector<T, 2> v, Vector<T, 2> u) noexcept {             \
        return Vector<T, 2>{func(v.x, u.x), func(v.y, u.y)};                                 \
    }                                                                                        \
    template<typename T>                                                                     \
    [[nodiscard]] constexpr auto func(Vector<T, 3> v, Vector<T, 3> u) noexcept {             \
        return Vector<T, 3>{func(v.x, u.x), func(v.y, u.y), func(v.z, u.z)};                 \
    }                                                                                        \
    template<typename T>                                                                     \
    [[nodiscard]] constexpr auto func(Vector<T, 4> v, Vector<T, 4> u) noexcept {             \
        return Vector<T, 4>{func(v.x, u.x), func(v.y, u.y), func(v.z, u.z), func(v.w, u.w)}; \
    }

LUISA_MAKE_VECTOR_BINARY_FUNC(atan2)
LUISA_MAKE_VECTOR_BINARY_FUNC(pow)
LUISA_MAKE_VECTOR_BINARY_FUNC(min)
LUISA_MAKE_VECTOR_BINARY_FUNC(max)

#undef LUISA_MAKE_VECTOR_BINARY_FUNC

// min
template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto min(T v, Vector<T, 2> u) noexcept {
    return Vector<T, 2>{min(v, u.x), min(v, u.y)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto min(T v, Vector<T, 3> u) noexcept {
    return Vector<T, 3>{min(v, u.x), min(v, u.y), min(v, u.z)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto min(T v, Vector<T, 4> u) noexcept {
    return Vector<T, 4>{min(v, u.x), min(v, u.y), min(v, u.z), min(v, u.w)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto min(Vector<T, 2> v, T u) noexcept {
    return Vector<T, 2>{min(v.x, u), min(v.y, u)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto min(Vector<T, 3> v, T u) noexcept {
    return Vector<T, 3>{min(v.x, u), min(v.y, u), min(v.z, u)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto min(Vector<T, 4> v, T u) noexcept {
    return Vector<T, 4>{min(v.x, u), min(v.y, u), min(v.z, u), min(v.w, u)};
}

// max
template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto max(T v, Vector<T, 2> u) noexcept {
    return Vector<T, 2>{max(v, u.x), max(v, u.y)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto max(T v, Vector<T, 3> u) noexcept {
    return Vector<T, 3>{max(v, u.x), max(v, u.y), max(v, u.z)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto max(T v, Vector<T, 4> u) noexcept {
    return Vector<T, 4>{max(v, u.x), max(v, u.y), max(v, u.z), max(v, u.w)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto max(Vector<T, 2> v, T u) noexcept {
    return Vector<T, 2>{max(v.x, u), max(v.y, u)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto max(Vector<T, 3> v, T u) noexcept {
    return Vector<T, 3>{max(v.x, u), max(v.y, u), max(v.z, u)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto max(Vector<T, 4> v, T u) noexcept {
    return Vector<T, 4>{max(v.x, u), max(v.y, u), max(v.z, u), max(v.w, u)};
}

template<typename T, std::enable_if_t<std::disjunction_v<is_scalar<T>, is_vector<T>>, int> = 0>
[[nodiscard]] constexpr auto select(T f, T t, bool pred) noexcept {
    return pred ? t : f;
}

template<typename T>
[[nodiscard]] constexpr auto select(Vector<T, 2> f, Vector<T, 2> t, bool2 pred) noexcept {
    return Vector<T, 2>{select(f.x, t.x, pred.x),
                        select(f.y, t.y, pred.y)};
}

template<typename T>
[[nodiscard]] constexpr auto select(Vector<T, 3> f, Vector<T, 3> t, bool3 pred) noexcept {
    return Vector<T, 3>{select(f.x, t.x, pred.x),
                        select(f.y, t.y, pred.y),
                        select(f.z, t.z, pred.z)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto select(Vector<T, 4> f, Vector<T, 4> t, bool4 pred) noexcept {
    return Vector<T, 4>{select(f.x, t.x, pred.x),
                        select(f.y, t.y, pred.y),
                        select(f.z, t.z, pred.z),
                        select(f.w, t.w, pred.w)};
}

[[nodiscard]] constexpr auto lerp(float a, float b, float t) noexcept {
    return a + t * (b - a);
}

[[nodiscard]] constexpr auto lerp(float2 a, float2 b, float t) noexcept {
    return float2{lerp(a.x, b.x, t),
                  lerp(a.y, b.y, t)};
}

[[nodiscard]] constexpr auto lerp(float3 a, float3 b, float t) noexcept {
    return float3{lerp(a.x, b.x, t),
                  lerp(a.y, b.y, t),
                  lerp(a.z, b.z, t)};
}

[[nodiscard]] constexpr auto lerp(float4 a, float4 b, float t) noexcept {
    return float4{lerp(a.x, b.x, t),
                  lerp(a.y, b.y, t),
                  lerp(a.z, b.z, t),
                  lerp(a.w, b.w, t)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto clamp(T x, T a, T b) noexcept {
    return std::clamp(x, a, b);
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto clamp(Vector<T, 2> v, T a, T b) noexcept {
    return Vector<T, 2>{clamp(v.x, a, b), clamp(v.y, a, b)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto clamp(Vector<T, 3> v, T a, T b) noexcept {
    return Vector<T, 3>{clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)};
}

template<typename T, std::enable_if_t<is_scalar_v<T>, int> = 0>
[[nodiscard]] constexpr auto clamp(Vector<T, 4> v, T a, T b) noexcept {
    return Vector<T, 4>{clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)};
}

// Vector Functions
[[nodiscard]] constexpr auto dot(float2 u, float2 v) noexcept {
    return u.x * v.x + u.y * v.y;
}
[[nodiscard]] constexpr auto dot(float3 u, float3 v) noexcept {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
[[nodiscard]] constexpr auto dot(float4 u, float4 v) noexcept {
    return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
}

template<size_t N>
[[nodiscard]] constexpr auto length(Vector<float, N> u) noexcept {
    return sqrt(dot(u, u));
}

template<size_t N>
[[nodiscard]] constexpr auto normalize(Vector<float, N> u) noexcept {
    return u * (1.0f / length(u));
}

template<size_t N>
[[nodiscard]] constexpr auto distance(Vector<float, N> u, Vector<float, N> v) noexcept {
    return length(u - v);
}

[[nodiscard]] constexpr auto cross(float3 u, float3 v) noexcept {
    return float3{u.y * v.z - v.y * u.z,
                  u.z * v.x - v.z * u.x,
                  u.x * v.y - v.x * u.y};
}

// Matrix Functions
[[nodiscard]] constexpr auto transpose(const float2x2 m) noexcept {
    return make_float2x2(m[0].x, m[1].x,
                         m[0].y, m[1].y);
}

[[nodiscard]] constexpr auto transpose(const float3x3 m) noexcept {
    return make_float3x3(m[0].x, m[1].x, m[2].x,
                         m[0].y, m[1].y, m[2].y,
                         m[0].z, m[1].z, m[2].z);
}

[[nodiscard]] constexpr auto transpose(const float4x4 m) noexcept {
    return make_float4x4(m[0].x, m[1].x, m[2].x, m[3].x,
                         m[0].y, m[1].y, m[2].y, m[3].y,
                         m[0].z, m[1].z, m[2].z, m[3].z,
                         m[0].w, m[1].w, m[2].w, m[3].w);
}

[[nodiscard]] constexpr auto inverse(const float2x2 m) noexcept {
    const auto one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
    return make_float2x2(m[1][1] * one_over_determinant,
                         -m[0][1] * one_over_determinant,
                         -m[1][0] * one_over_determinant,
                         +m[0][0] * one_over_determinant);
}

[[nodiscard]] constexpr auto inverse(const float3x3 m) noexcept {// from GLM
    const auto one_over_determinant = 1.0f /
                                      (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
                                       m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
                                       m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
    return make_float3x3(
        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

[[nodiscard]] constexpr auto inverse(const float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = float4{coef00, coef00, coef02, coef03};
    const auto fac1 = float4{coef04, coef04, coef06, coef07};
    const auto fac2 = float4{coef08, coef08, coef10, coef11};
    const auto fac3 = float4{coef12, coef12, coef14, coef15};
    const auto fac4 = float4{coef16, coef16, coef18, coef19};
    const auto fac5 = float4{coef20, coef20, coef22, coef23};
    const auto Vec0 = float4{m[1].x, m[0].x, m[0].x, m[0].x};
    const auto Vec1 = float4{m[1].y, m[0].y, m[0].y, m[0].y};
    const auto Vec2 = float4{m[1].z, m[0].z, m[0].z, m[0].z};
    const auto Vec3 = float4{m[1].w, m[0].w, m[0].w, m[0].w};
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = float4{+1.0f, -1.0f, +1.0f, -1.0f};
    constexpr auto sign_b = float4{-1.0f, +1.0f, -1.0f, +1.0f};
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * float4{inv_0.x, inv_1.x, inv_2.x, inv_3.x};
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = 1.0f / dot1;
    return float4x4{inv_0 * one_over_determinant,
                    inv_1 * one_over_determinant,
                    inv_2 * one_over_determinant,
                    inv_3 * one_over_determinant};
}

[[nodiscard]] constexpr auto determinant(const float2x2 m) noexcept {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

[[nodiscard]] constexpr auto determinant(const float3x3 m) noexcept {// from GLM
    return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
           m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
           m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
}

[[nodiscard]] constexpr auto determinant(const float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = float4{coef00, coef00, coef02, coef03};
    const auto fac1 = float4{coef04, coef04, coef06, coef07};
    const auto fac2 = float4{coef08, coef08, coef10, coef11};
    const auto fac3 = float4{coef12, coef12, coef14, coef15};
    const auto fac4 = float4{coef16, coef16, coef18, coef19};
    const auto fac5 = float4{coef20, coef20, coef22, coef23};
    const auto Vec0 = float4{m[1].x, m[0].x, m[0].x, m[0].x};
    const auto Vec1 = float4{m[1].y, m[0].y, m[0].y, m[0].y};
    const auto Vec2 = float4{m[1].z, m[0].z, m[0].z, m[0].z};
    const auto Vec3 = float4{m[1].w, m[0].w, m[0].w, m[0].w};
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    constexpr auto sign_a = float4{+1.0f, -1.0f, +1.0f, -1.0f};
    constexpr auto sign_b = float4{-1.0f, +1.0f, -1.0f, +1.0f};
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * float4{inv_0.x, inv_1.x, inv_2.x, inv_3.x};
    return dot0.x + dot0.y + dot0.z + dot0.w;
}

// transforms
constexpr auto translation(const float3 v) noexcept {
    return make_float4x4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        v.x, v.y, v.z, 1.0f);
}

inline auto rotation(const float3 axis, float angle) noexcept {
    if (angle == 0.0f) { return make_float4x4(1.0f); }
    auto c = cos(angle);
    auto s = sin(angle);
    auto a = normalize(axis);
    auto t = (1.0f - c) * a;
    return make_float4x4(
        c + t.x * a.x, t.x * a.y + s * a.z, t.x * a.z - s * a.y, 0.0f,
        t.y * a.x - s * a.z, c + t.y * a.y, t.y * a.z + s * a.x, 0.0f,
        t.z * a.x + s * a.y, t.z * a.y - s * a.x, c + t.z * a.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

constexpr auto scaling(const float3 s) noexcept {
    return make_float4x4(
        s.x, 0.0f, 0.0f, 0.0f,
        0.0f, s.y, 0.0f, 0.0f,
        0.0f, 0.0f, s.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

constexpr auto scaling(float s) noexcept {
    return scaling(make_float3(s));
}

}// namespace luisa
