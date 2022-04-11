//
// Created by Mike Smith on 2021/2/7.
//

#pragma once

#include <core/basic_traits.h>
#include <serialize/key_value_pair.h>

namespace luisa {

inline namespace size_literals {

/// Calculate size of KB
[[nodiscard]] constexpr auto operator""_kb(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u);
}

/// Calculate size of MB
[[nodiscard]] constexpr auto operator""_mb(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u * 1024u);
}

/// Calculate size of GB
[[nodiscard]] constexpr auto operator""_gb(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u * 1024u * 1024u);
}

}// namespace size_literals

// vectors
namespace detail {

/// Vector storage only allows size of 2, 3, 4
template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

/// Vector storage of size 2
template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    T x, y;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s} {}
    constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
    template<typename S>
    void serialize(S& s) { s.serialize(MAKE_NAME_PAIR(x), MAKE_NAME_PAIR(y)); }
#include <core/swizzle_2.inl.h>
};

/// Vector storage of size 3
template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x, y, z;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s}, z{s} {}
    constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    template<typename S>
    void serialize(S& s) { s.serialize(MAKE_NAME_PAIR(x), MAKE_NAME_PAIR(y), MAKE_NAME_PAIR(z)); }
#include <core/swizzle_3.inl.h>
};

/// Vector storage of size 4
template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x, y, z, w;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s}, z{s}, w{s} {}
    constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    template<typename S>
    void serialize(S& s) { s.serialize(MAKE_NAME_PAIR(x), MAKE_NAME_PAIR(y), MAKE_NAME_PAIR(z), MAKE_NAME_PAIR(w)); }
#include <core/swizzle_4.inl.h>
};

}// namespace detail

/**
 * @brief Vector class
 * 
 * We only support vector of size 2, 3, 4 and type bool, float, int, uint.
 * Any other kind of template parameters will fail on compilation.
 * 
 * @tparam T bool/float/int/uint
 * @tparam N 2/3/4
 */
template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {
    static constexpr auto dimension = N;
    using value_type = T;
    using Storage = detail::VectorStorage<T, N>;
    static_assert(std::disjunction_v<
                      std::is_same<T, bool>,
                      std::is_same<T, float>,
                      std::is_same<T, int>,
                      std::is_same<T, uint>>,
                  "Invalid vector type");
    static_assert(N == 2 || N == 3 || N == 4, "Invalid vector dimension");
    using Storage::VectorStorage;
    [[nodiscard]] constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
    using detail::VectorStorage<T, N>::serialize;
};

/**
 * @brief Make vector types of size 2, 3, 4
 *  
 * Ex. LUISA_MAKE_VECTOR_TYPES(int) -> int2, int3, int4
 */
#define LUISA_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;     \
    using T##3 = Vector<T, 3>;     \
    using T##4 = Vector<T, 4>;

LUISA_MAKE_VECTOR_TYPES(bool)
LUISA_MAKE_VECTOR_TYPES(float)
LUISA_MAKE_VECTOR_TYPES(int)
LUISA_MAKE_VECTOR_TYPES(uint)

#undef LUISA_MAKE_VECTOR_TYPES

/// Matrix only allows size of 2, 3, 4
template<size_t N>
struct Matrix {
    static_assert(always_false_v<std::integral_constant<size_t, N>>, "Invalid matrix type");
};

/// 2x2 matrix
template<>
struct Matrix<2> {

    float2 cols[2];

    constexpr Matrix() noexcept
        : cols{float2{1.0f, 0.0f}, float2{0.0f, 1.0f}} {}

    constexpr Matrix(const float2 c0, const float2 c1) noexcept
        : cols{c0, c1} {}

    [[nodiscard]] constexpr float2 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float2 &operator[](size_t i) const noexcept { return cols[i]; }

    template<typename S>
    void serialize(S& s){
        s.serialize(
            MAKE_NAME_PAIR(cols[0]),
            MAKE_NAME_PAIR(cols[1])
        );
    }
};

/// 3x3 matrix
template<>
struct Matrix<3> {

    float3 cols[3];

    constexpr Matrix() noexcept
        : cols{float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float3 c0, const float3 c1, const float3 c2) noexcept
        : cols{c0, c1, c2} {}

    [[nodiscard]] constexpr float3 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float3 &operator[](size_t i) const noexcept { return cols[i]; }

    template<typename S>
    void serialize(S& s){
        s.serialize(
            MAKE_NAME_PAIR(cols[0]),
            MAKE_NAME_PAIR(cols[1]),
            MAKE_NAME_PAIR(cols[2])
        );
    }
};

/// 4x4 matrix
template<>
struct Matrix<4> {

    float4 cols[4];

    constexpr Matrix() noexcept
        : cols{float4{1.0f, 0.0f, 0.0f, 0.0f},
               float4{0.0f, 1.0f, 0.0f, 0.0f},
               float4{0.0f, 0.0f, 1.0f, 0.0f},
               float4{0.0f, 0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float4 c0, const float4 c1, const float4 c2, const float4 c3) noexcept
        : cols{c0, c1, c2, c3} {}

    [[nodiscard]] constexpr float4 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float4 &operator[](size_t i) const noexcept { return cols[i]; }

    template<typename S>
    void serialize(S& s){
        s.serialize(
            MAKE_NAME_PAIR(cols[0]),
            MAKE_NAME_PAIR(cols[1]),
            MAKE_NAME_PAIR(cols[2]),
            MAKE_NAME_PAIR(cols[3])
        );
    }
};

using float2x2 = Matrix<2>;
using float3x3 = Matrix<3>;
using float4x4 = Matrix<4>;

using basic_types = std::tuple<
    bool, float, int, uint,
    bool2, float2, int2, uint2,
    bool3, float3, int3, uint3,
    bool4, float4, int4, uint4,
    float2x2, float3x3, float4x4>;

/// any of bool2 is true
[[nodiscard]] constexpr auto any(const bool2 v) noexcept { return v.x || v.y; }
/// any of bool3 is true
[[nodiscard]] constexpr auto any(const bool3 v) noexcept { return v.x || v.y || v.z; }
/// any of bool4 is true
[[nodiscard]] constexpr auto any(const bool4 v) noexcept { return v.x || v.y || v.z || v.w; }

/// all of bool2 is true
[[nodiscard]] constexpr auto all(const bool2 v) noexcept { return v.x && v.y; }
/// all of bool3 is true
[[nodiscard]] constexpr auto all(const bool3 v) noexcept { return v.x && v.y && v.z; }
/// all of bool4 is true
[[nodiscard]] constexpr auto all(const bool4 v) noexcept { return v.x && v.y && v.z && v.w; }

/// none of bool2 is true
[[nodiscard]] constexpr auto none(const bool2 v) noexcept { return !any(v); }
/// none of bool3 is true
[[nodiscard]] constexpr auto none(const bool3 v) noexcept { return !any(v); }
/// none of bool4 is true
[[nodiscard]] constexpr auto none(const bool4 v) noexcept { return !any(v); }

}// namespace luisa

/// Unary plus operator of vector. Bool vector is not allowed.
template<typename T, size_t N, std::enable_if_t<std::negation_v<luisa::is_boolean<T>>, int> = 0>
[[nodiscard]] constexpr auto operator+(const luisa::Vector<T, N> v) noexcept { return v; }

/// Unary minus operator of vector. Bool vector is not allowed.
template<typename T, size_t N, std::enable_if_t<std::negation_v<luisa::is_boolean<T>>, int> = 0>
[[nodiscard]] constexpr auto operator-(const luisa::Vector<T, N> v) noexcept {
    using R = luisa::Vector<T, N>;
    if constexpr (N == 2) {
        return R{-v.x, -v.y};
    } else if constexpr (N == 3) {
        return R{-v.x, -v.y, -v.z};
    } else {
        return R{-v.x, -v.y, -v.z, -v.w};
    }
}

/// Unary not operator of vector
template<typename T, size_t N>
[[nodiscard]] constexpr auto operator!(const luisa::Vector<T, N> v) noexcept {
    if constexpr (N == 2u) {
        return luisa::bool2{!v.x, !v.y};
    } else if constexpr (N == 3u) {
        return luisa::bool3{!v.x, !v.y, !v.z};
    } else {
        return luisa::bool3{!v.x, !v.y, !v.z, !v.w};
    }
}

/// Unary xor operator of vector. Only int vector is allowed.
template<typename T, size_t N,
         std::enable_if_t<luisa::is_integral_v<T>, int> = 0>
[[nodiscard]] constexpr auto operator~(const luisa::Vector<T, N> v) noexcept {
    using R = luisa::Vector<T, N>;
    if constexpr (N == 2) {
        return R{~v.x, ~v.y};
    } else if constexpr (N == 3) {
        return R{~v.x, ~v.y, ~v.z};
    } else {
        return R{~v.x, ~v.y, ~v.z, ~v.w};
    }
}

/// Binary & Assignment operators
#define LUISA_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                                      \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0>              \
    [[nodiscard]] constexpr auto operator op(                                           \
        luisa::Vector<T, N> lhs, luisa::Vector<T, N> rhs) noexcept {                    \
        if constexpr (N == 2) {                                                         \
            return luisa::Vector<T, 2>{                                                 \
                lhs.x op rhs.x,                                                         \
                lhs.y op rhs.y};                                                        \
        } else if constexpr (N == 3) {                                                  \
            return luisa::Vector<T, 3>{                                                 \
                lhs.x op rhs.x,                                                         \
                lhs.y op rhs.y,                                                         \
                lhs.z op rhs.z};                                                        \
        } else {                                                                        \
            return luisa::Vector<T, 4>{                                                 \
                lhs.x op rhs.x,                                                         \
                lhs.y op rhs.y,                                                         \
                lhs.z op rhs.z,                                                         \
                lhs.w op rhs.w};                                                        \
        }                                                                               \
    }                                                                                   \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0>              \
    [[nodiscard]] constexpr auto operator op(luisa::Vector<T, N> lhs, T rhs) noexcept { \
        return lhs op luisa::Vector<T, N>{rhs};                                         \
    }                                                                                   \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0>              \
    [[nodiscard]] constexpr auto operator op(T lhs, luisa::Vector<T, N> rhs) noexcept { \
        return luisa::Vector<T, N>{lhs} op rhs;                                         \
    }
LUISA_MAKE_VECTOR_BINARY_OPERATOR(+, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(-, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(*, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(/, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(%, luisa::is_integral_v<T>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(<<, luisa::is_integral_v<T>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(>>, luisa::is_integral_v<T>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(|, std::negation_v<luisa::is_floating_point<T>>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(&, std::negation_v<luisa::is_floating_point<T>>)
LUISA_MAKE_VECTOR_BINARY_OPERATOR(^, std::negation_v<luisa::is_floating_point<T>>)

/// Binary logic operators
#define LUISA_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                                       \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0>              \
    [[nodiscard]] constexpr auto operator op(                                           \
        luisa::Vector<T, N> lhs, luisa::Vector<T, N> rhs) noexcept {                    \
        if constexpr (N == 2) {                                                         \
            return luisa::bool2{                                                        \
                lhs.x op rhs.x,                                                         \
                lhs.y op rhs.y};                                                        \
        } else if constexpr (N == 3) {                                                  \
            return luisa::bool3{                                                        \
                lhs.x op rhs.x,                                                         \
                lhs.y op rhs.y,                                                         \
                lhs.z op rhs.z};                                                        \
        } else {                                                                        \
            return luisa::bool4{                                                        \
                lhs.x op rhs.x,                                                         \
                lhs.y op rhs.y,                                                         \
                lhs.z op rhs.z,                                                         \
                lhs.w op rhs.w};                                                        \
        }                                                                               \
    }                                                                                   \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0>              \
    [[nodiscard]] constexpr auto operator op(luisa::Vector<T, N> lhs, T rhs) noexcept { \
        return lhs op luisa::Vector<T, N>{rhs};                                         \
    }                                                                                   \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0>              \
    [[nodiscard]] constexpr auto operator op(T lhs, luisa::Vector<T, N> rhs) noexcept { \
        return luisa::Vector<T, N>{lhs} op rhs;                                         \
    }
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(||, luisa::is_boolean_v<T>)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(&&, luisa::is_boolean_v<T>)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(==, true)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(!=, true)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(<, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(>, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(<=, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_LOGIC_OPERATOR(>=, std::negation_v<luisa::is_boolean<T>>)

/// Assign operators
#define LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(op, ...)                         \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0> \
    constexpr decltype(auto) operator op(                                  \
        luisa::Vector<T, N> &lhs, luisa::Vector<T, N> rhs) noexcept {      \
        lhs.x op rhs.x;                                                    \
        lhs.y op rhs.y;                                                    \
        if constexpr (N >= 3) { lhs.z op rhs.z; }                          \
        if constexpr (N == 4) { lhs.w op rhs.w; }                          \
        return (lhs);                                                      \
    }                                                                      \
    template<typename T, size_t N, std::enable_if_t<__VA_ARGS__, int> = 0> \
    constexpr decltype(auto) operator op(                                  \
        luisa::Vector<T, N> &lhs, T rhs) noexcept {                        \
        return (lhs op luisa::Vector<T, N>{rhs});                          \
    }
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(+=, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(-=, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(*=, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(/=, std::negation_v<luisa::is_boolean<T>>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(%=, luisa::is_integral_v<T>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(<<=, luisa::is_integral_v<T>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(>>=, luisa::is_integral_v<T>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(|=, std::negation_v<luisa::is_floating_point<T>>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(&=, std::negation_v<luisa::is_floating_point<T>>)
LUISA_MAKE_VECTOR_ASSIGN_OPERATOR(^=, std::negation_v<luisa::is_floating_point<T>>)

#undef LUISA_MAKE_VECTOR_BINARY_OPERATOR
#undef LUISA_MAKE_VECTOR_LOGIC_OPERATOR
#undef LUISA_MAKE_VECTOR_ASSIGN_OPERATOR

/// float2x2 multiplied by float
[[nodiscard]] constexpr auto operator*(const luisa::float2x2 m, float s) noexcept {
    return luisa::float2x2{m[0] * s, m[1] * s};
}

/// float2x2 multiplied by float
[[nodiscard]] constexpr auto operator*(float s, const luisa::float2x2 m) noexcept {
    return m * s;
}

/// float2x2 divided by float
[[nodiscard]] constexpr auto operator/(const luisa::float2x2 m, float s) noexcept {
    return m * (1.0f / s);
}

/// float2x2 dot float2
[[nodiscard]] constexpr auto operator*(const luisa::float2x2 m, const luisa::float2 v) noexcept {
    return v.x * m[0] + v.y * m[1];
}

/// float2x2 multiply(matmul)
[[nodiscard]] constexpr auto operator*(const luisa::float2x2 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x2{lhs * rhs[0], lhs * rhs[1]};
}

/// float2x2 plus
[[nodiscard]] constexpr auto operator+(const luisa::float2x2 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x2{lhs[0] + rhs[0], lhs[1] + rhs[1]};
}

/// float2x2 minus
[[nodiscard]] constexpr auto operator-(const luisa::float2x2 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x2{lhs[0] - rhs[0], lhs[1] - rhs[1]};
}

/// float3x3 multiplied by float
[[nodiscard]] constexpr auto operator*(const luisa::float3x3 m, float s) noexcept {
    return luisa::float3x3{m[0] * s, m[1] * s, m[2] * s};
}

/// float3x3 multiplied by float
[[nodiscard]] constexpr auto operator*(float s, const luisa::float3x3 m) noexcept {
    return m * s;
}

/// float 3x3 divided by float
[[nodiscard]] constexpr auto operator/(const luisa::float3x3 m, float s) noexcept {
    return m * (1.0f / s);
}

/// float3x3 dot float3
[[nodiscard]] constexpr auto operator*(const luisa::float3x3 m, const luisa::float3 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2];
}

/// float3x3 multiply(matmul)
[[nodiscard]] constexpr auto operator*(const luisa::float3x3 lhs, const luisa::float3x3 rhs) noexcept {
    return luisa::float3x3{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

/// float3x3 plus
[[nodiscard]] constexpr auto operator+(const luisa::float3x3 lhs, const luisa::float3x3 rhs) noexcept {
    return luisa::float3x3{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

/// float3x3 minus
[[nodiscard]] constexpr auto operator-(const luisa::float3x3 lhs, const luisa::float3x3 rhs) noexcept {
    return luisa::float3x3{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

/// float4x4 multiplied by float
[[nodiscard]] constexpr auto operator*(const luisa::float4x4 m, float s) noexcept {
    return luisa::float4x4{m[0] * s, m[1] * s, m[2] * s, m[3] * s};
}

/// float4x4 multiplied by float
[[nodiscard]] constexpr auto operator*(float s, const luisa::float4x4 m) noexcept {
    return m * s;
}

/// float4x4 divided by float
[[nodiscard]] constexpr auto operator/(const luisa::float4x4 m, float s) noexcept {
    return m * (1.0f / s);
}

/// floa4x4 dot float4
[[nodiscard]] constexpr auto operator*(const luisa::float4x4 m, const luisa::float4 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
}

/// float4x4 multiply(matmul)
[[nodiscard]] constexpr auto operator*(const luisa::float4x4 lhs, const luisa::float4x4 rhs) noexcept {
    return luisa::float4x4{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
}

/// float4x4 plus
[[nodiscard]] constexpr auto operator+(const luisa::float4x4 lhs, const luisa::float4x4 rhs) noexcept {
    return luisa::float4x4{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
}

/// float4x4 minus
[[nodiscard]] constexpr auto operator-(const luisa::float4x4 lhs, const luisa::float4x4 rhs) noexcept {
    return luisa::float4x4{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
}

namespace luisa {

/// make_typeN definitions
#define LUISA_MAKE_TYPE_N(type)                                                                                              \
    [[nodiscard]] constexpr auto make_##type##2(type s = {}) noexcept { return type##2(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##2(type x, type y) noexcept { return type##2(x, y); }                           \
    template<typename T>                                                                                                     \
    [[nodiscard]] constexpr auto make_##type##2(Vector<T, 2> v) noexcept {                                                   \
        return type##2(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y));                                                                                         \
    }                                                                                                                        \
    [[nodiscard]] constexpr auto make_##type##2(type##3 v) noexcept { return type##2(v.x, v.y); }                            \
    [[nodiscard]] constexpr auto make_##type##2(type##4 v) noexcept { return type##2(v.x, v.y); }                            \
                                                                                                                             \
    [[nodiscard]] constexpr auto make_##type##3(type s = {}) noexcept { return type##3(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##3(type x, type y, type z) noexcept { return type##3(x, y, z); }                \
    template<typename T>                                                                                                     \
    [[nodiscard]] constexpr auto make_##type##3(Vector<T, 3> v) noexcept {                                                   \
        return type##3(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y),                                                                                          \
            static_cast<type>(v.z));                                                                                         \
    }                                                                                                                        \
    [[nodiscard]] constexpr auto make_##type##3(type##2 v, type z) noexcept { return type##3(v.x, v.y, z); }                 \
    [[nodiscard]] constexpr auto make_##type##3(type x, type##2 v) noexcept { return type##3(x, v.x, v.y); }                 \
    [[nodiscard]] constexpr auto make_##type##3(type##4 v) noexcept { return type##3(v.x, v.y, v.z); }                       \
                                                                                                                             \
    [[nodiscard]] constexpr auto make_##type##4(type s = {}) noexcept { return type##4(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type z, type w) noexcept { return type##4(x, y, z, w); }     \
    template<typename T>                                                                                                     \
    [[nodiscard]] constexpr auto make_##type##4(Vector<T, 4> v) noexcept {                                                   \
        return type##4(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y),                                                                                          \
            static_cast<type>(v.z),                                                                                          \
            static_cast<type>(v.w));                                                                                         \
    }                                                                                                                        \
    [[nodiscard]] constexpr auto make_##type##4(type##2 v, type z, type w) noexcept { return type##4(v.x, v.y, z, w); }      \
    [[nodiscard]] constexpr auto make_##type##4(type x, type##2 v, type w) noexcept { return type##4(x, v.x, v.y, w); }      \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type##2 v) noexcept { return type##4(x, y, v.x, v.y); }      \
    [[nodiscard]] constexpr auto make_##type##4(type##2 xy, type##2 zw) noexcept { return type##4(xy.x, xy.y, zw.x, zw.y); } \
    [[nodiscard]] constexpr auto make_##type##4(type##3 v, type w) noexcept { return type##4(v.x, v.y, v.z, w); }            \
    [[nodiscard]] constexpr auto make_##type##4(type x, type##3 v) noexcept { return type##4(x, v.x, v.y, v.z); }

LUISA_MAKE_TYPE_N(bool)
LUISA_MAKE_TYPE_N(float)
LUISA_MAKE_TYPE_N(int)
LUISA_MAKE_TYPE_N(uint)
#undef LUISA_MAKE_TYPE_N

/// make float2x2
[[nodiscard]] constexpr auto make_float2x2(float s = 1.0f) noexcept {
    return float2x2{float2{s, 0.0f},
                    float2{0.0f, s}};
}

/// make float2x2
[[nodiscard]] constexpr auto make_float2x2(
    float m00, float m01,
    float m10, float m11) noexcept {
    return float2x2{float2{m00, m01},
                    float2{m10, m11}};
}

/// make float2x2
[[nodiscard]] constexpr auto make_float2x2(float2 c0, float2 c1) noexcept {
    return float2x2{c0, c1};
}

/// make float2x2
[[nodiscard]] constexpr auto make_float2x2(float2x2 m) noexcept {
    return m;
}

/// make float2x2
[[nodiscard]] constexpr auto make_float2x2(float3x3 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

/// make float2x2
[[nodiscard]] constexpr auto make_float2x2(float4x4 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

/// make float3x3
[[nodiscard]] constexpr auto make_float3x3(float s = 1.0f) noexcept {
    return float3x3{float3{s, 0.0f, 0.0f},
                    float3{0.0f, s, 0.0f},
                    float3{0.0f, 0.0f, s}};
}

/// make float3x3
[[nodiscard]] constexpr auto make_float3x3(float3 c0, float3 c1, float3 c2) noexcept {
    return float3x3{c0, c1, c2};
}

/// make float3x3
[[nodiscard]] constexpr auto make_float3x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22) noexcept {
    return float3x3{float3{m00, m01, m02},
                    float3{m10, m11, m12},
                    float3{m20, m21, m22}};
}

/// make float3x3
[[nodiscard]] constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{make_float3(m[0], 0.0f),
                    make_float3(m[1], 0.0f),
                    make_float3(m[2], 1.0f)};
}

/// make float3x3
[[nodiscard]] constexpr auto make_float3x3(float3x3 m) noexcept {
    return m;
}

/// make float3x3
[[nodiscard]] constexpr auto make_float3x3(float4x4 m) noexcept {
    return float3x3{make_float3(m[0]),
                    make_float3(m[1]),
                    make_float3(m[2])};
}

/// make float4x4
[[nodiscard]] constexpr auto make_float4x4(float s = 1.0f) noexcept {
    return float4x4{float4{s, 0.0f, 0.0f, 0.0f},
                    float4{0.0f, s, 0.0f, 0.0f},
                    float4{0.0f, 0.0f, s, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, s}};
}

/// make float4x4
[[nodiscard]] constexpr auto make_float4x4(float4 c0, float4 c1, float4 c2, float4 c3) noexcept {
    return float4x4{c0, c1, c2, c3};
}

/// make float4x4
[[nodiscard]] constexpr auto make_float4x4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33) noexcept {
    return float4x4{float4{m00, m01, m02, m03},
                    float4{m10, m11, m12, m13},
                    float4{m20, m21, m22, m23},
                    float4{m30, m31, m32, m33}};
}

/// make float4x4
[[nodiscard]] constexpr auto make_float4x4(float2x2 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f, 0.0f),
                    make_float4(m[1], 0.0f, 0.0f),
                    float4{0.0f, 0.0f, 1.0f, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

/// make float4x4
[[nodiscard]] constexpr auto make_float4x4(float3x3 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f),
                    make_float4(m[1], 0.0f),
                    make_float4(m[2], 0.0f),
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

/// make float4x4
[[nodiscard]] constexpr auto make_float4x4(float4x4 m) noexcept {
    return m;
}

}// namespace luisa

//template<size_t N>
//constexpr auto operator*(std::array<float, N> a, float s) noexcept {
//    std::array<float, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = a[i] * s; }
//    return a;
//}
//
//template<size_t N>
//constexpr auto operator*(float s, std::array<float, N> a) noexcept {
//    return a * s;
//}
//
//template<size_t N>
//constexpr auto operator*(std::array<float, N> lhs, std::array<float, N> rhs) noexcept {
//    std::array<float, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = lhs[i] * rhs[i]; }
//    return r;
//}
//
//template<size_t N>
//constexpr auto operator+(std::array<float, N> lhs, std::array<float, N> rhs) noexcept {
//    std::array<float, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = lhs[i] + rhs[i]; }
//    return r;
//}
//
//template<size_t N>
//constexpr auto operator-(std::array<float, N> lhs, std::array<float, N> rhs) noexcept {
//    std::array<float, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = lhs[i] - rhs[i]; }
//    return r;
//}
//
//template<size_t M, size_t N>
//constexpr auto operator*(std::array<std::array<float, M>, N> m, float s) noexcept {
//    std::array<std::array<float, M>, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = m[i] * s; }
//}
//
//template<size_t M, size_t N>
//constexpr auto operator*(float s, std::array<std::array<float, M>, N> m) noexcept {
//    return m * s;
//}
//
//template<size_t M, size_t N>
//constexpr auto operator*(std::array<std::array<float, M>, N> m, std::array<float, N> v) noexcept {
//    std::array<float, N> r{};
//    for (auto i = 0u; i < N; i++) { r = r + m[i] * v[i]; }
//}
//
//template<size_t M, size_t N, size_t P>
//constexpr auto operator*(std::array<std::array<float, M>, N> lhs, std::array<std::array<float, N>, P> rhs) noexcept {
//    std::array<std::array<float, M>, P> r;
//    for (auto i = 0u; i < P; i++) { r[i] = lhs * rhs[i]; }
//    return r;
//}
//
//template<size_t M, size_t N>
//constexpr auto operator+(std::array<std::array<float, M>, N> lhs, std::array<std::array<float, M>, N> rhs) noexcept {
//    std::array<std::array<float, M>, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = lhs + rhs[i]; }
//    return r;
//}
//
//template<size_t M, size_t N>
//constexpr auto operator-(std::array<std::array<float, M>, N> lhs, std::array<std::array<float, M>, N> rhs) noexcept {
//    std::array<std::array<float, M>, N> r;
//    for (auto i = 0u; i < N; i++) { r[i] = lhs - rhs[i]; }
//    return r;
//}
