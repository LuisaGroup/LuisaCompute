//
// Created by Mike Smith on 2021/2/7.
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace luisa {

template<typename... T>
struct always_false : std::false_type {};

template<typename... T>
constexpr auto always_false_v = always_false<T...>::value;

template<typename... T>
struct always_true : std::true_type {};

template<typename... T>
constexpr auto always_true_v = always_true<T...>::value;

template<typename T, std::enable_if_t<std::disjunction_v<std::is_enum<T>>, int> = 0>
[[nodiscard]] constexpr auto to_underlying(T e) noexcept {
    return static_cast<std::underlying_type_t<T>>(e);
}

inline namespace size_literals {

[[nodiscard]] constexpr auto operator""_kb(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u);
}

[[nodiscard]] constexpr auto operator""_mb(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u * 1024u);
}

[[nodiscard]] constexpr auto operator""_gb(unsigned long long size) noexcept {
    return static_cast<size_t>(size * 1024u * 1024u * 1024u);
}

}// namespace size_literals

// scalars
using uint = unsigned int;

template<typename T>
using is_integral = std::disjunction<
    std::is_same<std::remove_cvref_t<T>, int>,
    std::is_same<std::remove_cvref_t<T>, uint>>;

template<typename T>
constexpr auto is_integral_v = is_integral<T>::value;

template<typename T>
using is_boolean = std::is_same<std::remove_cvref_t<T>, bool>;

template<typename T>
constexpr auto is_boolean_v = is_boolean<T>::value;

template<typename T>
using is_floating_point = std::is_same<std::remove_cvref_t<T>, float>;

template<typename T>
constexpr auto is_floating_point_v = is_floating_point<T>::value;

template<typename T>
using is_signed = std::disjunction<
    is_floating_point<T>,
    std::is_same<std::remove_cvref_t<T>, int>>;

template<typename T>
constexpr auto is_signed_v = is_signed<T>::value;

template<typename T>
using is_unsigned = std::is_same<std::remove_cvref_t<T>, uint>;

template<typename T>
constexpr auto is_unsigned_v = is_unsigned<T>::value;

template<typename T>
using is_scalar = std::disjunction<
    is_integral<T>,
    is_boolean<T>,
    is_floating_point<T>>;

template<typename T>
constexpr auto is_scalar_v = is_scalar<T>::value;

template<typename T, size_t N>
struct Vector;

// vectors
namespace detail {

template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    T x, y;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s} {}
    constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}

    template<typename U>
    constexpr VectorStorage(VectorStorage<U, 2> v) noexcept
        : VectorStorage{static_cast<T>(v.x),
                        static_cast<T>(v.y)} {}

#include <core/swizzle_2.inl.h>
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x, y, z;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s}, z{s} {}
    constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}

    template<typename U>
    constexpr VectorStorage(VectorStorage<U, 3> v) noexcept
        : VectorStorage{static_cast<T>(v.x),
                        static_cast<T>(v.y),
                        static_cast<T>(v.z)} {}
#include <core/swizzle_3.inl.h>
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x, y, z, w;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s}, z{s}, w{s} {}
    constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}

    template<typename U>
    constexpr VectorStorage(VectorStorage<U, 4> v) noexcept
        : VectorStorage{static_cast<T>(v.x),
                        static_cast<T>(v.y),
                        static_cast<T>(v.z),
                        static_cast<T>(v.w)} {}
#include <core/swizzle_4.inl.h>
};

}// namespace detail

template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {

    static constexpr auto dimension = N;
    using value_type = T;

    using Storage = detail::VectorStorage<T, N>;
    static_assert(std::disjunction_v<
                      std::is_same<T, bool>,
                      std::is_same<T, float>,
                      std::is_same<T, int>,
                      std::is_same<T, uint>> && (N == 2 || N == 3 || N == 4),
                  "Invalid vector type");

    using Storage::VectorStorage;

    [[nodiscard]] constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }

#define LUISA_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                              \
    template<typename U,                                                        \
             std::enable_if_t<                                                  \
                 std::conjunction_v<std::is_same<T, U>, __VA_ARGS__>,           \
                 int> = 0>                                                      \
    [[nodiscard]] constexpr auto operator op(Vector<U, N> rhs) const noexcept { \
        using R = Vector<                                                       \
            std::decay_t<decltype(static_cast<T>(0) op static_cast<T>(0))>, N>; \
        if constexpr (N == 2) {                                                 \
            return R{this->x op rhs.x,                                          \
                     this->y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                          \
            return R{this->x op rhs.x,                                          \
                     this->y op rhs.y,                                          \
                     this->z op rhs.z};                                         \
        } else {                                                                \
            return R{this->x op rhs.x,                                          \
                     this->y op rhs.y,                                          \
                     this->z op rhs.z,                                          \
                     this->w op rhs.w};                                         \
        }                                                                       \
    }                                                                           \
    template<typename U,                                                        \
             std::enable_if_t<                                                  \
                 std::conjunction_v<std::is_same<T, U>, __VA_ARGS__>,           \
                 int> = 0>                                                      \
    [[nodiscard]] constexpr auto operator op(U rhs) const noexcept {            \
        using R = Vector<                                                       \
            std::decay_t<decltype(static_cast<T>(0) op static_cast<T>(0))>, N>; \
        if constexpr (N == 2) {                                                 \
            return R{this->x op rhs,                                            \
                     this->y op rhs};                                           \
        } else if constexpr (N == 3) {                                          \
            return R{this->x op rhs,                                            \
                     this->y op rhs,                                            \
                     this->z op rhs};                                           \
        } else {                                                                \
            return R{this->x op rhs,                                            \
                     this->y op rhs,                                            \
                     this->z op rhs,                                            \
                     this->w op rhs};                                           \
        }                                                                       \
    }

#define LUISA_MAKE_VECTOR_ASSIGNMENT_OPERATOR(op, ...)                \
    template<typename U,                                              \
             std::enable_if_t<                                        \
                 std::conjunction_v<std::is_same<T, U>, __VA_ARGS__>, \
                 int> = 0>                                            \
    Vector &operator op(Vector<U, N> rhs) noexcept {                  \
        if constexpr (N == 2) {                                       \
            this->x op rhs.x;                                         \
            this->y op rhs.y;                                         \
        } else if constexpr (N == 3) {                                \
            this->x op rhs.x;                                         \
            this->y op rhs.y;                                         \
            this->z op rhs.z;                                         \
        } else {                                                      \
            this->x op rhs.x;                                         \
            this->y op rhs.y;                                         \
            this->z op rhs.z;                                         \
            this->w op rhs.w;                                         \
        }                                                             \
        return *this;                                                 \
    }                                                                 \
    template<typename U,                                              \
             std::enable_if_t<                                        \
                 std::conjunction_v<std::is_same<T, U>, __VA_ARGS__>, \
                 int> = 0>                                            \
    Vector &operator op(U rhs) noexcept {                             \
        if constexpr (N == 2) {                                       \
            this->x op rhs;                                           \
            this->y op rhs;                                           \
        } else if constexpr (N == 3) {                                \
            this->x op rhs;                                           \
            this->y op rhs;                                           \
            this->z op rhs;                                           \
        } else {                                                      \
            this->x op rhs;                                           \
            this->y op rhs;                                           \
            this->z op rhs;                                           \
            this->w op rhs;                                           \
        }                                                             \
        return *this;                                                 \
    }

#define LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(op, ...) \
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(op, __VA_ARGS__)             \
    LUISA_MAKE_VECTOR_ASSIGNMENT_OPERATOR(op## =, __VA_ARGS__)

    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(+, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(-, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(*, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(/, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(%, is_integral<T>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(<<, is_integral<T>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(>>, is_integral<T>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(|, std::negation<is_floating_point<T>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(&, std::negation<is_floating_point<T>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(^, std::negation<is_floating_point<T>>)

    LUISA_MAKE_VECTOR_BINARY_OPERATOR(||, is_boolean<T>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(&&, is_boolean<T>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(==, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(!=, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(<, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(>, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(<=, std::negation<is_boolean<T>>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(>=, std::negation<is_boolean<T>>)

#undef LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS
#undef LUISA_MAKE_VECTOR_ASSIGNMENT_OPERATOR
#undef LUISA_MAKE_VECTOR_BINARY_OPERATOR
};

#define LUISA_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;     \
    using T##3 = Vector<T, 3>;     \
    using T##4 = Vector<T, 4>;

LUISA_MAKE_VECTOR_TYPES(bool)
LUISA_MAKE_VECTOR_TYPES(float)
LUISA_MAKE_VECTOR_TYPES(int)
LUISA_MAKE_VECTOR_TYPES(uint)

#undef LUISA_MAKE_VECTOR_TYPES

template<size_t N>
struct Matrix {
    static_assert(always_false_v<std::integral_constant<size_t, N>>, "Invalid matrix type");
};

template<>
struct Matrix<2> {

    float2 cols[2];

    constexpr Matrix() noexcept
        : cols{float2{1.0f, 0.0f}, float2{0.0f, 1.0f}} {}

    constexpr Matrix(const float2 c0, const float2 c1) noexcept
        : cols{c0, c1} {}

    [[nodiscard]] constexpr float2 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float2 &operator[](size_t i) const noexcept { return cols[i]; }
};

template<>
struct Matrix<3> {

    float3 cols[3];

    constexpr Matrix() noexcept
        : cols{float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float3 c0, const float3 c1, const float3 c2) noexcept
        : cols{c0, c1, c2} {}

    [[nodiscard]] constexpr float3 &operator[](size_t i) noexcept { return cols[i]; }
    [[nodiscard]] constexpr const float3 &operator[](size_t i) const noexcept { return cols[i]; }
};

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

[[nodiscard]] constexpr auto any(const bool2 v) noexcept { return v.x || v.y; }
[[nodiscard]] constexpr auto any(const bool3 v) noexcept { return v.x || v.y || v.z; }
[[nodiscard]] constexpr auto any(const bool4 v) noexcept { return v.x || v.y || v.z || v.w; }

[[nodiscard]] constexpr auto all(const bool2 v) noexcept { return v.x && v.y; }
[[nodiscard]] constexpr auto all(const bool3 v) noexcept { return v.x && v.y && v.z; }
[[nodiscard]] constexpr auto all(const bool4 v) noexcept { return v.x && v.y && v.z && v.w; }

[[nodiscard]] constexpr auto none(const bool2 v) noexcept { return !any(v); }
[[nodiscard]] constexpr auto none(const bool3 v) noexcept { return !any(v); }
[[nodiscard]] constexpr auto none(const bool4 v) noexcept { return !any(v); }

}// namespace luisa

template<typename T, size_t N, std::enable_if_t<std::negation_v<luisa::is_boolean<T>>, int> = 0>
[[nodiscard]] constexpr auto operator+(const luisa::Vector<T, N> v) noexcept { return v; }

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

// scalar-vector binary operators
#define LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(op, ...)                               \
    template<typename T, size_t N,                                                      \
             std::enable_if_t<                                                          \
                 std::conjunction_v<luisa::is_scalar<T>, __VA_ARGS__>,                  \
                 int> = 0>                                                              \
    [[nodiscard]] constexpr auto operator op(T lhs, luisa::Vector<T, N> rhs) noexcept { \
        if constexpr (N == 2) {                                                         \
            return luisa::Vector<T, 2>{lhs, lhs} op rhs;                                \
        } else if constexpr (N == 3) {                                                  \
            return luisa::Vector<T, 3>{lhs, lhs, lhs} op rhs;                           \
        } else {                                                                        \
            return luisa::Vector<T, 4>{lhs, lhs, lhs, lhs} op rhs;                      \
        }                                                                               \
    }

LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(+, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(-, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(*, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(/, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(%, luisa::is_integral<T>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(<<, luisa::is_integral<T>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(>>, luisa::is_integral<T>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(|, std::negation<luisa::is_floating_point<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(&, std::negation<luisa::is_floating_point<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(^, std::negation<luisa::is_floating_point<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(||, luisa::is_boolean<T>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(&&, luisa::is_boolean<T>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(<, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(>, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(<=, std::negation<luisa::is_boolean<T>>)
LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR(>=, std::negation<luisa::is_boolean<T>>)

#undef LUISA_MAKE_SCALAR_VECTOR_BINARY_OPERATOR

// float2x2
[[nodiscard]] constexpr auto operator*(const luisa::float2x2 m, float s) noexcept {
    return luisa::float2x2{m[0] * s, m[1] * s};
}

[[nodiscard]] constexpr auto operator*(float s, const luisa::float2x2 m) noexcept {
    return m * s;
}

[[nodiscard]] constexpr auto operator/(const luisa::float2x2 m, float s) noexcept {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(const luisa::float2x2 m, const luisa::float2 v) noexcept {
    return v.x * m[0] + v.y * m[1];
}

[[nodiscard]] constexpr auto operator*(const luisa::float2x2 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x2{lhs * rhs[0], lhs * rhs[1]};
}

[[nodiscard]] constexpr auto operator+(const luisa::float2x2 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x2{lhs[0] + rhs[0], lhs[1] + rhs[1]};
}

[[nodiscard]] constexpr auto operator-(const luisa::float2x2 lhs, const luisa::float2x2 rhs) noexcept {
    return luisa::float2x2{lhs[0] - rhs[0], lhs[1] - rhs[1]};
}

// float3x3
[[nodiscard]] constexpr auto operator*(const luisa::float3x3 m, float s) noexcept {
    return luisa::float3x3{m[0] * s, m[1] * s, m[2] * s};
}

[[nodiscard]] constexpr auto operator*(float s, const luisa::float3x3 m) noexcept {
    return m * s;
}

[[nodiscard]] constexpr auto operator/(const luisa::float3x3 m, float s) noexcept {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(const luisa::float3x3 m, const luisa::float3 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2];
}

[[nodiscard]] constexpr auto operator*(const luisa::float3x3 lhs, const luisa::float3x3 rhs) noexcept {
    return luisa::float3x3{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

[[nodiscard]] constexpr auto operator+(const luisa::float3x3 lhs, const luisa::float3x3 rhs) noexcept {
    return luisa::float3x3{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

[[nodiscard]] constexpr auto operator-(const luisa::float3x3 lhs, const luisa::float3x3 rhs) noexcept {
    return luisa::float3x3{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

// float4x4
[[nodiscard]] constexpr auto operator*(const luisa::float4x4 m, float s) noexcept {
    return luisa::float4x4{m[0] * s, m[1] * s, m[2] * s, m[3] * s};
}

[[nodiscard]] constexpr auto operator*(float s, const luisa::float4x4 m) noexcept {
    return m * s;
}

[[nodiscard]] constexpr auto operator/(const luisa::float4x4 m, float s) noexcept {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(const luisa::float4x4 m, const luisa::float4 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
}

[[nodiscard]] constexpr auto operator*(const luisa::float4x4 lhs, const luisa::float4x4 rhs) noexcept {
    return luisa::float4x4{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
}

[[nodiscard]] constexpr auto operator+(const luisa::float4x4 lhs, const luisa::float4x4 rhs) noexcept {
    return luisa::float4x4{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
}

[[nodiscard]] constexpr auto operator-(const luisa::float4x4 lhs, const luisa::float4x4 rhs) noexcept {
    return luisa::float4x4{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
}

namespace luisa {

namespace detail {

template<typename T>
struct IsVector : std::false_type {};

template<typename T, size_t N>
struct IsVector<Vector<T, N>> : std::true_type {};

template<typename T, size_t N>
struct IsVectorN : std::false_type {};

template<typename T, size_t N>
struct IsVectorN<Vector<T, N>, N> : std::true_type {};

template<typename T>
struct IsMatrix : std::false_type {};

template<size_t N>
struct IsMatrix<Matrix<N>> : std::true_type {};

template<typename T, size_t N>
struct IsMatrixN : std::false_type {};

template<size_t N>
struct IsMatrixN<Matrix<N>, N> : std::true_type {};

template<typename T>
struct VectorValue {
    using type = T;
};

template<typename T, size_t N>
struct VectorValue<Vector<T, N>> {
    using type = T;
};

template<typename T>
struct VectorDim {
    static_assert(is_scalar_v<T>);
    static constexpr auto value = static_cast<size_t>(1u);
};

template<typename T, size_t N>
struct VectorDim<Vector<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
struct MatrixDim {
    static constexpr auto value = static_cast<size_t>(1u);
};

template<size_t N>
struct MatrixDim<Matrix<N>> {
    static constexpr auto value = N;
};

template<typename U, typename V>
struct VectorSameDim : std::false_type {};

template<typename U, typename V, size_t N>
struct VectorSameDim<Vector<U, N>, Vector<V, N>> : std::true_type {};

template<typename... T>
struct VectorAllSameDim : std::false_type {};

template<>
struct VectorAllSameDim<> : std::true_type {};

template<typename T>
struct VectorAllSameDim<T> : std::true_type {};

template<typename First, typename... Other>
struct VectorAllSameDim<First, Other...> : std::conjunction<VectorSameDim<First, Other>...> {};

}// namespace detail

template<typename... T>
using is_vector_same_dimension = detail::VectorAllSameDim<std::remove_cvref_t<T>...>;

template<typename... T>
constexpr auto is_vector_same_dimension_v = is_vector_same_dimension<T...>::value;

template<typename T>
using vector_dimension = detail::VectorDim<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto vector_dimension_v = vector_dimension<T>::value;

template<typename T>
using matrix_dimension = detail::MatrixDim<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto matrix_dimension_v = matrix_dimension<T>::value;

template<typename T>
using vector_element = detail::VectorValue<std::remove_cvref_t<T>>;

template<typename T>
using vector_element_t = typename vector_element<T>::type;

template<typename T>
using is_vector = detail::IsVector<T>;

template<typename T>
using is_vector2 = detail::IsVectorN<T, 2>;

template<typename T>
using is_vector3 = detail::IsVectorN<T, 3>;

template<typename T>
using is_vector4 = detail::IsVectorN<T, 4>;

template<typename T>
using is_bool_vector = std::conjunction<is_vector<T>, std::is_same<vector_element_t<T>, bool>>;

template<typename T>
constexpr auto is_bool_vector_v = is_bool_vector<T>::value;

template<typename T>
using is_float_vector = std::conjunction<is_vector<T>, std::is_same<vector_element_t<T>, float>>;

template<typename T>
constexpr auto is_float_vector_v = is_float_vector<T>::value;

template<typename T>
using is_int_vector = std::conjunction<is_vector<T>, std::is_same<vector_element_t<T>, int>>;

template<typename T>
constexpr auto is_int_vector_v = is_int_vector<T>::value;

template<typename T>
using is_uint_vector = std::conjunction<is_vector<T>, std::is_same<vector_element_t<T>, uint>>;

template<typename T>
constexpr auto is_uint_vector_v = is_uint_vector<T>::value;

template<typename T>
constexpr auto is_vector_v = is_vector<T>::value;

template<typename T>
constexpr auto is_vector2_v = is_vector2<T>::value;

template<typename T>
constexpr auto is_vector3_v = is_vector3<T>::value;

template<typename T>
constexpr auto is_vector4_v = is_vector4<T>::value;

template<typename T>
using is_matrix = detail::IsMatrix<T>;

template<typename T>
using is_matrix2 = detail::IsMatrixN<T, 2>;

template<typename T>
using is_matrix3 = detail::IsMatrixN<T, 3>;

template<typename T>
using is_matrix4 = detail::IsMatrixN<T, 4>;

template<typename T>
constexpr auto is_matrix_v = is_matrix<T>::value;

template<typename T>
constexpr auto is_matrix2_v = is_matrix2<T>::value;

template<typename T>
constexpr auto is_matrix3_v = is_matrix3<T>::value;

template<typename T>
constexpr auto is_matrix4_v = is_matrix4<T>::value;

template<typename T>
using is_basic = std::disjunction<is_scalar<T>, is_vector<T>, is_matrix<T>>;

template<typename T>
constexpr auto is_basic_v = is_basic<T>::value;

// make typeN

#define LUISA_MAKE_TYPE_N(type)                                                                                              \
    [[nodiscard]] constexpr auto make_##type##2(type s = {}) noexcept { return type##2(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##2(type x, type y) noexcept { return type##2(x, y); }                           \
    template<typename T>                                                                                                     \
    [[nodiscard]] constexpr auto make_##type##2(Vector<T, 2> v) noexcept { return type##2(v); }                              \
    [[nodiscard]] constexpr auto make_##type##2(type##3 v) noexcept { return type##2(v.x, v.y); }                            \
    [[nodiscard]] constexpr auto make_##type##2(type##4 v) noexcept { return type##2(v.x, v.y); }                            \
                                                                                                                             \
    [[nodiscard]] constexpr auto make_##type##3(type s = {}) noexcept { return type##3(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##3(type x, type y, type z) noexcept { return type##3(x, y, z); }                \
    template<typename T>                                                                                                     \
    [[nodiscard]] constexpr auto make_##type##3(Vector<T, 3> v) noexcept { return type##3(v); }                              \
    [[nodiscard]] constexpr auto make_##type##3(type##2 v, type z) noexcept { return type##3(v.x, v.y, z); }                 \
    [[nodiscard]] constexpr auto make_##type##3(type x, type##2 v) noexcept { return type##3(x, v.x, v.y); }                 \
    [[nodiscard]] constexpr auto make_##type##3(type##4 v) noexcept { return type##3(v.x, v.y, v.z); }                       \
                                                                                                                             \
    [[nodiscard]] constexpr auto make_##type##4(type s = {}) noexcept { return type##4(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type z, type w) noexcept { return type##4(x, y, z, w); }     \
    template<typename T>                                                                                                     \
    [[nodiscard]] constexpr auto make_##type##4(Vector<T, 4> v) noexcept { return type##4(v); }                              \
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

// make float2x2
[[nodiscard]] constexpr auto make_float2x2(float s = 1.0f) noexcept {
    return float2x2{float2{s, 0.0f},
                    float2{0.0f, s}};
}

[[nodiscard]] constexpr auto make_float2x2(
    float m00, float m01,
    float m10, float m11) noexcept {
    return float2x2{float2{m00, m01},
                    float2{m10, m11}};
}

[[nodiscard]] constexpr auto make_float2x2(float2 c0, float2 c1) noexcept {
    return float2x2{c0, c1};
}

[[nodiscard]] constexpr auto make_float2x2(float2x2 m) noexcept {
    return m;
}

[[nodiscard]] constexpr auto make_float2x2(float3x3 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

[[nodiscard]] constexpr auto make_float2x2(float4x4 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

// make float3x3
[[nodiscard]] constexpr auto make_float3x3(float s = 1.0f) noexcept {
    return float3x3{float3{s, 0.0f, 0.0f},
                    float3{0.0f, s, 0.0f},
                    float3{0.0f, 0.0f, s}};
}

[[nodiscard]] constexpr auto make_float3x3(float3 c0, float3 c1, float3 c2) noexcept {
    return float3x3{c0, c1, c2};
}

[[nodiscard]] constexpr auto make_float3x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22) noexcept {
    return float3x3{float3{m00, m01, m02},
                    float3{m10, m11, m12},
                    float3{m20, m21, m22}};
}

[[nodiscard]] constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{make_float3(m[0], 0.0f),
                    make_float3(m[1], 0.0f),
                    make_float3(m[2], 1.0f)};
}

[[nodiscard]] constexpr auto make_float3x3(float3x3 m) noexcept {
    return m;
}

[[nodiscard]] constexpr auto make_float3x3(float4x4 m) noexcept {
    return float3x3{make_float3(m[0]),
                    make_float3(m[1]),
                    make_float3(m[2])};
}

// make float4x4
[[nodiscard]] constexpr auto make_float4x4(float s = 1.0f) noexcept {
    return float4x4{float4{s, 0.0f, 0.0f, 0.0f},
                    float4{0.0f, s, 0.0f, 0.0f},
                    float4{0.0f, 0.0f, s, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, s}};
}

[[nodiscard]] constexpr auto make_float4x4(float4 c0, float4 c1, float4 c2, float4 c3) noexcept {
    return float4x4{c0, c1, c2, c3};
}

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

[[nodiscard]] constexpr auto make_float4x4(float2x2 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f, 0.0f),
                    make_float4(m[1], 0.0f, 0.0f),
                    float4{0.0f, 0.0f, 1.0f, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float3x3 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f),
                    make_float4(m[1], 0.0f),
                    make_float4(m[2], 0.0f),
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

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
