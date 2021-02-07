//
// Created by Mike Smith on 2021/2/7.
//

#pragma once

#include <cstdint>
#include <type_traits>

#include <core/concepts.h>

namespace luisa {

// scalars
using uchar = uint8_t;
using ushort = uint16_t;
using uint = uint32_t;

// vectors
namespace detail {

template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    T x, y;
    constexpr VectorStorage() noexcept: x{}, y{} {}
    constexpr VectorStorage(const VectorStorage &v) noexcept: x{v.x}, y{v.y} {}
    explicit constexpr VectorStorage(T s) noexcept: x{s}, y{s} {}
    explicit constexpr VectorStorage(T x, T y) noexcept: x{x}, y{y} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x, y, z;
    constexpr VectorStorage() noexcept: x{}, y{}, z{} {}
    constexpr VectorStorage(const VectorStorage &v) noexcept: x{v.x}, y{v.y}, z{v.z} {}
    explicit constexpr VectorStorage(T s) noexcept: x{s}, y{s}, z{s} {}
    explicit constexpr VectorStorage(T x, T y, T z) noexcept: x{x}, y{y}, z{z} {}
    explicit constexpr VectorStorage(VectorStorage<T, 2> xy, T z) noexcept: x{xy.x}, y{xy.y}, z{z} {}
    explicit constexpr VectorStorage(T x, VectorStorage<T, 2> yz) noexcept: x{x}, y{yz.x}, z{yz.y} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x, y, z, w;
    constexpr VectorStorage() noexcept: x{}, y{}, z{}, w{} {}
    constexpr VectorStorage(const VectorStorage &v) noexcept: x{v.x}, y{v.y}, z{v.z}, w{v.w} {}
    explicit constexpr VectorStorage(T s) noexcept: x{s}, y{s}, z{s}, w{s} {}
    explicit constexpr VectorStorage(T x, T y, T z, T w) noexcept: x{x}, y{y}, z{z}, w{w} {}
    explicit constexpr VectorStorage(VectorStorage<T, 2> xy, T z, T w) noexcept: x{xy.x}, y{xy.y}, z{z}, w{w} {}
    explicit constexpr VectorStorage(VectorStorage<T, 2> xy, VectorStorage<T, 2> zw) noexcept: x{xy.x}, y{xy.y}, z{zw.x}, w{zw.y} {}
    explicit constexpr VectorStorage(T x, VectorStorage<T, 2> yz, T w) noexcept: x{x}, y{yz.x}, z{yz.y}, w{w} {}
    explicit constexpr VectorStorage(T x, T y, VectorStorage<T, 2> zw) noexcept: x{x}, y{y}, z{zw.x}, w{zw.y} {}
    explicit constexpr VectorStorage(VectorStorage<T, 3> xyz, T w) noexcept: x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {}
    explicit constexpr VectorStorage(T x, VectorStorage<T, 3> yzw) noexcept: x{x}, y{yzw.x}, z{yzw.y}, w{yzw.z} {}
};

}

template<typename T, size_t N>
struct Vector;

template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {
    
    using Storage = detail::VectorStorage<T, N>;
    static_assert(std::disjunction_v<
                      std::is_same<T, bool>,
                      std::is_same<T, float>,
                      std::is_same<T, char>, std::is_same<T, uchar>,
                      std::is_same<T, short>, std::is_same<T, ushort>,
                      std::is_same<T, int>, std::is_same<T, uint>> &&
                  (N == 2 || N == 3 || N == 4),
                  "Invalid vector type");
    
    template<typename ...Args>
    explicit constexpr Vector(Args ...args) noexcept : Storage{args...} {}
    
    constexpr Vector(const Vector &v) noexcept: Storage{v} {}
    
    [[nodiscard]] T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }

#define LUISA_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                                                        \
    template<typename U, std::enable_if_t<std::conjunction_v<                                             \
        std::is_same<T, U>,                                                                               \
        __VA_ARGS__>, int> = 0>                                                                           \
    [[nodiscard]] constexpr auto operator op(Vector<U, N> rhs) const noexcept {                           \
        using R = Vector<std::decay_t<decltype(static_cast<T>(0) op static_cast<T>(0))>, N>;              \
        if constexpr (N == 2) { return R{this->x op rhs.x, this->y op rhs.y}; }                           \
        else if constexpr (N == 3) { return R{this->x op rhs.x, this->y op rhs.y, this->z op rhs.z}; }    \
        else { return R{this->x op rhs.x, this->y op rhs.y, this->z op rhs.z, this->w op rhs.w}; }        \
    }

#define LUISA_MAKE_VECTOR_ASSIGNMENT_OPERATOR(op, ...)                                                    \
    template<typename U, std::enable_if_t<std::conjunction_v<                                             \
        std::is_same<T, U>,                                                                               \
        __VA_ARGS__>, int> = 0>                                                                           \
    Vector &operator op(Vector<U, N> rhs) noexcept {                                                      \
        if constexpr (N == 2) {                                                                           \
            this->x op rhs.x;                                                                             \
            this->y op rhs.y;                                                                             \
        } else if constexpr (N == 3) {                                                                    \
            this->x op rhs.x;                                                                             \
            this->y op rhs.y;                                                                             \
            this->z op rhs.z;                                                                             \
        } else {                                                                                          \
            this->x op rhs.x;                                                                             \
            this->y op rhs.y;                                                                             \
            this->z op rhs.z;                                                                             \
            this->w op rhs.w;                                                                             \
        }                                                                                                 \
        return *this;                                                                                     \
    }

#define LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(op, ...)  \
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(op, __VA_ARGS__)              \
    LUISA_MAKE_VECTOR_ASSIGNMENT_OPERATOR(op##=, __VA_ARGS__)
    
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(+, std::negation<std::is_same<T, bool>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(-, std::negation<std::is_same<T, bool>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(*, std::negation<std::is_same<T, bool>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(/, std::negation<std::is_same<T, bool>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(%, std::negation<std::disjunction<std::is_same<T, bool>, std::is_same<T, float>>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(<<, std::negation<std::disjunction<std::is_same<T, bool>, std::is_same<T, float>>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(>>, std::negation<std::disjunction<std::is_same<T, bool>, std::is_same<T, float>>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(|, std::negation<std::is_same<T, float>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(&, std::negation<std::is_same<T, float>>)
    LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS(^, std::negation<std::is_same<T, float>>)
    
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(||, std::is_same<T, bool>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(&&, std::is_same<T, bool>)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(==, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(!=, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(<, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(>, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(<=, std::true_type)
    LUISA_MAKE_VECTOR_BINARY_OPERATOR(>=, std::true_type)

#undef LUISA_MAKE_VECTOR_BINARY_AND_ASSIGNMENT_OPERATORS
#undef LUISA_MAKE_VECTOR_ASSIGNMENT_OPERATOR
#undef LUISA_MAKE_VECTOR_BINARY_OPERATOR
    
    template<typename U, std::enable_if_t<std::conjunction_v<
        std::is_same<T, std::decay_t<U>>, std::negation<std::is_same<T, bool>>>, int> = 0>
    Vector &operator*(U rhs) noexcept { return this->operator*(Vector{rhs}); }
    
    template<typename U, std::enable_if_t<std::conjunction_v<
        std::is_same<T, std::decay_t<U>>, std::negation<std::is_same<T, bool>>>, int> = 0>
    Vector &operator/(U rhs) noexcept { return this->operator/(Vector{rhs}); }
    
    template<typename U, std::enable_if_t<std::conjunction_v<
        std::is_same<T, std::decay_t<U>>, std::negation<std::is_same<T, bool>>>, int> = 0>
    Vector &operator*=(U rhs) noexcept { return this->operator*=(Vector{rhs}); }
    
    template<typename U, std::enable_if_t<std::conjunction_v<
        std::is_same<T, std::decay_t<U>>, std::negation<std::is_same<T, bool>>>, int> = 0>
    Vector &operator/=(U rhs) noexcept { return this->operator/=(Vector{rhs}); }
};

#define LUISA_MAKE_VECTOR_TYPES(T)  \
using T##2 = Vector<T, 2>;          \
using T##3 = Vector<T, 3>;          \
using T##4 = Vector<T, 4>;

LUISA_MAKE_VECTOR_TYPES(bool)
LUISA_MAKE_VECTOR_TYPES(float)
LUISA_MAKE_VECTOR_TYPES(char)
LUISA_MAKE_VECTOR_TYPES(uchar)
LUISA_MAKE_VECTOR_TYPES(short)
LUISA_MAKE_VECTOR_TYPES(ushort)
LUISA_MAKE_VECTOR_TYPES(int)
LUISA_MAKE_VECTOR_TYPES(uint)

#undef LUISA_MAKE_VECTOR_TYPES

template<size_t N>
struct Matrix {
    static_assert(always_false<std::integral_constant<size_t, N>>, "Invalid matrix type");
};

template<>
struct Matrix<3> {};

template<>
struct Matrix<4> {};

using float3x3 = Matrix<3>;
using float4x4 = Matrix<4>;

}

template<typename T, size_t N, std::enable_if_t<std::negation_v<std::is_same<T, bool>>, int> = 0>
[[nodiscard]] constexpr auto operator*(T lhs, luisa::Vector<T, N> rhs) noexcept {
    return luisa::Vector<T, N>{lhs} * rhs;
}

template<typename T, size_t N, std::enable_if_t<std::negation_v<std::is_same<T, bool>>, int> = 0>
[[nodiscard]] constexpr auto operator+(luisa::Vector<T, N> v) noexcept { return v; }

template<typename T, size_t N, std::enable_if_t<std::negation_v<std::is_same<T, bool>>, int> = 0>
[[nodiscard]] constexpr auto operator-(luisa::Vector<T, N> v) noexcept {
    using R = luisa::Vector<T, N>;
    if constexpr (N == 2) { return R{-v.x, -v.y}; }
    else if constexpr (N == 3) { return R{-v.x, -v.y, -v.z}; }
    else { return R{-v.x, -v.y, -v.z, -v.w}; }
}

[[nodiscard]] constexpr auto operator!(luisa::bool2 v) noexcept { return luisa::bool2{!v.x, !v.y}; }
[[nodiscard]] constexpr auto operator!(luisa::bool3 v) noexcept { return luisa::bool3{!v.x, !v.y, !v.z}; }
[[nodiscard]] constexpr auto operator!(luisa::bool4 v) noexcept { return luisa::bool4{!v.x, !v.y, !v.z, !v.w}; }

template<typename T, size_t N, std::enable_if_t<
    std::negation_v<std::disjunction<std::is_same<T, bool>, std::is_same<T, float>>>, int> = 0>
[[nodiscard]] constexpr auto operator~(luisa::Vector<T, N> v) noexcept {
    using R = luisa::Vector<T, N>;
    if constexpr (N == 2) { return R{~v.x, ~v.y}; }
    else if constexpr (N == 3) { return R{~v.x, ~v.y, ~v.z}; }
    else { return R{~v.x, ~v.y, ~v.z, ~v.w}; }
}
