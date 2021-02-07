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
    constexpr VectorStorage() noexcept : x{}, y{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s} {}
    explicit constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x, y, z;
    constexpr VectorStorage() noexcept : x{}, y{}, z{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s}, z{s} {}
    explicit constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    explicit constexpr VectorStorage(VectorStorage<T, 2> xy, T z) noexcept : x{xy.x}, y{xy.y}, z{z} {}
    explicit constexpr VectorStorage(T x, VectorStorage<T, 2> yz) noexcept : x{x}, y{yz.x}, z{yz.y} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x, y, z, w;
    constexpr VectorStorage() noexcept : x{}, y{}, z{}, w{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s}, z{s}, w{s} {}
    explicit constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    explicit constexpr VectorStorage(VectorStorage<T, 2> xy, T z, T w) noexcept : x{xy.x}, y{xy.y}, z{z}, w{w} {}
    explicit constexpr VectorStorage(VectorStorage<T, 2> xy, VectorStorage<T, 2> zw) noexcept : x{xy.x}, y{xy.y}, z{zw.x}, w{zw.y} {}
    explicit constexpr VectorStorage(T x, VectorStorage<T, 2> yz, T w) noexcept : x{x}, y{yz.x}, z{yz.y}, w{w} {}
    explicit constexpr VectorStorage(T x, T y, VectorStorage<T, 2> zw) noexcept : x{x}, y{y}, z{zw.x}, w{zw.y} {}
    explicit constexpr VectorStorage(VectorStorage<T, 3> xyz, T w) noexcept : x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {}
    explicit constexpr VectorStorage(T x, VectorStorage<T, 3> yzw) noexcept : x{x}, y{yzw.x}, z{yzw.y}, w{yzw.z} {}
};

}

template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {

    using Storage = detail::VectorStorage<T, N>;

    template<typename ...Args>
    explicit constexpr Vector(Args ...args) noexcept : Storage{args...} {}

    
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
struct Matrix {};

using float3x3 = Matrix<3>;
using float4x4 = Matrix<4>;

}
