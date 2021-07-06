//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#include <core/basic_types.h>
#include <dsl/expr.h>

namespace luisa::compute {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(detail::Expr<Src> s) noexcept { return s.template cast<Dest>(); }

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(detail::Expr<Src> s) noexcept { return s.template as<Dest>(); }

[[nodiscard]] inline auto thread_id() noexcept {
    return detail::Expr<uint3>{detail::FunctionBuilder::current()->thread_id()};
}

[[nodiscard]] inline auto thread_x() noexcept {
    return thread_id().x;
}

[[nodiscard]] inline auto thread_y() noexcept {
    return thread_id().y;
}

[[nodiscard]] inline auto thread_z() noexcept {
    return thread_id().z;
}

[[nodiscard]] inline auto block_id() noexcept {
    return detail::Expr<uint3>{detail::FunctionBuilder::current()->block_id()};
}

[[nodiscard]] inline auto block_x() noexcept {
    return block_id().x;
}

[[nodiscard]] inline auto block_y() noexcept {
    return block_id().y;
}

[[nodiscard]] inline auto block_z() noexcept {
    return block_id().z;
}

[[nodiscard]] inline auto dispatch_id() noexcept {
    return detail::Expr<uint3>{detail::FunctionBuilder::current()->dispatch_id()};
}

[[nodiscard]] inline auto dispatch_x() noexcept {
    return dispatch_id().x;
}

[[nodiscard]] inline auto dispatch_y() noexcept {
    return dispatch_id().y;
}

[[nodiscard]] inline auto dispatch_z() noexcept {
    return dispatch_id().z;
}

[[nodiscard]] inline auto dispatch_size() noexcept {
    return detail::Expr<uint3>{detail::FunctionBuilder::current()->dispatch_size()};
}

[[nodiscard]] inline auto dispatch_size_x() noexcept {
    return dispatch_size().x;
}

[[nodiscard]] inline auto dispatch_size_y() noexcept {
    return dispatch_size().y;
}

[[nodiscard]] inline auto dispatch_size_z() noexcept {
    return dispatch_size().z;
}

[[nodiscard]] inline auto block_size() noexcept {
    return detail::FunctionBuilder::current()->block_size();
}

[[nodiscard]] inline auto block_size_x() noexcept {
    return block_size().x;
}

[[nodiscard]] inline auto block_size_y() noexcept {
    return block_size().y;
}

[[nodiscard]] inline auto block_size_z() noexcept {
    return block_size().z;
}

inline void set_block_size(uint x, uint y = 1u, uint z = 1u) noexcept {
    detail::FunctionBuilder::current()->set_block_size(
        uint3{std::max(x, 1u), std::max(y, 1u), std::max(z, 1u)});
}

template<typename... T>
[[nodiscard]] inline auto multiple(T &&...v) noexcept {
    return std::make_tuple(detail::Expr{v}...);
}

namespace detail {

template<typename T>
[[nodiscard]] constexpr auto make_vector_tag() noexcept {
    if constexpr (std::is_same_v<T, bool2>) {
        return CallOp::MAKE_BOOL2;
    } else if constexpr (std::is_same_v<T, bool3>) {
        return CallOp::MAKE_BOOL3;
    } else if constexpr (std::is_same_v<T, bool4>) {
        return CallOp::MAKE_BOOL4;
    } else if constexpr (std::is_same_v<T, int2>) {
        return CallOp::MAKE_INT2;
    } else if constexpr (std::is_same_v<T, int3>) {
        return CallOp::MAKE_INT3;
    } else if constexpr (std::is_same_v<T, int4>) {
        return CallOp::MAKE_INT4;
    } else if constexpr (std::is_same_v<T, uint2>) {
        return CallOp::MAKE_UINT2;
    } else if constexpr (std::is_same_v<T, uint3>) {
        return CallOp::MAKE_UINT3;
    } else if constexpr (std::is_same_v<T, uint4>) {
        return CallOp::MAKE_UINT4;
    } else if constexpr (std::is_same_v<T, float2>) {
        return CallOp::MAKE_FLOAT2;
    } else if constexpr (std::is_same_v<T, float3>) {
        return CallOp::MAKE_FLOAT3;
    } else if constexpr (std::is_same_v<T, float4>) {
        return CallOp::MAKE_FLOAT4;
    } else {
        static_assert(always_false_v<T>);
    }
}

}// namespace detail

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector2(detail::Expr<T> s) noexcept {
    using V = Vector<T, 2>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {s.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector2(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    using V = Vector<T, 2>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression()})};
}

template<concepts::vector T>
[[nodiscard]] inline auto make_vector2(detail::Expr<T> v) noexcept {
    using V = Vector<typename T::value_type, 2>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {v.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector3(detail::Expr<T> s) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {s.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector3(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), z.expression()})};
}

template<concepts::scalar T, size_t N>
requires(N == 3) || (N == 4) [[nodiscard]] inline auto make_vector3(detail::Expr<Vector<T, N>> v) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {v.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector3(detail::Expr<Vector<T, 2>> xy, detail::Expr<T> z) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {xy.expression(), z.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector3(detail::Expr<T> x, detail::Expr<Vector<T, 2>> yz) noexcept {
    using V = Vector<T, 3>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), yz.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<T> s) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {s.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z, detail::Expr<T> w) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), z.expression(), w.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<Vector<T, 4>> v) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {v.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<Vector<T, 2>> xy, detail::Expr<T> z, detail::Expr<T> w) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {xy.expression(), z.expression(), w.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<T> x, detail::Expr<Vector<T, 2>> yz, detail::Expr<T> w) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), yz.expression(), w.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<Vector<T, 2>> zw) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), y.expression(), zw.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<Vector<T, 2>> xy, detail::Expr<Vector<T, 2>> zw) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {xy.expression(), zw.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<Vector<T, 3>> xyz, detail::Expr<T> w) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {xyz.expression(), w.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto make_vector4(detail::Expr<T> x, detail::Expr<Vector<T, 3>> yzw) noexcept {
    using V = Vector<T, 4>;
    return detail::Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), detail::make_vector_tag<V>(),
            {x.expression(), yzw.expression()})};
}

#define LUISA_MAKE_VECTOR(type)                                                                                \
    [[nodiscard]] inline auto make_##type##2(type s) noexcept {                                                \
        return make_vector2(detail::Expr{s});                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<type> s) noexcept {                                  \
        return make_vector2(s);                                                                                \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<type> x, detail::Expr<type> y) noexcept {            \
        return make_vector2(x, y);                                                                             \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<type, 3>> v) noexcept {                       \
        return make_vector2(v);                                                                                \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<type, 4>> v) noexcept {                       \
        return make_vector2(v);                                                                                \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##2(detail::Expr<Vector<T, 2>> v) noexcept {                          \
        if constexpr (std::is_same_v<T, type>) {                                                               \
            return make_vector2(v);                                                                            \
        } else {                                                                                               \
            return cast<type##2>(make_vector2(v));                                                             \
        }                                                                                                      \
    }                                                                                                          \
                                                                                                               \
    [[nodiscard]] inline auto make_##type##3(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<type> z) noexcept {                           \
        return make_vector3(x, y, z);                                                                          \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(type s) noexcept {                                                \
        return make_vector3(detail::Expr{s});                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<type> s) noexcept {                                  \
        return make_vector3(s);                                                                                \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<type, 2>> v, detail::Expr<type> z) noexcept { \
        return make_vector3(v, z);                                                                             \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<type> x, detail::Expr<Vector<type, 2>> v) noexcept { \
        return make_vector3(x, v);                                                                             \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<type, 4>> v) noexcept {                       \
        return make_vector3(v);                                                                                \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##3(detail::Expr<Vector<T, 3>> v) noexcept {                          \
        if constexpr (std::is_same_v<T, type>) {                                                               \
            return make_vector3(v);                                                                            \
        } else {                                                                                               \
            return cast<type##3>(make_vector3(v));                                                             \
        }                                                                                                      \
    }                                                                                                          \
                                                                                                               \
    [[nodiscard]] inline auto make_##type##4(type s) noexcept {                                                \
        return make_vector4(detail::Expr{s});                                                                  \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(detail::Expr<type> s) noexcept {                                  \
        return make_vector4(s);                                                                                \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<type> z, detail::Expr<type> w) noexcept {     \
        return make_vector4(x, y, z, w);                                                                       \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 2>> v, detail::Expr<type> z, detail::Expr<type> w) noexcept {                \
        return make_vector4(v, z, w);                                                                          \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<Vector<type, 2>> yz, detail::Expr<type> w) noexcept {               \
        return make_vector4(x, yz, w);                                                                         \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<type> y, detail::Expr<Vector<type, 2>> zw) noexcept {               \
        return make_vector4(x, y, zw);                                                                         \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 2>> xy, detail::Expr<Vector<type, 2>> zw) noexcept {                         \
        return make_vector4(xy, zw);                                                                           \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<Vector<type, 3>> xyz, detail::Expr<type> w) noexcept {                                    \
        return make_vector4(xyz, w);                                                                           \
    }                                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                                  \
        detail::Expr<type> x, detail::Expr<Vector<type, 3>> yzw) noexcept {                                    \
        return make_vector4(x, yzw);                                                                           \
    }                                                                                                          \
    template<typename T>                                                                                       \
    [[nodiscard]] inline auto make_##type##4(detail::Expr<Vector<T, 4>> v) noexcept {                          \
        if constexpr (std::is_same_v<T, type>) {                                                               \
            return make_vector4(v);                                                                            \
        } else {                                                                                               \
            return cast<type##4>(make_vector4(v));                                                             \
        }                                                                                                      \
    }
LUISA_MAKE_VECTOR(bool)
LUISA_MAKE_VECTOR(int)
LUISA_MAKE_VECTOR(uint)
LUISA_MAKE_VECTOR(float)
#undef LUISA_MAKE_VECTOR

// make float2x2
[[nodiscard]] inline auto make_float2x2(detail::Expr<float> s) noexcept {
    return detail::Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {s.expression()})};
}

[[nodiscard]] inline auto make_float2x2(
    detail::Expr<float> m00, detail::Expr<float> m01,
    detail::Expr<float> m10, detail::Expr<float> m11) noexcept {
    return detail::Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {m00.expression(), m01.expression(),
             m10.expression(), m11.expression()})};
}

[[nodiscard]] inline auto make_float2x2(detail::Expr<float2> c0, detail::Expr<float2> c1) noexcept {
    return detail::Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {c0.expression(), c1.expression()})};
}

[[nodiscard]] inline auto make_float2x2(detail::Expr<float2x2> m) noexcept {
    return detail::Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {m.expression()})};
}

[[nodiscard]] inline auto make_float2x2(detail::Expr<float3x3> m) noexcept {
    return detail::Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {m.expression()})};
}

[[nodiscard]] inline auto make_float2x2(detail::Expr<float4x4> m) noexcept {
    return detail::Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {m.expression()})};
}

// make float3x3
[[nodiscard]] inline auto make_float3x3(detail::Expr<float> s) noexcept {
    return detail::Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {s.expression()})};
}

[[nodiscard]] inline auto make_float3x3(detail::Expr<float3> c0, detail::Expr<float3> c1, detail::Expr<float3> c2) noexcept {
    return detail::Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {c0.expression(), c1.expression(), c2.expression()})};
}

[[nodiscard]] inline auto make_float3x3(
    detail::Expr<float> m00, detail::Expr<float> m01, detail::Expr<float> m02,
    detail::Expr<float> m10, detail::Expr<float> m11, detail::Expr<float> m12,
    detail::Expr<float> m20, detail::Expr<float> m21, detail::Expr<float> m22) noexcept {
    return detail::Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {m00.expression(), m01.expression(), m02.expression(),
             m10.expression(), m11.expression(), m12.expression(),
             m20.expression(), m21.expression(), m22.expression()})};
}

[[nodiscard]] inline auto make_float3x3(detail::Expr<float2x2> m) noexcept {
    return detail::Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {m.expression()})};
}

[[nodiscard]] inline auto make_float3x3(detail::Expr<float3x3> m) noexcept {
    return detail::Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {m.expression()})};
}

[[nodiscard]] inline auto make_float3x3(detail::Expr<float4x4> m) noexcept {
    return detail::Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {m.expression()})};
}

// make float4x4
[[nodiscard]] inline auto make_float4x4(detail::Expr<float> s) noexcept {
    return detail::Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {s.expression()})};
}

[[nodiscard]] inline auto make_float4x4(
    detail::Expr<float4> c0,
    detail::Expr<float4> c1,
    detail::Expr<float4> c2,
    detail::Expr<float4> c3) noexcept {
    return detail::Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {c0.expression(), c1.expression(), c2.expression(), c3.expression()})};
}

[[nodiscard]] inline auto make_float4x4(
    detail::Expr<float> m00, detail::Expr<float> m01, detail::Expr<float> m02, detail::Expr<float> m03,
    detail::Expr<float> m10, detail::Expr<float> m11, detail::Expr<float> m12, detail::Expr<float> m13,
    detail::Expr<float> m20, detail::Expr<float> m21, detail::Expr<float> m22, detail::Expr<float> m23,
    detail::Expr<float> m30, detail::Expr<float> m31, detail::Expr<float> m32, detail::Expr<float> m33) noexcept {
    return detail::Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {m00.expression(), m01.expression(), m02.expression(), m03.expression(),
             m10.expression(), m11.expression(), m12.expression(), m13.expression(),
             m20.expression(), m21.expression(), m22.expression(), m23.expression(),
             m30.expression(), m31.expression(), m32.expression(), m33.expression()})};
}

[[nodiscard]] inline auto make_float4x4(detail::Expr<float2x2> m) noexcept {
    return detail::Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {m.expression()})};
}

[[nodiscard]] inline auto make_float4x4(detail::Expr<float3x3> m) noexcept {
    return detail::Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {m.expression()})};
}

[[nodiscard]] inline auto make_float4x4(detail::Expr<float4x4> m) noexcept {
    return detail::Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {m.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto all(detail::Expr<Vector<bool, N>> x) noexcept {
    return detail::Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ALL, {x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto any(detail::Expr<Vector<bool, N>> x) noexcept {
    return detail::Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ANY, {x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto none(detail::Expr<Vector<bool, N>> x) noexcept {
    return detail::Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::NONE, {x.expression()})};
}

template<typename T>
[[nodiscard]] inline auto select(detail::Expr<T> false_value, detail::Expr<T> true_value, detail::Expr<bool> pred) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SELECT,
            {false_value.expression(), true_value.expression(), pred.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto select(detail::Expr<Vector<T, N>> false_value, detail::Expr<Vector<T, N>> true_value, detail::Expr<Vector<bool, N>> pred) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::SELECT,
            {false_value.expression(), true_value.expression(), pred.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto select(detail::Expr<T> false_value, detail::Expr<T> true_value, detail::Expr<Vector<bool, N>> pred) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::SELECT,
            {false_value.expression(), true_value.expression(), pred.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto select(X &&x, Y &&y, Z &&z) noexcept {
    return select(detail::Expr{std::forward<X>(x)},
                  detail::Expr{std::forward<Y>(y)},
                  detail::Expr{std::forward<Z>(z)});
}

template<typename T>
[[nodiscard]] inline auto ite(detail::Expr<bool> pred, detail::Expr<T> true_value, detail::Expr<T> false_value) noexcept {
    return select(false_value, true_value, pred);
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto ite(detail::Expr<Vector<bool, N>> pred, detail::Expr<Vector<T, N>> true_value, detail::Expr<Vector<T, N>> false_value) noexcept {
    return select(false_value, true_value, pred);
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto ite(detail::Expr<Vector<bool, N>> pred, detail::Expr<T> true_value, detail::Expr<T> false_value) noexcept {
    return select(false_value, true_value, pred);
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto ite(X &&x, Y &&y, Z &&z) noexcept {
    return ite(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)},
               detail::Expr{std::forward<Z>(z)});
}

template<concepts::scalar T>
[[nodiscard]] inline auto clamp(detail::Expr<T> value, detail::Expr<T> lb, detail::Expr<T> ub) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto clamp(detail::Expr<Vector<T, N>> value, detail::Expr<T> lb, detail::Expr<T> ub) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto clamp(detail::Expr<T> value, detail::Expr<Vector<T, N>> lb, detail::Expr<Vector<T, N>> ub) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<concepts::vector T>
[[nodiscard]] inline auto clamp(detail::Expr<T> value, detail::Expr<T> lb, detail::Expr<T> ub) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto clamp(X &&x, Y &&y, Z &&z) noexcept {
    return clamp(detail::Expr{std::forward<X>(x)},
                 detail::Expr{std::forward<Y>(y)},
                 detail::Expr{std::forward<Z>(z)});
}

[[nodiscard]] inline auto lerp(detail::Expr<float> left, detail::Expr<float> right, detail::Expr<float> t) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto lerp(detail::Expr<Vector<float, N>> left, detail::Expr<Vector<float, N>> right, detail::Expr<float> t) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto lerp(detail::Expr<float> left, detail::Expr<float> right, detail::Expr<Vector<float, N>> t) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto lerp(detail::Expr<Vector<float, N>> left, detail::Expr<Vector<float, N>> right, detail::Expr<Vector<float, N>> t) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto lerp(X &&x, Y &&y, Z &&z) noexcept {
    return lerp(detail::Expr{std::forward<X>(x)},
                detail::Expr{std::forward<Y>(y)},
                detail::Expr{std::forward<Z>(z)});
}

[[nodiscard]] inline auto saturate(detail::Expr<float> value) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::SATURATE, {value.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto saturate(detail::Expr<Vector<float, N>> value) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SATURATE, {value.expression()})};
}

[[nodiscard]] inline auto sign(detail::Expr<float> value) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::SIGN, {value.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto sign(detail::Expr<Vector<float, N>> value) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SIGN, {value.expression()})};
}

[[nodiscard]] inline auto step(detail::Expr<float> edge, detail::Expr<float> x) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto step(detail::Expr<float> edge, detail::Expr<Vector<float, N>> x) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto step(detail::Expr<Vector<float, N>> edge, detail::Expr<Vector<float, N>> x) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto step(detail::Expr<Vector<float, N>> edge, detail::Expr<float> x) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto step(X &&x, Y &&y) noexcept {
    return step(detail::Expr{std::forward<X>(x)},
                detail::Expr{std::forward<Y>(y)});
}

[[nodiscard]] inline auto smoothstep(detail::Expr<float> left, detail::Expr<float> right, detail::Expr<float> t) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::SMOOTHSTEP, {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto smoothstep(detail::Expr<float> left, detail::Expr<float> right, detail::Expr<Vector<float, N>> t) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SMOOTHSTEP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto smoothstep(detail::Expr<Vector<float, N>> left, detail::Expr<Vector<float, N>> right, detail::Expr<Vector<float, N>> t) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SMOOTHSTEP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto smoothstep(detail::Expr<Vector<float, N>> left, detail::Expr<Vector<float, N>> right, detail::Expr<float> t) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SMOOTHSTEP,
            {left.expression(), right.expression(), t.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto smoothstep(X &&x, Y &&y, Z &&z) noexcept {
    return smoothstep(detail::Expr{std::forward<X>(x)},
                      detail::Expr{std::forward<Y>(y)},
                      detail::Expr{std::forward<Z>(z)});
}

template<typename T>
requires std::same_as<T, int> || std::same_as<T, float>
[[nodiscard]] inline auto abs(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ABS, {x.expression()})};
}

template<typename T, size_t N>
requires std::same_as<T, int> || std::same_as<T, float>
[[nodiscard]] inline auto abs(detail::Expr<Vector<T, N>> x) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::ABS, {x.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto mod(detail::Expr<Vector<T, N>> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto mod(detail::Expr<Vector<T, N>> x, detail::Expr<T> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto mod(detail::Expr<T> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto mod(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto mod(X &&x, Y &&y) noexcept {
    return mod(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)});
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto fmod(detail::Expr<Vector<T, N>> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto fmod(detail::Expr<Vector<T, N>> x, detail::Expr<T> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto fmod(detail::Expr<T> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto fmod(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto fmod(X &&x, Y &&y) noexcept {
    return fmod(detail::Expr{std::forward<X>(x)},
                detail::Expr{std::forward<Y>(y)});
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto min(detail::Expr<Vector<T, N>> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto min(detail::Expr<Vector<T, N>> x, detail::Expr<T> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto min(detail::Expr<T> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto min(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto min(X &&x, Y &&y) noexcept {
    return min(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)});
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto max(detail::Expr<Vector<T, N>> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto max(detail::Expr<Vector<T, N>> x, detail::Expr<T> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto max(detail::Expr<T> x, detail::Expr<Vector<T, N>> y) noexcept {
    return detail::Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto max(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto max(X &&x, Y &&y) noexcept {
    return max(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)});
}

[[nodiscard]] inline auto clz(detail::Expr<uint> x) noexcept {
    return detail::Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::CLZ,
            {x.expression()})};
}

[[nodiscard]] inline auto ctz(detail::Expr<uint> x) noexcept {
    return detail::Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::CTZ,
            {x.expression()})};
}

[[nodiscard]] inline auto popcount(detail::Expr<uint> x) noexcept {
    return detail::Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::POPCOUNT,
            {x.expression()})};
}

[[nodiscard]] inline auto reverse(detail::Expr<uint> x) noexcept {
    return detail::Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::REVERSE,
            {x.expression()})};
}

[[nodiscard]] inline auto isinf(detail::Expr<float> x) noexcept {
    return detail::Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ISINF,
            {x.expression()})};
}

[[nodiscard]] inline auto isnan(detail::Expr<float> x) noexcept {
    return detail::Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ISNAN, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto acos(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ACOS, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto acosh(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ACOSH, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto asin(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ASIN, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto asinh(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ASINH, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto atan(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATAN, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto atan2(detail::Expr<T> y, detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATAN2, {y.expression(), x.expression()})};
}

template<typename Y, typename X,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<X>>>,
                          int> = 0>
[[nodiscard]] inline auto atan2(Y &&y, X &&x) noexcept {
    return atan2(detail::Expr{std::forward<Y>(y)},
                 detail::Expr{std::forward<X>(x)});
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto atanh(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATANH, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto cos(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COS, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto cosh(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COSH, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto sin(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SIN, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto sinh(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SINH, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto tan(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TAN, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto tanh(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TANH, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto exp(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::EXP, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto exp2(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::EXP2, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto exp10(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::EXP10, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto log(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LOG, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto log2(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LOG2, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto log10(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::LOG10, {x.expression()})};
}

[[nodiscard]] inline auto pow(detail::Expr<float> x, detail::Expr<float> a) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::POW,
            {x.expression(), a.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto pow(detail::Expr<Vector<float, N>> x, detail::Expr<float> a) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::POW,
            {x.expression(), a.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto pow(detail::Expr<Vector<float, N>> x, detail::Expr<Vector<float, N>> a) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::POW,
            {x.expression(), a.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto pow(X &&x, Y &&y) noexcept {
    return pow(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)});
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto sqrt(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SQRT, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto rsqrt(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::RSQRT, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto ceil(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CEIL, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto floor(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FLOOR, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto fract(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FRACT, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto trunc(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRUNC, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto round(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ROUND, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto degrees(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::DEGREES, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto radians(detail::Expr<T> x) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::RADIANS, {x.expression()})};
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto fma(detail::Expr<T> x, detail::Expr<T> y, detail::Expr<T> z) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FMA, {x.expression(), y.expression(), z.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto fma(X &&x, Y &&y, Z &&z) noexcept {
    return fma(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)},
               detail::Expr{std::forward<Z>(z)});
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto copysign(detail::Expr<T> x, detail::Expr<T> y) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COPYSIGN, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto copysign(X &&x, Y &&y) noexcept {
    return copysign(detail::Expr{std::forward<X>(x)},
                    detail::Expr{std::forward<Y>(y)});
}

[[nodiscard]] inline auto cross(detail::Expr<float3> x, detail::Expr<float3> y) noexcept {
    return detail::Expr<float3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::CROSS, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto cross(X &&x, Y &&y) noexcept {
    return cross(detail::Expr{std::forward<X>(x)},
                 detail::Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto dot(detail::Expr<Vector<float, N>> x, detail::Expr<Vector<float, N>> y) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DOT, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto dot(X &&x, Y &&y) noexcept {
    return dot(detail::Expr{std::forward<X>(x)},
               detail::Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto distance(detail::Expr<Vector<float, N>> x, detail::Expr<Vector<float, N>> y) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DISTANCE, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto distance(X &&x, Y &&y) noexcept {
    return distance(detail::Expr{std::forward<X>(x)},
                    detail::Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto distance_squared(detail::Expr<Vector<float, N>> x, detail::Expr<Vector<float, N>> y) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DISTANCE_SQUARED, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto distance_squared(X &&x, Y &&y) noexcept {
    return distance_squared(detail::Expr{std::forward<X>(x)},
                            detail::Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto length(detail::Expr<Vector<float, N>> u) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH, {u.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto length_squared(detail::Expr<Vector<float, N>> u) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH_SQUARED, {u.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto normalize(detail::Expr<Vector<float, N>> u) noexcept {
    return detail::Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::NORMALIZE, {u.expression()})};
}

[[nodiscard]] inline auto faceforward(detail::Expr<float3> n, detail::Expr<float3> i, detail::Expr<float3> n_ref) noexcept {
    return detail::Expr<float3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::FACEFORWARD,
            {n.expression(), i.expression(), n_ref.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              detail::is_expr<std::remove_cvref_t<X>>,
                              detail::is_expr<std::remove_cvref_t<Y>>,
                              detail::is_expr<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto faceforward(X &&n, Y &&i, Z &&n_ref) noexcept {
    return faceforward(detail::Expr{std::forward<X>(n)},
                       detail::Expr{std::forward<Y>(i)},
                       detail::Expr{std::forward<Z>(n_ref)});
}

template<concepts::matrix T>
[[nodiscard]] inline auto determinant(detail::Expr<T> mat) noexcept {
    return detail::Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DETERMINANT, {mat.expression()})};
}

template<concepts::matrix T>
[[nodiscard]] inline auto transpose(detail::Expr<T> mat) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRANSPOSE, {mat.expression()})};
}

template<concepts::matrix T>
[[nodiscard]] inline auto inverse(detail::Expr<T> mat) noexcept {
    return detail::Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::INVERSE, {mat.expression()})};
}

// memory barriers
inline void group_memory_barrier() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::GROUP_MEMORY_BARRIER, {});
}

inline void all_memory_barrier() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::ALL_MEMORY_BARRIER, {});
}

inline void device_memory_barrier() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::DEVICE_MEMORY_BARRIER, {});
}

}// namespace luisa::compute
