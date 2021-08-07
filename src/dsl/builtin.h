//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#include <core/basic_types.h>
#include <dsl/expr.h>

namespace luisa::compute {

inline namespace dsl {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(Src &&s) noexcept {
    return Expr{std::forward<Src>(s)}.template cast<Dest>();
}

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(Src &&s) noexcept {
    return Expr{std::forward<Src>(s)}.template as<Dest>();
}

[[nodiscard]] inline auto thread_id() noexcept {
    return Expr<uint3>{detail::FunctionBuilder::current()->thread_id()};
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
    return Expr<uint3>{detail::FunctionBuilder::current()->block_id()};
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
    return Expr<uint3>{detail::FunctionBuilder::current()->dispatch_id()};
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
    return Expr<uint3>{detail::FunctionBuilder::current()->dispatch_size()};
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
    return std::make_tuple(Expr{v}...);
}

}// namespace dsl

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

#define LUISA_EXPR(value) \
    detail::extract_expression(std::forward<decltype(value)>(value))

template<typename Ts>
requires is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector2(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 2>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(s)})};
}

template<typename Tx, typename Ty>
requires is_scalar_expr_v<Tx> && is_scalar_expr_v<Ty> && is_same_expr_v<Tx, Ty>
[[nodiscard]] inline auto make_vector2(Tx &&x, Ty &&y) noexcept {
    using V = Vector<expr_value_t<Tx>, 2>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y)})};
}

template<typename T, typename Tv>
requires is_vector_expr_v<Tv>
[[nodiscard]] inline auto make_vector2(Tv &&v) noexcept {
    using V = Vector<T, 2>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(v)})};
}

template<typename Ts>
requires is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector3(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 3>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(s)})};
}

template<typename Tx, typename Ty, typename Tz>
requires is_scalar_expr_v<Tx> && is_scalar_expr_v<Ty> && is_scalar_expr_v<Tz> && is_same_expr_v<Tx, Ty, Tz>
[[nodiscard]] inline auto make_vector3(Tx &&x, Ty &&y, Tz &&z) noexcept {
    using V = Vector<expr_value_t<Tx>, 3>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y), LUISA_EXPR(z)})};
}

template<typename T, typename Tv>
requires is_vector3_expr_v<Tv> || is_vector4_expr_v<Tv>
[[nodiscard]] inline auto make_vector3(Tv &&v) noexcept {
    using V = Vector<T, 3>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(v)})};
}

template<typename Txy, typename Tz>
requires is_vector2_expr_v<Txy> && is_scalar_expr_v<Tz> && std::same_as<vector_expr_element_t<Txy>, expr_value_t<Tz>>
[[nodiscard]] inline auto make_vector3(Txy &&xy, Tz &&z) noexcept {
    using V = Vector<expr_value_t<Tz>, 3>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xy), LUISA_EXPR(z)})};
}

template<typename Tx, typename Tyz>
requires is_scalar_expr_v<Tx> && is_vector2_expr_v<Tyz> && std::same_as<expr_value_t<Tx>, vector_expr_element_t<Tyz>>
[[nodiscard]] inline auto make_vector3(Tx &&x, Tyz &&yz) noexcept {
    using V = Vector<expr_value_t<Tx>, 3>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(yz)})};
}

template<typename Ts>
requires is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector4(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(s)})};
}

template<typename Tx, typename Ty, typename Tz, typename Tw>
requires is_scalar_expr_v<Tx> && is_scalar_expr_v<Ty> && is_scalar_expr_v<Tz> && is_scalar_expr_v<Tw> && is_same_expr_v<Tx, Ty, Tz, Tw>
[[nodiscard]] inline auto make_vector4(Tx &&x, Ty &&y, Tz &&z, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y), LUISA_EXPR(z), LUISA_EXPR(w)})};
}

template<typename T, typename Tv>
requires is_vector4_expr_v<Tv>
[[nodiscard]] inline auto make_vector4(Tv &&v) noexcept {
    using V = Vector<T, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(v)})};
}

template<typename Txy, typename Tz, typename Tw>
requires is_vector2_expr_v<Txy> && is_scalar_expr_v<Tz> && is_scalar_expr_v<Tw> && concepts::same<vector_expr_element_t<Txy>, expr_value_t<Tz>, expr_value_t<Tw>>
[[nodiscard]] inline auto make_vector4(Txy &&xy, Tz &&z, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tz>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xy), LUISA_EXPR(z), LUISA_EXPR(w)})};
}

template<typename Tx, typename Tyz, typename Tw>
requires is_scalar_expr_v<Tx> && is_vector2_expr_v<Tyz> && is_scalar_expr_v<Tw> && concepts::same<expr_value_t<Tx>, vector_expr_element_t<Tyz>, expr_value_t<Tw>>
[[nodiscard]] inline auto make_vector4(Tx &&x, Tyz &&yz, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(yz), LUISA_EXPR(w)})};
}

template<typename Tx, typename Ty, typename Tzw>
requires is_scalar_expr_v<Tx> && is_scalar_expr_v<Ty> && is_vector2_expr_v<Tzw> && concepts::same<expr_value_t<Tx>, expr_value_t<Ty>, vector_expr_element_t<Tzw>>
[[nodiscard]] inline auto make_vector4(Tx &&x, Ty &&y, Tzw &&zw) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y), LUISA_EXPR(zw)})};
}

template<typename Txy, typename Tzw>
requires is_vector2_expr_v<Txy> && is_vector2_expr_v<Tzw> && std::same_as<vector_expr_element_t<Txy>, vector_expr_element_t<Tzw>>
[[nodiscard]] inline auto make_vector4(Txy &&xy, Tzw &&zw) noexcept {
    using V = Vector<vector_expr_element_t<Txy>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xy), LUISA_EXPR(zw)})};
}

template<typename Txyz, typename Tw>
requires is_vector3_expr_v<Txyz> && is_scalar_expr_v<Tw> && std::same_as<vector_expr_element_t<Txyz>, expr_value_t<Tw>>
[[nodiscard]] inline auto make_vector4(Txyz &&xyz, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tw>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xyz), LUISA_EXPR(w)})};
}

template<typename Tx, typename Tyzw>
requires is_scalar_expr_v<Tx> && is_vector3_expr_v<Tyzw> && std::same_as<expr_value_t<Tx>, vector_expr_element_t<Tyzw>>
[[nodiscard]] inline auto make_vector4(Tx &&x, Tyzw &&yzw) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return Expr<V>{
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(yzw)})};
}

template<template<typename> typename scalar_check, typename Tx>
requires is_scalar_expr_v<Tx> && scalar_check<expr_value_t<Tx>>::value
    [[nodiscard]] auto
    make_vector_call(CallOp op, Tx &&x) noexcept {
    using T = expr_value_t<Tx>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), op, {LUISA_EXPR(x)})};
}

template<template<typename> typename scalar_check, typename Tx>
requires is_vector_expr_v<Tx> && scalar_check<vector_expr_element_t<Tx>>::value
    [[nodiscard]] auto
    make_vector_call(CallOp op, Tx &&x) noexcept {
    using T = expr_value_t<Tx>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), op, {LUISA_EXPR(x)})};
}

template<typename Tx, typename Ty>
requires any_dsl_v<Tx, Ty> && is_same_expr_v<Tx, Ty> &&(is_scalar_expr_v<Tx> || is_vector_expr_v<Tx>)
    [[nodiscard]] auto make_vector_call(CallOp op, Tx &&x, Ty &&y) noexcept {
    using T = expr_value_t<Tx>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), op, {LUISA_EXPR(x), LUISA_EXPR(y)})};
}

template<typename Tx, typename Ty>
requires any_dsl_v<Tx, Ty> && is_scalar_expr_v<Tx> && is_vector_expr_v<Ty> && std::same_as<expr_value_t<Tx>, vector_expr_element_t<Ty>>
[[nodiscard]] auto make_vector_call(CallOp op, Tx &&x, Ty &&y) noexcept {
    using T = expr_value_t<Ty>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), op, {LUISA_EXPR(x), LUISA_EXPR(y)})};
}

template<typename Tx, typename Ty>
requires any_dsl_v<Tx, Ty> && is_vector_expr_v<Tx> && is_scalar_expr_v<Ty> && std::same_as<vector_expr_element_t<Tx>, expr_value_t<Ty>>
[[nodiscard]] auto make_vector_call(CallOp op, Tx &&x, Ty &&y) noexcept {
    using T = expr_value_t<Tx>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), op, {LUISA_EXPR(x), LUISA_EXPR(y)})};
}

}// namespace detail

inline namespace dsl {// to avoid conflicts

#define LUISA_MAKE_VECTOR(type)                                                                \
    [[nodiscard]] inline auto make_##type##2(Expr<type> s) noexcept {                          \
        return detail::make_vector2(s);                                                        \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##2(Expr<type> x, Expr<type> y) noexcept {            \
        return detail::make_vector2(x, y);                                                     \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##2(Expr<Vector<type, 3>> v) noexcept {               \
        return detail::make_vector2<type>(v);                                                  \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##2(Expr<Vector<type, 4>> v) noexcept {               \
        return detail::make_vector2<type>(v);                                                  \
    }                                                                                          \
    template<typename Tv>                                                                      \
    requires is_dsl_v<Tv> && is_vector2_expr_v<Tv>                                             \
    [[nodiscard]] inline auto make_##type##2(Tv && v) noexcept {                               \
        return detail::make_vector2<type>(std::forward<Tv>(v));                                \
    }                                                                                          \
                                                                                               \
    [[nodiscard]] inline auto make_##type##3(                                                  \
        Expr<type> x, Expr<type> y, Expr<type> z) noexcept {                                   \
        return detail::make_vector3(x, y, z);                                                  \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##3(Expr<type> s) noexcept {                          \
        return detail::make_vector3(s);                                                        \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##3(Expr<Vector<type, 2>> v, Expr<type> z) noexcept { \
        return detail::make_vector3(v, z);                                                     \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##3(Expr<type> x, Expr<Vector<type, 2>> v) noexcept { \
        return detail::make_vector3(x, v);                                                     \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##3(Expr<Vector<type, 4>> v) noexcept {               \
        return detail::make_vector3<type>(v);                                                  \
    }                                                                                          \
    template<typename Tv>                                                                      \
    requires is_dsl_v<Tv> && is_vector3_expr_v<Tv>                                             \
    [[nodiscard]] inline auto make_##type##3(Tv && v) noexcept {                               \
        return detail::make_vector3<type>(std::forward<Tv>(v));                                \
    }                                                                                          \
                                                                                               \
    [[nodiscard]] inline auto make_##type##4(Expr<type> s) noexcept {                          \
        return detail::make_vector4(s);                                                        \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<type> x, Expr<type> y, Expr<type> z, Expr<type> w) noexcept {                     \
        return detail::make_vector4(x, y, z, w);                                               \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<Vector<type, 2>> v, Expr<type> z, Expr<type> w) noexcept {                        \
        return detail::make_vector4(v, z, w);                                                  \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<type> x, Expr<Vector<type, 2>> yz, Expr<type> w) noexcept {                       \
        return detail::make_vector4(x, yz, w);                                                 \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<type> x, Expr<type> y, Expr<Vector<type, 2>> zw) noexcept {                       \
        return detail::make_vector4(x, y, zw);                                                 \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<Vector<type, 2>> xy, Expr<Vector<type, 2>> zw) noexcept {                         \
        return detail::make_vector4(xy, zw);                                                   \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<Vector<type, 3>> xyz, Expr<type> w) noexcept {                                    \
        return detail::make_vector4(xyz, w);                                                   \
    }                                                                                          \
    [[nodiscard]] inline auto make_##type##4(                                                  \
        Expr<type> x, Expr<Vector<type, 3>> yzw) noexcept {                                    \
        return detail::make_vector4(x, yzw);                                                   \
    }                                                                                          \
    template<typename Tv>                                                                      \
    requires is_dsl_v<Tv> && is_vector4_expr_v<Tv>                                             \
    [[nodiscard]] inline auto make_##type##4(Tv && v) noexcept {                               \
        return detail::make_vector4<type>(std::forward<Tv>(v));                                \
    }
LUISA_MAKE_VECTOR(bool)
LUISA_MAKE_VECTOR(int)
LUISA_MAKE_VECTOR(uint)
LUISA_MAKE_VECTOR(float)
#undef LUISA_MAKE_VECTOR

// make float2x2
[[nodiscard]] inline auto make_float2x2(Expr<float> s) noexcept {
    return Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {s.expression()})};
}

[[nodiscard]] inline auto make_float2x2(
    Expr<float> m00, Expr<float> m01,
    Expr<float> m10, Expr<float> m11) noexcept {
    return Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {m00.expression(), m01.expression(),
             m10.expression(), m11.expression()})};
}

[[nodiscard]] inline auto make_float2x2(Expr<float2> c0, Expr<float2> c1) noexcept {
    return Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {c0.expression(), c1.expression()})};
}

[[nodiscard]] inline auto make_float2x2(Expr<float2x2> m) noexcept {
    return Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {m.expression()})};
}

[[nodiscard]] inline auto make_float2x2(Expr<float3x3> m) noexcept {
    return Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {m.expression()})};
}

[[nodiscard]] inline auto make_float2x2(Expr<float4x4> m) noexcept {
    return Expr<float2x2>{
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2, {m.expression()})};
}

// make float3x3
[[nodiscard]] inline auto make_float3x3(Expr<float> s) noexcept {
    return Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {s.expression()})};
}

[[nodiscard]] inline auto make_float3x3(Expr<float3> c0, Expr<float3> c1, Expr<float3> c2) noexcept {
    return Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {c0.expression(), c1.expression(), c2.expression()})};
}

[[nodiscard]] inline auto make_float3x3(
    Expr<float> m00, Expr<float> m01, Expr<float> m02,
    Expr<float> m10, Expr<float> m11, Expr<float> m12,
    Expr<float> m20, Expr<float> m21, Expr<float> m22) noexcept {
    return Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {m00.expression(), m01.expression(), m02.expression(),
             m10.expression(), m11.expression(), m12.expression(),
             m20.expression(), m21.expression(), m22.expression()})};
}

[[nodiscard]] inline auto make_float3x3(Expr<float2x2> m) noexcept {
    return Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {m.expression()})};
}

[[nodiscard]] inline auto make_float3x3(Expr<float3x3> m) noexcept {
    return Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {m.expression()})};
}

[[nodiscard]] inline auto make_float3x3(Expr<float4x4> m) noexcept {
    return Expr<float3x3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3, {m.expression()})};
}

// make float4x4
[[nodiscard]] inline auto make_float4x4(Expr<float> s) noexcept {
    return Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {s.expression()})};
}

[[nodiscard]] inline auto make_float4x4(
    Expr<float4> c0,
    Expr<float4> c1,
    Expr<float4> c2,
    Expr<float4> c3) noexcept {
    return Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {c0.expression(), c1.expression(), c2.expression(), c3.expression()})};
}

[[nodiscard]] inline auto make_float4x4(
    Expr<float> m00, Expr<float> m01, Expr<float> m02, Expr<float> m03,
    Expr<float> m10, Expr<float> m11, Expr<float> m12, Expr<float> m13,
    Expr<float> m20, Expr<float> m21, Expr<float> m22, Expr<float> m23,
    Expr<float> m30, Expr<float> m31, Expr<float> m32, Expr<float> m33) noexcept {
    return Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {m00.expression(), m01.expression(), m02.expression(), m03.expression(),
             m10.expression(), m11.expression(), m12.expression(), m13.expression(),
             m20.expression(), m21.expression(), m22.expression(), m23.expression(),
             m30.expression(), m31.expression(), m32.expression(), m33.expression()})};
}

[[nodiscard]] inline auto make_float4x4(Expr<float2x2> m) noexcept {
    return Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {m.expression()})};
}

[[nodiscard]] inline auto make_float4x4(Expr<float3x3> m) noexcept {
    return Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {m.expression()})};
}

[[nodiscard]] inline auto make_float4x4(Expr<float4x4> m) noexcept {
    return Expr<float4x4>{
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4, {m.expression()})};
}

template<typename Tx>
requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto all(Tx &&x) noexcept {
    return Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ALL, {LUISA_EXPR(x)})};
}

template<typename Tx>
requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto any(Tx &&x) noexcept {
    return Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ANY, {LUISA_EXPR(x)})};
}

template<typename Tx>
requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto none(Tx &&x) noexcept {
    return Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::NONE, {LUISA_EXPR(x)})};
}

template<typename Tf, typename Tt>
requires is_same_expr_v<Tf, Tt>
[[nodiscard]] inline auto select(Tf &&f, Tt &&t, Expr<bool> pred) noexcept {
    using T = expr_value_t<Tf>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SELECT,
            {LUISA_EXPR(f), LUISA_EXPR(t), pred.expression()})};
}

template<typename Tf, typename Tt, typename Tp>
requires any_dsl_v<Tf, Tt, Tp> && is_same_expr_v<Tf, Tt> && is_vector_expr_v<Tf> && is_bool_vector_expr_v<Tp> && is_vector_expr_same_dimension_v<Tf, Tt, Tp>
[[nodiscard]] inline auto select(Tf &&f, Tt &&t, Tp &&p) noexcept {
    using V = Vector<vector_expr_element_t<Tf>, vector_expr_dimension_v<Tf>>;
    return Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), CallOp::SELECT,
            {LUISA_EXPR(f), LUISA_EXPR(t), LUISA_EXPR(p)})};
}

template<typename Tf, typename Tt, typename Tp>
requires any_dsl_v<Tf, Tt, Tp> && is_same_expr_v<Tf, Tt> && is_scalar_expr_v<Tf> && is_bool_vector_expr_v<Tp>
[[nodiscard]] inline auto select(Tf &&f, Tt &&t, Tp &&p) noexcept {
    using V = Vector<expr_value_t<Tf>, vector_expr_dimension_v<Tp>>;
    return Expr<V>{
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), CallOp::SELECT,
            {LUISA_EXPR(f), LUISA_EXPR(t), LUISA_EXPR(p)})};
}

template<typename Tp, typename Tt, typename Tf>
requires any_dsl_v<Tp, Tt, Tf>
[[nodiscard]] inline auto ite(Tp &&p, Tt &&t, Tf &&f) noexcept {
    return select(std::forward<Tf>(f),
                  std::forward<Tt>(t),
                  std::forward<Tp>(p));
}

template<typename Tv, typename Tl, typename Tu>
requires any_dsl_v<Tv, Tl, Tu> && is_same_expr_v<Tv, Tl, Tu> && is_scalar_expr_v<Tv>
[[nodiscard]] inline auto clamp(Tv &&v, Tl &&l, Tu &&u) noexcept {
    using T = expr_value_t<Tv>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CLAMP,
            {LUISA_EXPR(v), LUISA_EXPR(l), LUISA_EXPR(u)})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto clamp(Expr<Vector<T, N>> value, Expr<T> lb, Expr<T> ub) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto clamp(Expr<T> value, Expr<Vector<T, N>> lb, Expr<Vector<T, N>> ub) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<concepts::vector T>
[[nodiscard]] inline auto clamp(Expr<T> value, Expr<T> lb, Expr<T> ub) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::CLAMP,
            {value.expression(), lb.expression(), ub.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>,
                              is_dsl<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto clamp(X &&x, Y &&y, Z &&z) noexcept {
    return clamp(Expr{std::forward<X>(x)},
                 Expr{std::forward<Y>(y)},
                 Expr{std::forward<Z>(z)});
}

[[nodiscard]] inline auto lerp(Expr<float> left, Expr<float> right, Expr<float> t) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto lerp(Expr<Vector<float, N>> left, Expr<Vector<float, N>> right, Expr<float> t) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto lerp(Expr<float> left, Expr<float> right, Expr<Vector<float, N>> t) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto lerp(Expr<Vector<float, N>> left, Expr<Vector<float, N>> right, Expr<Vector<float, N>> t) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::LERP,
            {left.expression(), right.expression(), t.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>,
                              is_dsl<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto lerp(X &&x, Y &&y, Z &&z) noexcept {
    return lerp(Expr{std::forward<X>(x)},
                Expr{std::forward<Y>(y)},
                Expr{std::forward<Z>(z)});
}

template<typename Tv>
requires is_dsl_v<Tv>
[[nodiscard]] inline auto saturate(Tv &&v) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::SATURATE, std::forward<Tv>(v));
}

template<typename Tv>
requires is_dsl_v<Tv>
[[nodiscard]] inline auto sign(Tv &&v) noexcept {
    return detail::make_vector_call<is_signed>(CallOp::SIGN, std::forward<Tv>(v));
}

[[nodiscard]] inline auto step(Expr<float> edge, Expr<float> x) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto step(Expr<float> edge, Expr<Vector<float, N>> x) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto step(Expr<Vector<float, N>> edge, Expr<Vector<float, N>> x) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto step(Expr<Vector<float, N>> edge, Expr<float> x) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::STEP,
            {edge.expression(), x.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto step(X &&x, Y &&y) noexcept {
    return step(Expr{std::forward<X>(x)},
                Expr{std::forward<Y>(y)});
}

[[nodiscard]] inline auto smoothstep(Expr<float> left, Expr<float> right, Expr<float> t) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::SMOOTHSTEP, {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto smoothstep(Expr<float> left, Expr<float> right, Expr<Vector<float, N>> t) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SMOOTHSTEP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto smoothstep(Expr<Vector<float, N>> left, Expr<Vector<float, N>> right, Expr<Vector<float, N>> t) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SMOOTHSTEP,
            {left.expression(), right.expression(), t.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto smoothstep(Expr<Vector<float, N>> left, Expr<Vector<float, N>> right, Expr<float> t) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::SMOOTHSTEP,
            {left.expression(), right.expression(), t.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>,
                              is_dsl<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto smoothstep(X &&x, Y &&y, Z &&z) noexcept {
    return smoothstep(Expr{std::forward<X>(x)},
                      Expr{std::forward<Y>(y)},
                      Expr{std::forward<Z>(z)});
}

template<typename Tx>
requires is_dsl_v<Tx>
[[nodiscard]] inline auto abs(Tx &&x) noexcept {
    return detail::make_vector_call<is_signed>(CallOp::ABS, std::forward<Tx>(x));
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto mod(Expr<Vector<T, N>> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto mod(Expr<Vector<T, N>> x, Expr<T> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto mod(Expr<T> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto mod(Expr<T> x, Expr<T> y) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MOD, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto mod(X &&x, Y &&y) noexcept {
    return mod(Expr{std::forward<X>(x)},
               Expr{std::forward<Y>(y)});
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto fmod(Expr<Vector<T, N>> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto fmod(Expr<Vector<T, N>> x, Expr<T> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto fmod(Expr<T> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto fmod(Expr<T> x, Expr<T> y) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FMOD, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto fmod(X &&x, Y &&y) noexcept {
    return fmod(Expr{std::forward<X>(x)},
                Expr{std::forward<Y>(y)});
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto min(Expr<Vector<T, N>> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto min(Expr<Vector<T, N>> x, Expr<T> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto min(Expr<T> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto min(Expr<T> x, Expr<T> y) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MIN, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto min(X &&x, Y &&y) noexcept {
    return min(Expr{std::forward<X>(x)},
               Expr{std::forward<Y>(y)});
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto max(Expr<Vector<T, N>> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto max(Expr<Vector<T, N>> x, Expr<T> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<concepts::scalar T, size_t N>
[[nodiscard]] inline auto max(Expr<T> x, Expr<Vector<T, N>> y) noexcept {
    return Expr<Vector<T, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<T, N>>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<concepts::scalar T>
[[nodiscard]] inline auto max(Expr<T> x, Expr<T> y) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::MAX, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto max(X &&x, Y &&y) noexcept {
    return max(Expr{std::forward<X>(x)},
               Expr{std::forward<Y>(y)});
}

[[nodiscard]] inline auto clz(Expr<uint> x) noexcept {
    return Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::CLZ,
            {x.expression()})};
}

[[nodiscard]] inline auto ctz(Expr<uint> x) noexcept {
    return Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::CTZ,
            {x.expression()})};
}

[[nodiscard]] inline auto popcount(Expr<uint> x) noexcept {
    return Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::POPCOUNT,
            {x.expression()})};
}

[[nodiscard]] inline auto reverse(Expr<uint> x) noexcept {
    return Expr<uint>{
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::REVERSE,
            {x.expression()})};
}

[[nodiscard]] inline auto isinf(Expr<float> x) noexcept {
    return Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ISINF,
            {x.expression()})};
}

[[nodiscard]] inline auto isnan(Expr<float> x) noexcept {
    return Expr<bool>{
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ISNAN, {x.expression()})};
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto acos(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ACOS, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto acosh(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ACOSH, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto asin(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ASIN, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto asinh(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ASINH, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto atan(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ATAN, std::forward<T>(x));
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto atan2(Expr<T> y, Expr<T> x) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::ATAN2, {y.expression(), x.expression()})};
}

template<typename Y, typename X,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<Y>>,
                              is_dsl<std::remove_cvref_t<X>>>,
                          int> = 0>
[[nodiscard]] inline auto atan2(Y &&y, X &&x) noexcept {
    return atan2(Expr{std::forward<Y>(y)},
                 Expr{std::forward<X>(x)});
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto atanh(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ATANH, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto cos(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::COS, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto cosh(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::COSH, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto sin(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::SIN, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto sinh(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::SINH, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto tan(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::TAN, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto tanh(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::TANH, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto exp(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::EXP, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto exp2(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::EXP2, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto exp10(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::EXP10, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto log(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::LOG, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto log2(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::LOG2, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto log10(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::LOG10, std::forward<T>(x));
}

[[nodiscard]] inline auto pow(Expr<float> x, Expr<float> a) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::POW,
            {x.expression(), a.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto pow(Expr<Vector<float, N>> x, Expr<float> a) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::POW,
            {x.expression(), a.expression()})};
}

template<size_t N>
[[nodiscard]] inline auto pow(Expr<Vector<float, N>> x, Expr<Vector<float, N>> a) noexcept {
    return Expr<Vector<float, N>>{
        detail::FunctionBuilder::current()->call(
            Type::of<Vector<float, N>>(), CallOp::POW,
            {x.expression(), a.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto pow(X &&x, Y &&y) noexcept {
    return pow(Expr{std::forward<X>(x)},
               Expr{std::forward<Y>(y)});
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto sqrt(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::SQRT, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto rsqrt(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::RSQRT, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto ceil(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::CEIL, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto floor(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::FLOOR, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto fract(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::FRACT, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto trunc(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::TRUNC, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto round(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::ROUND, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto degrees(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::DEGREES, std::forward<T>(x));
}

template<typename T>
requires is_dsl_v<T>
[[nodiscard]] inline auto radians(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::RADIANS, std::forward<T>(x));
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto fma(Expr<T> x, Expr<T> y, Expr<T> z) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::FMA, {x.expression(), y.expression(), z.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>,
                              is_dsl<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto fma(X &&x, Y &&y, Z &&z) noexcept {
    return fma(Expr{std::forward<X>(x)},
               Expr{std::forward<Y>(y)},
               Expr{std::forward<Z>(z)});
}

template<typename T>
requires std::same_as<T, float> || std::same_as<T, float2> || std::same_as<T, float3> || std::same_as<T, float4>
[[nodiscard]] inline auto copysign(Expr<T> x, Expr<T> y) noexcept {
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::COPYSIGN, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto copysign(X &&x, Y &&y) noexcept {
    return copysign(Expr{std::forward<X>(x)},
                    Expr{std::forward<Y>(y)});
}

[[nodiscard]] inline auto cross(Expr<float3> x, Expr<float3> y) noexcept {
    return Expr<float3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::CROSS, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto cross(X &&x, Y &&y) noexcept {
    return cross(Expr{std::forward<X>(x)},
                 Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto dot(Expr<Vector<float, N>> x, Expr<Vector<float, N>> y) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DOT, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto dot(X &&x, Y &&y) noexcept {
    return dot(Expr{std::forward<X>(x)},
               Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto distance(Expr<Vector<float, N>> x, Expr<Vector<float, N>> y) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DISTANCE, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto distance(X &&x, Y &&y) noexcept {
    return distance(Expr{std::forward<X>(x)},
                    Expr{std::forward<Y>(y)});
}

template<size_t N>
[[nodiscard]] inline auto distance_squared(Expr<Vector<float, N>> x, Expr<Vector<float, N>> y) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DISTANCE_SQUARED, {x.expression(), y.expression()})};
}

template<typename X, typename Y,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>>,
                          int> = 0>
[[nodiscard]] inline auto distance_squared(X &&x, Y &&y) noexcept {
    return distance_squared(Expr{std::forward<X>(x)},
                            Expr{std::forward<Y>(y)});
}

template<typename Tx>
requires is_dsl_v<Tx> && is_float_vector_expr_v<Tx>
[[nodiscard]] inline auto length(Tx &&x) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH, {LUISA_EXPR(x)})};
}

template<typename Tx>
requires is_dsl_v<Tx> && is_float_vector_expr_v<Tx>
[[nodiscard]] inline auto length_squared(Tx &&x) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH_SQUARED, {LUISA_EXPR(x)})};
}

template<typename T>
requires is_dsl_v<T> && is_vector_expr_v<T>
[[nodiscard]] inline auto normalize(T &&x) noexcept {
    return detail::make_vector_call<is_floating_point>(CallOp::NORMALIZE, std::forward<T>(x));
}

[[nodiscard]] inline auto faceforward(Expr<float3> n, Expr<float3> i, Expr<float3> n_ref) noexcept {
    return Expr<float3>{
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::FACEFORWARD,
            {n.expression(), i.expression(), n_ref.expression()})};
}

template<typename X, typename Y, typename Z,
         std::enable_if_t<std::disjunction_v<
                              is_dsl<std::remove_cvref_t<X>>,
                              is_dsl<std::remove_cvref_t<Y>>,
                              is_dsl<std::remove_cvref_t<Z>>>,
                          int> = 0>
[[nodiscard]] inline auto faceforward(X &&n, Y &&i, Z &&n_ref) noexcept {
    return faceforward(Expr{std::forward<X>(n)},
                       Expr{std::forward<Y>(i)},
                       Expr{std::forward<Z>(n_ref)});
}

template<typename Tm>
requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto determinant(Tm &&m) noexcept {
    return Expr<float>{
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DETERMINANT, {LUISA_EXPR(m)})};
}

template<typename Tm>
requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto transpose(Tm &&m) noexcept {
    using T = expr_value_t<Tm>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRANSPOSE, {LUISA_EXPR(m)})};
}

template<typename Tm>
requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto inverse(Tm &&m) noexcept {
    using T = expr_value_t<Tm>;
    return Expr<T>{
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::INVERSE, {LUISA_EXPR(m)})};
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

}// namespace dsl

}// namespace luisa::compute
