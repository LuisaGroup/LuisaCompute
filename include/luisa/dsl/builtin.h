#pragma once

#include <luisa/core/constants.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/operators.h>
#include <luisa/dsl/expr_traits.h>

namespace luisa::compute {

namespace detail {
LC_DSL_API void validate_block_size(uint x, uint y, uint z) noexcept;
}// namespace detail

inline namespace dsl {

/// Expression cast operation
template<typename Dest, typename Src>
    requires is_dsl_v<Src>
[[nodiscard]] inline auto cast(Src &&s) noexcept {
    Expr expr{std::forward<Src>(s)};
    return expr.template cast<expr_value_t<Dest>>();
}

template<typename Dest, typename Src>
    requires std::negation_v<is_dsl<Src>>
[[nodiscard]] inline auto cast(Src &&s) noexcept {
    return static_cast<Dest>(std::forward<Src>(s));
}

/// Expression as operation
template<typename Dest, typename Src>
    requires is_dsl_v<Src>
[[nodiscard]] inline auto as(Src &&s) noexcept {
    Expr expr{std::forward<Src>(s)};
    return expr.template as<Dest>();
}

template<typename Dest, typename Src>
    requires std::negation_v<is_dsl<Src>>
[[nodiscard]] inline auto as(Src &&s) noexcept {
    return luisa::bit_cast<Dest>(std::forward<Src>(s));
}

/// Call assume on bool expression
inline void assume(Expr<bool> pred) noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::ASSUME, {pred.expression()});
}

/// Call unreachable
inline void unreachable() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::UNREACHABLE, {});
}

/// Get thread_id(uint3)
[[nodiscard]] inline auto thread_id() noexcept {
    return def<uint3>(detail::FunctionBuilder::current()->thread_id());
}

/// Get thread_id.x
[[nodiscard]] inline auto thread_x() noexcept {
    return thread_id().x;
}

/// Get thread_id.y
[[nodiscard]] inline auto thread_y() noexcept {
    return thread_id().y;
}

/// Get thread_id.z
[[nodiscard]] inline auto thread_z() noexcept {
    return thread_id().z;
}

/// Get block_id(uint3)
[[nodiscard]] inline auto block_id() noexcept {
    return def<uint3>(detail::FunctionBuilder::current()->block_id());
}

/// Get block_id.x
[[nodiscard]] inline auto block_x() noexcept {
    return block_id().x;
}

/// Get block_id.y
[[nodiscard]] inline auto block_y() noexcept {
    return block_id().y;
}

/// Get block_id.z
[[nodiscard]] inline auto block_z() noexcept {
    return block_id().z;
}

/// Get dispatch_id(uint3)
[[nodiscard]] inline auto dispatch_id() noexcept {
    return def<uint3>(detail::FunctionBuilder::current()->dispatch_id());
}
[[nodiscard]] inline auto object_id() noexcept {
    return def<uint>(detail::FunctionBuilder::current()->object_id());
}
[[nodiscard]] inline auto kernel_id() noexcept {
    return def<uint>(detail::FunctionBuilder::current()->kernel_id());
}
[[nodiscard]] inline auto warp_lane_count() noexcept {
    return def<uint>(detail::FunctionBuilder::current()->warp_lane_count());
}
[[nodiscard]] inline auto warp_lane_id() noexcept {
    return def<uint>(detail::FunctionBuilder::current()->warp_lane_id());
}
/// Get dispatch_id.x
[[nodiscard]] inline auto dispatch_x() noexcept {
    return dispatch_id().x;
}

/// Get dispatch_id.y
[[nodiscard]] inline auto dispatch_y() noexcept {
    return dispatch_id().y;
}

/// Get dispatch.z
[[nodiscard]] inline auto dispatch_z() noexcept {
    return dispatch_id().z;
}

/// Get dispatch size(uint3)
[[nodiscard]] inline auto dispatch_size() noexcept {
    return def<uint3>(detail::FunctionBuilder::current()->dispatch_size());
}

/// Get dispatch_size.x
[[nodiscard]] inline auto dispatch_size_x() noexcept {
    return dispatch_size().x;
}

/// Get dispatch_size.y
[[nodiscard]] inline auto dispatch_size_y() noexcept {
    return dispatch_size().y;
}

/// Get dispatch_size.z
[[nodiscard]] inline auto dispatch_size_z() noexcept {
    return dispatch_size().z;
}

/// Get block size(uint3)
[[nodiscard]] inline auto block_size() noexcept {
    return detail::FunctionBuilder::current()->block_size();
}

/// Get block_size.x
[[nodiscard]] inline auto block_size_x() noexcept {
    return block_size().x;
}

/// Get block_size.y
[[nodiscard]] inline auto block_size_y() noexcept {
    return block_size().y;
}

/// Get block_size.z
[[nodiscard]] inline auto block_size_z() noexcept {
    return block_size().z;
}

/// Set current function block size as (x, y, z)
inline void set_block_size(uint x, uint y = 1u, uint z = 1u) noexcept {
    detail::validate_block_size(x, y, z);
    detail::FunctionBuilder::current()->set_block_size(
        uint3{std::max(x, 1u), std::max(y, 1u), std::max(z, 1u)});
}

inline void set_block_size(uint3 size) noexcept {
    set_block_size(size.x, size.y, size.z);
}

inline void set_block_size(uint2 size) noexcept {
    set_block_size(size.x, size.y, 1u);
}

}// namespace dsl

inline namespace dsl {

/// Define a var of type T and construct params args
template<typename T, typename... Args>
    requires std::negation_v<std::disjunction<std::is_pointer<std::remove_cvref_t<Args>>...>>
[[nodiscard]] auto def(Args &&...args) noexcept {
    // TODO: generate default initializer list?
    return Var<expr_value_t<T>>{std::forward<Args>(args)...};
}

/// Define a var using rvalue
template<typename T>
[[nodiscard]] auto def(T &&x) noexcept -> Var<expr_value_t<T>> {
    return Var{Expr{std::forward<T>(x)}};
}

/// Define a var using expression
template<typename T>
[[nodiscard]] auto def(const Expression *expr) noexcept -> Var<expr_value_t<T>> {
    return Var{Expr<expr_value_t<T>>{expr}};
}

}// namespace dsl

namespace detail {

/// Return make_vector CallOp according to given type
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
    } else if constexpr (std::is_same_v<T, half2>) {
        return CallOp::MAKE_HALF2;
    } else if constexpr (std::is_same_v<T, half3>) {
        return CallOp::MAKE_HALF3;
    } else if constexpr (std::is_same_v<T, half4>) {
        return CallOp::MAKE_HALF4;
    } else if constexpr (std::is_same_v<T, short2>) {
        return CallOp::MAKE_SHORT2;
    } else if constexpr (std::is_same_v<T, short3>) {
        return CallOp::MAKE_SHORT3;
    } else if constexpr (std::is_same_v<T, short4>) {
        return CallOp::MAKE_SHORT4;
    } else if constexpr (std::is_same_v<T, ushort2>) {
        return CallOp::MAKE_USHORT2;
    } else if constexpr (std::is_same_v<T, ushort3>) {
        return CallOp::MAKE_USHORT3;
    } else if constexpr (std::is_same_v<T, ushort4>) {
        return CallOp::MAKE_USHORT4;
    } else if constexpr (std::is_same_v<T, slong2>) {
        return CallOp::MAKE_LONG2;
    } else if constexpr (std::is_same_v<T, slong3>) {
        return CallOp::MAKE_LONG3;
    } else if constexpr (std::is_same_v<T, slong4>) {
        return CallOp::MAKE_LONG4;
    } else if constexpr (std::is_same_v<T, ulong2>) {
        return CallOp::MAKE_ULONG2;
    } else if constexpr (std::is_same_v<T, ulong3>) {
        return CallOp::MAKE_ULONG3;
    } else if constexpr (std::is_same_v<T, ulong4>) {
        return CallOp::MAKE_ULONG4;
    } else {
        static_assert(always_false_v<T>);
    }
}

#define LUISA_EXPR(value) \
    detail::extract_expression(std::forward<decltype(value)>(value))

/// Make vector2 (s, s)
template<typename Ts>
    requires is_dsl_v<Ts> && is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector2(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 2>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(s)}));
}

/// Make vector2 (x, y)
template<typename Tx, typename Ty>
    requires any_dsl_v<Tx, Ty> &&
             is_scalar_expr_v<Tx> &&
             is_scalar_expr_v<Ty> &&
             is_same_expr_v<Tx, Ty>
[[nodiscard]] inline auto make_vector2(Tx &&x, Ty &&y) noexcept {
    using V = Vector<expr_value_t<Tx>, 2>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y)}));
}

/// Make vector2 from vector.
/// Only first two values will be taken if vector's length is larger than 2.
template<typename T, typename Tv>
    requires is_dsl_v<Tv> && is_vector_expr_v<Tv>
[[nodiscard]] inline auto make_vector2(Tv &&v) noexcept {
    using V = Vector<T, 2>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(v)}));
}

/// Make vector3 (s, s, s)
template<typename Ts>
    requires is_dsl_v<Ts> && is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector3(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 3>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(s)}));
}

/// Make vector3 (x, y, z)
template<typename Tx, typename Ty, typename Tz>
    requires any_dsl_v<Tx, Ty, Tz> &&
             is_scalar_expr_v<Tx> &&
             is_scalar_expr_v<Ty> &&
             is_scalar_expr_v<Tz> &&
             is_same_expr_v<Tx, Ty, Tz>
[[nodiscard]] inline auto make_vector3(Tx &&x, Ty &&y, Tz &&z) noexcept {
    using V = Vector<expr_value_t<Tx>, 3>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y), LUISA_EXPR(z)}));
}

/// Make vector3 from vector
/// Only first three values will be taken if vector's length is larger than 3.
template<typename T, typename Tv>
    requires is_dsl_v<Tv> &&
             std::disjunction_v<
                 is_vector3_expr<Tv>,
                 is_vector4_expr<Tv>>
[[nodiscard]] inline auto make_vector3(Tv &&v) noexcept {
    using V = Vector<T, 3>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(v)}));
}

/// Make vector3 (x, y, z) from (x, y) and z
template<typename Txy, typename Tz>
    requires any_dsl_v<Txy, Tz> &&
             is_vector2_expr_v<Txy> &&
             is_scalar_expr_v<Tz> &&
             std::same_as<
                 vector_expr_element_t<Txy>,
                 expr_value_t<Tz>>
[[nodiscard]] inline auto make_vector3(Txy &&xy, Tz &&z) noexcept {
    using V = Vector<expr_value_t<Tz>, 3>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xy), LUISA_EXPR(z)}));
}

/// Make vector3 (x, y, z) from x and (y, z)
template<typename Tx, typename Tyz>
    requires any_dsl_v<Tx, Tyz> &&
             is_scalar_expr_v<Tx> &&
             is_vector2_expr_v<Tyz> &&
             std::same_as<
                 expr_value_t<Tx>,
                 vector_expr_element_t<Tyz>>
[[nodiscard]] inline auto make_vector3(Tx &&x, Tyz &&yz) noexcept {
    using V = Vector<expr_value_t<Tx>, 3>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(yz)}));
}

/// Make vector4 (s, s, s, s)
template<typename Ts>
    requires is_dsl_v<Ts> && is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector4(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(s)}));
}

/// Make vector4 (x, y, z, w)
template<typename Tx, typename Ty, typename Tz, typename Tw>
    requires any_dsl_v<Tx, Ty, Tz, Tw> &&
             is_scalar_expr_v<Tx> &&
             is_scalar_expr_v<Ty> &&
             is_scalar_expr_v<Tz> &&
             is_scalar_expr_v<Tw> &&
             is_same_expr_v<Tx, Ty, Tz, Tw>
[[nodiscard]] inline auto make_vector4(Tx &&x, Ty &&y, Tz &&z, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y), LUISA_EXPR(z), LUISA_EXPR(w)}));
}

/// Make vector4 from vector4
template<typename T, typename Tv>
    requires is_dsl_v<Tv> && is_vector4_expr_v<Tv>
[[nodiscard]] inline auto make_vector4(Tv &&v) noexcept {
    using V = Vector<T, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(v)}));
}

/// Make vector4 (x, y, z, w) from (x, y), z and w
template<typename Txy, typename Tz, typename Tw>
    requires any_dsl_v<Txy, Tz, Tw> &&
             is_vector2_expr_v<Txy> &&
             is_scalar_expr_v<Tz> &&
             is_scalar_expr_v<Tw> &&
             concepts::same<
                 vector_expr_element_t<Txy>,
                 expr_value_t<Tz>,
                 expr_value_t<Tw>>
[[nodiscard]] inline auto make_vector4(Txy &&xy, Tz &&z, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tz>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xy), LUISA_EXPR(z), LUISA_EXPR(w)}));
}

/// Make vector4 (x, y, z, w) from x, (y, z) and w
template<typename Tx, typename Tyz, typename Tw>
    requires any_dsl_v<Tx, Tyz, Tw> &&
             is_scalar_expr_v<Tx> &&
             is_vector2_expr_v<Tyz> &&
             is_scalar_expr_v<Tw> &&
             concepts::same<
                 expr_value_t<Tx>,
                 vector_expr_element_t<Tyz>,
                 expr_value_t<Tw>>
[[nodiscard]] inline auto make_vector4(Tx &&x, Tyz &&yz, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(yz), LUISA_EXPR(w)}));
}

/// Make vector4 (x, y, z, w) from x, y and (z, w)
template<typename Tx, typename Ty, typename Tzw>
    requires any_dsl_v<Tx, Ty, Tzw> &&
             is_scalar_expr_v<Tx> &&
             is_scalar_expr_v<Ty> &&
             is_vector2_expr_v<Tzw> &&
             concepts::same<
                 expr_value_t<Tx>,
                 expr_value_t<Ty>,
                 vector_expr_element_t<Tzw>>
[[nodiscard]] inline auto make_vector4(Tx &&x, Ty &&y, Tzw &&zw) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(y), LUISA_EXPR(zw)}));
}

/// Make vector4 (x, y, z, w) from (x, y) and (z, w)
template<typename Txy, typename Tzw>
    requires any_dsl_v<Txy, Tzw> &&
             is_vector2_expr_v<Txy> &&
             is_vector2_expr_v<Tzw> &&
             std::same_as<
                 vector_expr_element_t<Txy>,
                 vector_expr_element_t<Tzw>>
[[nodiscard]] inline auto make_vector4(Txy &&xy, Tzw &&zw) noexcept {
    using V = Vector<vector_expr_element_t<Txy>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xy), LUISA_EXPR(zw)}));
}

/// Make vector4 (x, y, z, w) from (x, y, z) and w
template<typename Txyz, typename Tw>
    requires any_dsl_v<Txyz, Tw> &&
             is_vector3_expr_v<Txyz> &&
             is_scalar_expr_v<Tw> &&
             std::same_as<
                 vector_expr_element_t<Txyz>,
                 expr_value_t<Tw>>
[[nodiscard]] inline auto make_vector4(Txyz &&xyz, Tw &&w) noexcept {
    using V = Vector<expr_value_t<Tw>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(xyz), LUISA_EXPR(w)}));
}

/// Make vector4 (x, y, z, w) from x and (y, z, w)
template<typename Tx, typename Tyzw>
    requires any_dsl_v<Tx, Tyzw> &&
             is_scalar_expr_v<Tx> &&
             is_vector3_expr_v<Tyzw> &&
             std::same_as<
                 expr_value_t<Tx>,
                 vector_expr_element_t<Tyzw>>
[[nodiscard]] inline auto make_vector4(Tx &&x, Tyzw &&yzw) noexcept {
    using V = Vector<expr_value_t<Tx>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(x), LUISA_EXPR(yzw)}));
}

template<typename T, size_t N>
struct vectorized {
    using type = Vector<T, N>;
};

template<typename T>
struct vectorized<T, 1u> {
    using type = T;
};

template<typename T, size_t N>
using vectorized_t = typename vectorized<T, N>::type;

/// Make vectorN of scalar x
template<size_t N, typename T>
    requires is_scalar_expr_v<T>
[[nodiscard]] inline auto vectorize(T &&x) noexcept {
    if constexpr (N == 1u) {
        return Expr{std::forward<T>(x)};
    } else if constexpr (N == 2u) {
        return make_vector2(Expr{std::forward<T>(x)});
    } else if constexpr (N == 3u) {
        return make_vector3(Expr{std::forward<T>(x)});
    } else if constexpr (N == 4u) {
        return make_vector4(Expr{std::forward<T>(x)});
    } else {
        static_assert(always_false_v<T>);
    }
}

/// Make vectorN of vectorN x
template<size_t N, typename T>
    requires is_vector_expr_v<T>
[[nodiscard]] inline auto vectorize(T &&x) noexcept {
    static_assert(vector_expr_dimension_v<T> == N);
    return Expr{std::forward<T>(x)};
}

template<typename... T>
struct vectorized_dimension {
    static constexpr size_t value = 0u;
};

template<typename T>
struct vectorized_dimension<T> {
    static constexpr auto value = vector_expr_dimension_v<T>;
};

template<typename First, typename... Other>
struct vectorized_dimension<First, Other...> {
    static constexpr auto value = std::max(
        vector_expr_dimension_v<First>,
        vectorized_dimension<Other...>::value);
};

template<typename... T>
constexpr auto vectorized_dimension_v = vectorized_dimension<T...>::value;

template<size_t max, typename T>
using is_dimension_compatible = std::disjunction<
    std::bool_constant<vector_expr_dimension_v<T> == max>,
    std::bool_constant<vector_expr_dimension_v<T> == 1u>>;

template<typename... T>
concept vector_expr_compatible = std::conjunction<
    is_dimension_compatible<
        vectorized_dimension_v<T...>, T>...>::value;

/// Vectorized call
template<typename Scalar, typename... T>
    requires vector_expr_compatible<T...>
[[nodiscard]] inline auto make_vector_call(CallOp op, T &&...x) noexcept {
    constexpr auto N = vectorized_dimension_v<T...>;
    using V = vectorized_t<Scalar, N>;
    return def<V>(
        detail::FunctionBuilder::current()->call(
            Type::of<V>(), op,
            {vectorize<N>(std::forward<T>(x)).expression()...}));
}

}// namespace detail

inline namespace dsl {// to avoid conflicts

using namespace luisa;

using luisa::make_bool2;
using luisa::make_bool3;
using luisa::make_bool4;
using luisa::make_float2;
using luisa::make_float2x2;
using luisa::make_float3;
using luisa::make_float3x3;
using luisa::make_float4;
using luisa::make_float4x4;
using luisa::make_half2;
using luisa::make_half3;
using luisa::make_half4;
using luisa::make_int2;
using luisa::make_int3;
using luisa::make_int4;
using luisa::make_short2;
using luisa::make_short3;
using luisa::make_short4;
using luisa::make_slong2;
using luisa::make_slong3;
using luisa::make_slong4;
using luisa::make_uint2;
using luisa::make_uint3;
using luisa::make_uint4;
using luisa::make_ulong2;
using luisa::make_ulong3;
using luisa::make_ulong4;
using luisa::make_ushort2;
using luisa::make_ushort3;
using luisa::make_ushort4;

#define LUISA_MAKE_VECTOR(type)                                  \
    template<typename S>                                         \
        requires is_dsl_v<S> && is_same_expr_v<S, type>          \
    [[nodiscard]] inline auto make_##type##2(S && s) noexcept {  \
        return detail::make_vector2(std::forward<S>(s));         \
    }                                                            \
    template<typename X, typename Y>                             \
        requires any_dsl_v<X, Y> &&                              \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<Y, type>                         \
    [[nodiscard]] inline auto make_##type##2(                    \
        X && x, Y && y) noexcept {                               \
        return detail::make_vector2(                             \
            std::forward<X>(x),                                  \
            std::forward<Y>(y));                                 \
    }                                                            \
    template<typename V>                                         \
        requires is_dsl_v<V> && is_same_expr_v<V, type##3>       \
    [[nodiscard]] inline auto make_##type##2(V && v) noexcept {  \
        return detail::make_vector2<type>(std::forward<V>(v));   \
    }                                                            \
    template<typename V>                                         \
        requires is_dsl_v<V> && is_same_expr_v<V, type##4>       \
    [[nodiscard]] inline auto make_##type##2(V && v) noexcept {  \
        return detail::make_vector2<type>(std::forward<V>(v));   \
    }                                                            \
    template<typename Tv>                                        \
        requires is_dsl_v<Tv> && is_vector2_expr_v<Tv>           \
    [[nodiscard]] inline auto make_##type##2(Tv && v) noexcept { \
        return detail::make_vector2<type>(std::forward<Tv>(v));  \
    }                                                            \
                                                                 \
    template<typename X, typename Y, typename Z>                 \
        requires any_dsl_v<X, Y, Z> &&                           \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<Y, type> &&                      \
                 is_same_expr_v<Z, type>                         \
    [[nodiscard]] inline auto make_##type##3(                    \
        X && x, Y && y, Z && z) noexcept {                       \
        return detail::make_vector3(                             \
            std::forward<X>(x),                                  \
            std::forward<Y>(y),                                  \
            std::forward<Z>(z));                                 \
    }                                                            \
    template<typename S>                                         \
        requires is_dsl_v<S> && is_same_expr_v<S, type>          \
    [[nodiscard]] inline auto make_##type##3(S && s) noexcept {  \
        return detail::make_vector3(std::forward<S>(s));         \
    }                                                            \
    template<typename V, typename Z>                             \
        requires any_dsl_v<V, Z> &&                              \
                 is_same_expr_v<V, type##2> &&                   \
                 is_same_expr_v<Z, type>                         \
    [[nodiscard]] inline auto make_##type##3(                    \
        V && v, Z && z) noexcept {                               \
        return detail::make_vector3(                             \
            std::forward<V>(v),                                  \
            std::forward<Z>(z));                                 \
    }                                                            \
    template<typename X, typename V>                             \
        requires any_dsl_v<X, V> &&                              \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<V, type##2>                      \
    [[nodiscard]] inline auto make_##type##3(                    \
        X && x, V && v) noexcept {                               \
        return detail::make_vector3(                             \
            std::forward<X>(x),                                  \
            std::forward<V>(v));                                 \
    }                                                            \
    template<typename V>                                         \
        requires is_dsl_v<V> && is_same_expr_v<V, type##4>       \
    [[nodiscard]] inline auto make_##type##3(V && v) noexcept {  \
        return detail::make_vector3<type>(std::forward<V>(v));   \
    }                                                            \
    template<typename Tv>                                        \
        requires is_dsl_v<Tv> && is_vector3_expr_v<Tv>           \
    [[nodiscard]] inline auto make_##type##3(Tv && v) noexcept { \
        return detail::make_vector3<type>(std::forward<Tv>(v));  \
    }                                                            \
                                                                 \
    template<typename S>                                         \
        requires is_dsl_v<S> && is_same_expr_v<S, type>          \
    [[nodiscard]] inline auto make_##type##4(S && s) noexcept {  \
        return detail::make_vector4(std::forward<S>(s));         \
    }                                                            \
    template<typename X, typename Y, typename Z, typename W>     \
        requires any_dsl_v<X, Y, Z, W> &&                        \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<Y, type> &&                      \
                 is_same_expr_v<Z, type> &&                      \
                 is_same_expr_v<W, type>                         \
    [[nodiscard]] inline auto make_##type##4(                    \
        X && x, Y && y, Z && z, W && w) noexcept {               \
        return detail::make_vector4(                             \
            std::forward<X>(x),                                  \
            std::forward<Y>(y),                                  \
            std::forward<Z>(z),                                  \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename V, typename Z, typename W>                 \
        requires any_dsl_v<V, Z, W> &&                           \
                 is_same_expr_v<V, type##2> &&                   \
                 is_same_expr_v<Z, type> &&                      \
                 is_same_expr_v<W, type>                         \
    [[nodiscard]] inline auto make_##type##4(                    \
        V && v, Z && z, W && w) noexcept {                       \
        return detail::make_vector4(                             \
            std::forward<V>(v),                                  \
            std::forward<Z>(z),                                  \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename X, typename YZ, typename W>                \
        requires any_dsl_v<X, YZ, W> &&                          \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<YZ, type##2> &&                  \
                 is_same_expr_v<W, type>                         \
    [[nodiscard]] inline auto make_##type##4(                    \
        X && x, YZ && yz, W && w) noexcept {                     \
        return detail::make_vector4(                             \
            std::forward<X>(x),                                  \
            std::forward<YZ>(yz),                                \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename X, typename Y, typename ZW>                \
        requires any_dsl_v<X, Y, ZW> &&                          \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<Y, type> &&                      \
                 is_same_expr_v<ZW, type##2>                     \
    [[nodiscard]] inline auto make_##type##4(                    \
        X && x, Y && y, ZW && zw) noexcept {                     \
        return detail::make_vector4(                             \
            std::forward<X>(x),                                  \
            std::forward<Y>(y),                                  \
            std::forward<ZW>(zw));                               \
    }                                                            \
    template<typename XY, typename ZW>                           \
        requires any_dsl_v<XY, ZW> &&                            \
                 is_same_expr_v<XY, type##2> &&                  \
                 is_same_expr_v<ZW, type##2>                     \
    [[nodiscard]] inline auto make_##type##4(                    \
        XY && xy, ZW && zw) noexcept {                           \
        return detail::make_vector4(                             \
            std::forward<XY>(xy),                                \
            std::forward<ZW>(zw));                               \
    }                                                            \
    template<typename XYZ, typename W>                           \
        requires any_dsl_v<XYZ, W> &&                            \
                 is_same_expr_v<XYZ, type##3> &&                 \
                 is_same_expr_v<W, type>                         \
    [[nodiscard]] inline auto make_##type##4(                    \
        XYZ && xyz, W && w) noexcept {                           \
        return detail::make_vector4(                             \
            std::forward<XYZ>(xyz),                              \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename X, typename YZW>                           \
        requires any_dsl_v<X, YZW> &&                            \
                 is_same_expr_v<X, type> &&                      \
                 is_same_expr_v<YZW, type##3>                    \
    [[nodiscard]] inline auto make_##type##4(                    \
        X && x, YZW && yzw) noexcept {                           \
        return detail::make_vector4(                             \
            std::forward<X>(x),                                  \
            std::forward<YZW>(yzw));                             \
    }                                                            \
    template<typename Tv>                                        \
        requires is_dsl_v<Tv> && is_vector4_expr_v<Tv>           \
    [[nodiscard]] inline auto make_##type##4(Tv && v) noexcept { \
        return detail::make_vector4<type>(std::forward<Tv>(v));  \
    }

LUISA_MAKE_VECTOR(bool)
LUISA_MAKE_VECTOR(int)
LUISA_MAKE_VECTOR(uint)
LUISA_MAKE_VECTOR(float)
LUISA_MAKE_VECTOR(short)
LUISA_MAKE_VECTOR(ushort)
LUISA_MAKE_VECTOR(slong)
LUISA_MAKE_VECTOR(ulong)
LUISA_MAKE_VECTOR(half)

#undef LUISA_MAKE_VECTOR

// make float2x2

/// Make float2x2 from 2 column vector float2
template<typename C0, typename C1>
    requires any_dsl_v<C0, C1> &&
             is_same_expr_v<C0, float2> &&
             is_same_expr_v<C1, float2>
[[nodiscard]] inline auto make_float2x2(
    C0 &&c0, C1 &&c1) noexcept {
    return def<float2x2>(
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {LUISA_EXPR(c0), LUISA_EXPR(c1)}));
}

/// Make float2x2 [ [M00, M10], [M01, M11] ]
template<typename M00, typename M01, typename M10, typename M11>
    requires any_dsl_v<M00, M01, M10, M11> &&
             is_floating_point_expr_v<M00> &&
             is_floating_point_expr_v<M01> &&
             is_floating_point_expr_v<M10> &&
             is_floating_point_expr_v<M11>
[[nodiscard]] inline auto make_float2x2(
    M00 &&m00, M01 &&m01,
    M10 &&m10, M11 &&m11) noexcept {
    return make_float2x2(
        make_float2(m00, m01),
        make_float2(m10, m11));
}

/// Make float2x2 from matrix.
/// Submatrix will be taken if matrix is larger than 2x2.
template<typename M>
    requires is_dsl_v<M> && is_matrix_expr_v<M>
[[nodiscard]] inline auto make_float2x2(M &&m) noexcept {
    return make_float2x2(
        make_float2(m[0]),
        make_float2(m[1]));
}

/// Make float3x3 from 3 column vector float3
template<typename C0, typename C1, typename C2>
    requires any_dsl_v<C0, C1, C2> &&
             is_same_expr_v<C0, float3> &&
             is_same_expr_v<C1, float3> &&
             is_same_expr_v<C2, float3>
[[nodiscard]] inline auto make_float3x3(C0 &&c0, C1 &&c1, C2 &&c2) noexcept {
    return def<float3x3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {LUISA_EXPR(c0), LUISA_EXPR(c1), LUISA_EXPR(c2)}));
}

/// Make float3x3 [ [M00, M10, M20], [M01, M11, M21], [M02, M12, M22] ]
template<typename M00, typename M01, typename M02,
         typename M10, typename M11, typename M12,
         typename M20, typename M21, typename M22>
    requires any_dsl_v<M00, M01, M02, M10, M11, M12, M20, M21, M22> &&
             is_same_expr_v<M00, float> && is_same_expr_v<M01, float> && is_same_expr_v<M02, float> &&
             is_same_expr_v<M10, float> && is_same_expr_v<M11, float> && is_same_expr_v<M12, float> &&
             is_same_expr_v<M20, float> && is_same_expr_v<M21, float> && is_same_expr_v<M22, float>
[[nodiscard]] inline auto make_float3x3(
    M00 &&m00, M01 &&m01, M02 &&m02,
    M10 &&m10, M11 &&m11, M12 &&m12,
    M20 &&m20, M21 &&m21, M22 &&m22) noexcept {
    return make_float3x3(
        make_float3(m00, m01, m02),
        make_float3(m10, m11, m12),
        make_float3(m20, m21, m22));
}

/// Make float3x3 from float2x2 [ [M, 0], [0, 1] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix2_expr_v<M>
[[nodiscard]] inline auto make_float3x3(M &&m) noexcept {
    return make_float3x3(
        make_float3(m[0], 0.f),
        make_float3(m[1], 0.f),
        luisa::make_float3(0.f, 0.f, 1.f));
}

/// Make float3x3 from float3x3/float4x4
/// Submatrix will be taken if matrix is larger than 3x3.
template<typename M>
    requires is_dsl_v<M> && std::disjunction_v<is_matrix3_expr<M>, is_matrix4_expr<M>>
[[nodiscard]] inline auto make_float3x3(M &&m) noexcept {
    return make_float3x3(
        make_float3(m[0]),
        make_float3(m[1]),
        make_float3(m[2]));
}

/// Make float4x4 from 4 column vector float4
template<typename C0, typename C1, typename C2, typename C3>
    requires any_dsl_v<C0, C1, C2, C3> &&
             is_same_expr_v<C0, float4> &&
             is_same_expr_v<C1, float4> &&
             is_same_expr_v<C2, float4> &&
             is_same_expr_v<C3, float4>
[[nodiscard]] inline auto make_float4x4(
    C0 &&c0, C1 &&c1, C2 &&c2, C3 &&c3) noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {LUISA_EXPR(c0), LUISA_EXPR(c1), LUISA_EXPR(c2), LUISA_EXPR(c3)}));
}

/// Make float4x4 [ [M00, M10, M20, M30], [M01, M11, M21, M31], [M02, M12, M22, M32], [M03, M13, M23, M33] ]
template<
    typename M00, typename M01, typename M02, typename M03,
    typename M10, typename M11, typename M12, typename M13,
    typename M20, typename M21, typename M22, typename M23,
    typename M30, typename M31, typename M32, typename M33>
    requires any_dsl_v<
                 M00, M01, M02, M03,
                 M10, M11, M12, M13,
                 M20, M21, M22, M23,
                 M30, M31, M32, M33> &&
             is_same_expr_v<M00, float> && is_same_expr_v<M01, float> && is_same_expr_v<M02, float> && is_same_expr_v<M03, float> &&
             is_same_expr_v<M10, float> && is_same_expr_v<M11, float> && is_same_expr_v<M12, float> && is_same_expr_v<M13, float> &&
             is_same_expr_v<M20, float> && is_same_expr_v<M21, float> && is_same_expr_v<M22, float> && is_same_expr_v<M23, float> &&
             is_same_expr_v<M30, float> && is_same_expr_v<M31, float> && is_same_expr_v<M32, float> && is_same_expr_v<M33, float>
[[nodiscard]] inline auto make_float4x4(
    M00 &&m00, M01 &&m01, M02 &&m02, M03 &&m03,
    M10 &&m10, M11 &&m11, M12 &&m12, M13 &&m13,
    M20 &&m20, M21 &&m21, M22 &&m22, M23 &&m23,
    M30 &&m30, M31 &&m31, M32 &&m32, M33 &&m33) noexcept {
    return make_float4x4(
        make_float4(m00, m01, m02, m03),
        make_float4(m10, m11, m12, m13),
        make_float4(m20, m21, m22, m23),
        make_float4(m30, m31, m32, m33));
}

/// Make float4x4 from float2x2 [ [M, 0], [0, I] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix2_expr_v<M>
[[nodiscard]] inline auto make_float4x4(M &&m) noexcept {
    return make_float4x4(
        make_float4(m[0], 0.f, 0.f),
        make_float4(m[1], 0.f, 0.f),
        luisa::make_float4(0.f, 0.f, 1.f, 0.f),
        luisa::make_float4(0.f, 0.f, 0.f, 1.f));
}

/// Make float4x4 from float3x3 [ [M, 0], [0, 1] ]
template<typename M>
    requires is_dsl_v<M> && is_matrix3_expr_v<M>
[[nodiscard]] inline auto make_float4x4(M &&m) noexcept {
    return make_float4x4(
        make_float4(m[0], 0.f),
        make_float4(m[1], 0.f),
        make_float4(m[2], 0.f),
        luisa::make_float4(0.f, 0.f, 0.f, 1.f));
}

/// Make float4x4 from float4x4
template<typename M>
    requires is_dsl_v<M> && is_matrix4_expr_v<M>
[[nodiscard]] inline auto make_float4x4(M &&m) noexcept {
    return def(std::forward<M>(m));
}

/// Call function all on bool vector.
/// Test if all bools are true.
template<typename Tx>
    requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto all(Tx &&x) noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ALL,
            {LUISA_EXPR(x)}));
}

/// Call function any on bool vector.
/// Test if any bool is true
template<typename Tx>
    requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto any(Tx &&x) noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ANY,
            {LUISA_EXPR(x)}));
}

/// Call function none on bool vector.
/// Test if none bool is true
template<typename Tx>
    requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto none(Tx &&x) noexcept {
    return !any(std::forward<Tx>(x));
}

/// Select function. p ? t : f.
/// Tf: scalar or vector, Tt: scalar or vector, Tp: bool scalar or vector, vectorize
template<typename Tf, typename Tt, typename Tp>
    requires any_dsl_v<Tf, Tt, Tp> &&
             is_bool_or_vector_expr_v<Tp> &&
             std::disjunction_v<is_scalar_expr<Tf>, is_vector_expr<Tf>> &&
             std::disjunction_v<is_scalar_expr<Tt>, is_vector_expr<Tt>> &&
             is_vector_expr_same_element_v<Tf, Tt>
[[nodiscard]] inline auto select(Tf &&f, Tt &&t, Tp &&p) noexcept {
    return detail::make_vector_call<vector_expr_element_t<Tf>>(
        CallOp::SELECT,
        std::forward<Tf>(f),
        std::forward<Tt>(t),
        std::forward<Tp>(p));
}

/// Select function. p ? t : f.
/// Tf == Tt: non-scalar and non-vector, Tp: bool scalar, no vectorization
template<typename Tf, typename Tt, typename Tp>
    requires any_dsl_v<Tf, Tt, Tp> &&
             is_boolean_expr_v<Tp> &&
             is_same_expr_v<Tf, Tt> &&
             std::negation_v<
                 std::disjunction<
                     is_scalar_expr<Tf>,
                     is_vector_expr<Tf>>>
[[nodiscard]] inline auto select(Tf &&f, Tt &&t, Tp &&p) noexcept {
    using T = expr_value_t<Tf>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::SELECT,
            {LUISA_EXPR(f), LUISA_EXPR(t), LUISA_EXPR(p)}));
}

/// If-then-else. If p then t else f.
template<typename Tp, typename Tt, typename Tf>
    requires any_dsl_v<Tp, Tt, Tf> &&
             is_bool_or_vector_expr_v<Tp> &&
             std::disjunction_v<is_dsl<Tt>, is_basic<Tt>> &&
             std::disjunction_v<is_dsl<Tf>, is_basic<Tf>>
[[nodiscard]] inline auto ite(Tp &&p, Tt &&t, Tf &&f) noexcept {
    return select(std::forward<Tf>(f),
                  std::forward<Tt>(t),
                  std::forward<Tp>(p));
}

/// Clamp v in lower bound l and upper bound u
template<typename Tv, typename Tl, typename Tu>
    requires any_dsl_v<Tv, Tl, Tu> && is_vector_expr_same_element_v<Tv, Tl, Tu>
[[nodiscard]] inline auto clamp(Tv &&v, Tl &&l, Tu &&u) noexcept {
    return detail::make_vector_call<vector_expr_element_t<Tv>>(
        CallOp::CLAMP,
        std::forward<Tv>(v),
        std::forward<Tl>(l),
        std::forward<Tu>(u));
}

/// Lerp in [a, b] with param t
template<typename Ta, typename Tb, typename Tt>
    requires any_dsl_v<Ta, Tb, Tt> && is_float_or_vector_expr_v<Ta> && is_vector_expr_same_element_v<Ta, Tb, Tt>
[[nodiscard]] inline auto lerp(Ta &&a, Tb &&b, Tt &&t) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LERP,
        std::forward<Ta>(a),
        std::forward<Tb>(b),
        std::forward<Tt>(t));
}

/// Fma. Calcs x * y + z
template<typename X, typename Y, typename Z>
    requires any_dsl_v<X, Y, Z> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y> && is_float_or_vector_expr_v<Z>
[[nodiscard]] inline auto fma(X &&x, Y &&y, Z &&z) noexcept {
    return detail::make_vector_call<float>(
        CallOp::FMA,
        std::forward<X>(x),
        std::forward<Y>(y),
        std::forward<Z>(z));
}

/// Saturate. Equals to clamp(v, 0, 1)
template<typename Tv>
    requires is_dsl_v<Tv> && is_float_or_vector_expr_v<Tv>
[[nodiscard]] inline auto saturate(Tv &&v) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SATURATE, std::forward<Tv>(v));
}

/// Step. x < step ? 0 : 1
template<typename E, typename X>
    requires any_dsl_v<E, X> && is_float_or_vector_expr_v<E> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto step(E &&edge, X &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::STEP,
        std::forward<E>(edge),
        std::forward<X>(x));
}

/// Smooth step.
/// Reference https://en.wikipedia.org/wiki/Smoothstep
template<typename L, typename R, typename T>
    requires any_dsl_v<L, R, T> && is_float_or_vector_expr_v<L> && is_float_or_vector_expr_v<R> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto smoothstep(L &&left, R &&right, T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SMOOTHSTEP,
        std::forward<L>(left),
        std::forward<R>(right),
        std::forward<T>(x));
}

/// Abs of float or vector.
template<typename Tx>
    requires is_dsl_v<Tx> && is_float_or_vector_expr_v<Tx>
[[nodiscard]] inline auto abs(Tx &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ABS, std::forward<Tx>(x));
}

/// Abs of int or vector.
template<typename Tx>
    requires is_dsl_v<Tx> && is_int_or_vector_expr_v<Tx>
[[nodiscard]] inline auto abs(Tx &&x) noexcept {
    return detail::make_vector_call<int>(
        CallOp::ABS, std::forward<Tx>(x));
}

/// Mod. x - y * floor(x/y)
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y>
[[nodiscard]] inline auto mod(X &&x_in, Y &&y_in) noexcept {
    auto x = def(std::forward<X>(x_in));
    auto y = def(std::forward<Y>(y_in));
    return x - y * floor(x / y);
}

/// Fmod. x - y * trunc(x/y)
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y>
[[nodiscard]] inline auto fmod(X &&x_in, Y &&y_in) noexcept {
    auto x = def(std::forward<X>(x_in));
    auto y = def(std::forward<Y>(y_in));
    return x - y * trunc(x / y);
}

/// Min.
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_vector_expr_same_element_v<X, Y>
[[nodiscard]] inline auto min(X &&x, Y &&y) noexcept {
    return detail::make_vector_call<vector_expr_element_t<X>>(
        CallOp::MIN,
        std::forward<X>(x),
        std::forward<Y>(y));
}

/// Max.
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_vector_expr_same_element_v<X, Y>
[[nodiscard]] inline auto max(X &&x, Y &&y) noexcept {
    return detail::make_vector_call<vector_expr_element_t<X>>(
        CallOp::MAX,
        std::forward<X>(x),
        std::forward<Y>(y));
}

/// Reduce sum.
template<typename T>
    requires is_dsl_v<T> && is_vector_expr_v<T>
[[nodiscard]] inline auto reduce_sum(T &&x) noexcept {
    using E = vector_expr_element_t<T>;
    return def<E>(detail::FunctionBuilder::current()->call(
        Type::of<E>(), CallOp::REDUCE_SUM, {LUISA_EXPR(x)}));
}

/// Reduce product.
template<typename T>
    requires is_dsl_v<T> && is_vector_expr_v<T>
[[nodiscard]] inline auto reduce_prod(T &&x) noexcept {
    using E = vector_expr_element_t<T>;
    return def<E>(detail::FunctionBuilder::current()->call(
        Type::of<E>(), CallOp::REDUCE_PRODUCT, {LUISA_EXPR(x)}));
}

/// Reduce min.
template<typename T>
    requires is_dsl_v<T> && is_vector_expr_v<T>
[[nodiscard]] inline auto reduce_min(T &&x) noexcept {
    using E = vector_expr_element_t<T>;
    return def<E>(detail::FunctionBuilder::current()->call(
        Type::of<E>(), CallOp::REDUCE_MIN, {LUISA_EXPR(x)}));
}

/// Reduce max.
template<typename T>
    requires is_dsl_v<T> && is_vector_expr_v<T>
[[nodiscard]] inline auto reduce_max(T &&x) noexcept {
    using E = vector_expr_element_t<T>;
    return def<E>(detail::FunctionBuilder::current()->call(
        Type::of<E>(), CallOp::REDUCE_MAX, {LUISA_EXPR(x)}));
}

/// Clz. Count leading zeros.
template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto clz(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::CLZ, std::forward<X>(x));
}

/// Ctz. Coount trailing zeros
template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto ctz(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::CTZ, std::forward<X>(x));
}

/// Popcount. Count number of 1 bits in x's binary representation
template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto popcount(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::POPCOUNT, std::forward<X>(x));
}

/// Reverse. Reverse bits in x
template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto reverse(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::REVERSE, std::forward<X>(x));
}

/// Is x infinity.
template<typename X>
    requires is_dsl_v<X> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto isinf(X &&x) noexcept {
    return detail::make_vector_call<bool>(
        CallOp::ISINF, std::forward<X>(x));
}

/// Is x nan.
template<typename X>
    requires is_dsl_v<X> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto isnan(X &&x) noexcept {
    return detail::make_vector_call<bool>(
        CallOp::ISNAN, std::forward<X>(x));
}

/// arccos
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto acos(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ACOS, std::forward<T>(x));
}

/// arccosh
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto acosh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ACOSH, std::forward<T>(x));
}

/// arcsin
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto asin(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ASIN, std::forward<T>(x));
}

/// arcsinh
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto asinh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ASINH, std::forward<T>(x));
}

/// arctan
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto atan(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ATAN, std::forward<T>(x));
}

/// arctan2
template<typename Y, typename X>
    requires any_dsl_v<Y, X> && is_float_or_vector_expr_v<Y> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto atan2(Y &&y, X &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ATAN2,
        std::forward<Y>(y),
        std::forward<X>(x));
}

/// arctanh
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto atanh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ATANH, std::forward<T>(x));
}

/// cos
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto cos(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::COS, std::forward<T>(x));
}

/// cosh
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto cosh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::COSH, std::forward<T>(x));
}

/// sin
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto sin(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SIN, std::forward<T>(x));
}

/// sinh
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto sinh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SINH, std::forward<T>(x));
}

/// tan
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto tan(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::TAN, std::forward<T>(x));
}

/// tanh
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto tanh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::TANH, std::forward<T>(x));
}

/// exp. e^x
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto exp(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::EXP, std::forward<T>(x));
}

/// exp2. 2^x
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto exp2(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::EXP2, std::forward<T>(x));
}

/// exp10. 10^x
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto exp10(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::EXP10, std::forward<T>(x));
}

/// log. ln(x)
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto log(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LOG, std::forward<T>(x));
}

/// log2. log_2(x)
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto log2(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LOG2, std::forward<T>(x));
}

/// log10. log_10(x)
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto log10(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LOG10, std::forward<T>(x));
}

/// pow. x^a
template<typename X, typename A>
    requires any_dsl_v<X, A> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<A>
[[nodiscard]] inline auto pow(X &&x, A &&a) noexcept {
    return detail::make_vector_call<float>(
        CallOp::POW,
        std::forward<X>(x),
        std::forward<A>(a));
}

/// sqrt. Square root of x.
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto sqrt(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SQRT, std::forward<T>(x));
}

/// rsqrt. 1/sqrt(x)
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto rsqrt(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::RSQRT, std::forward<T>(x));
}

/// ceil. Return the smallest integer bigger than x.
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto ceil(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::CEIL, std::forward<T>(x));
}

/// floor. Return the biggest integer smaller than x.
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto floor(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::FLOOR, std::forward<T>(x));
}

/// fract. Return x - floor(x)
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto fract(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::FRACT, std::forward<T>(x));
}

/// trunc. Round x to 0.
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto trunc(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::TRUNC, std::forward<T>(x));
}

/// round.
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto round(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ROUND, std::forward<T>(x));
}

/// Radian to degree
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto degrees(T &&x) noexcept {
    return std::forward<T>(x) * (180.0f * constants::inv_pi);
}

/// Degree to radian
template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto radians(T &&x) noexcept {
    return std::forward<T>(x) * (constants::pi / 180.0f);
}

/// Copy y's sign to x
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y>
[[nodiscard]] inline auto copysign(X &&x, Y &&y) noexcept {
    return detail::make_vector_call<float>(
        CallOp::COPYSIGN,
        std::forward<X>(x),
        std::forward<Y>(y));
}

/// Return sgn(x)
template<typename X>
    requires is_dsl_v<X> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto sign(X &&x) noexcept {
    return copysign(1.0f, std::forward<X>(x));
}
template<typename X>
    requires is_dsl_v<X> && (is_scalar_v<expr_value_t<X>> || is_matrix_v<expr_value_t<X>> || is_vector_v<expr_value_t<X>>)
[[nodiscard]] inline auto ddx(X &&x) noexcept {
    using value_type = expr_value_t<X>;
    return def<value_type>(
        detail::FunctionBuilder::current()->call(
            Type::of<value_type>(), CallOp::DDX,
            {LUISA_EXPR(x)}));
}
template<typename X>
    requires is_dsl_v<X> && (is_scalar_v<expr_value_t<X>> || is_matrix_v<expr_value_t<X>> || is_vector_v<expr_value_t<X>>)
[[nodiscard]] inline auto ddy(X &&x) noexcept {
    using value_type = expr_value_t<X>;
    return def<value_type>(
        detail::FunctionBuilder::current()->call(
            Type::of<value_type>(), CallOp::DDY,
            {LUISA_EXPR(x)}));
}

/// Cross product.
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_same_expr_v<X, Y, float3>
[[nodiscard]] inline auto cross(X &&x, Y &&y) noexcept {
    return def<float3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::CROSS,
            {LUISA_EXPR(x), LUISA_EXPR(y)}));
}

/// Dot product.
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_vector_expr_v<X> &&
             is_float_vector_expr_v<Y> && is_vector_expr_same_dimension_v<X, Y>
[[nodiscard]] inline auto dot(X &&x, Y &&y) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DOT,
            {LUISA_EXPR(x), LUISA_EXPR(y)}));
}

/// L2 norm of vector
template<typename Tx>
    requires is_dsl_v<Tx> && is_float_vector_expr_v<Tx>
[[nodiscard]] inline auto length(Tx &&x) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH,
            {LUISA_EXPR(x)}));
}

/// Squared L2 norm
template<typename Tx>
    requires is_dsl_v<Tx> && is_float_vector_expr_v<Tx>
[[nodiscard]] inline auto length_squared(Tx &&x) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH_SQUARED,
            {LUISA_EXPR(x)}));
}

/// L2 norm of (x - y)
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_vector_expr_v<X> &&
             is_float_vector_expr_v<Y> && is_vector_expr_same_dimension_v<X, Y>
[[nodiscard]] inline auto distance(X &&x, Y &&y) noexcept {
    return length(std::forward<X>(x) - std::forward<Y>(y));
}

/// Squared L2 norm of (x - y)
template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_vector_expr_v<X> &&
             is_float_vector_expr_v<Y> && is_vector_expr_same_dimension_v<X, Y>
[[nodiscard]] inline auto distance_squared(X &&x, Y &&y) noexcept {
    return length_squared(std::forward<X>(x) - std::forward<Y>(y));
}

/// Normalize vector
template<typename T>
    requires is_dsl_v<T> && is_float_vector_expr_v<T>
[[nodiscard]] inline auto normalize(T &&x) noexcept {
    return detail::make_vector_call<float>(CallOp::NORMALIZE, std::forward<T>(x));
}

/// Reflect i about n, returns i - 2 * dot(n, i) * n.
template<typename I, typename N>
    requires any_dsl_v<I, N> &&
             std::same_as<expr_value_t<I>, float3> &&
             std::same_as<expr_value_t<N>, float3>
[[nodiscard]] inline auto reflect(I &&i, N &&n) noexcept {
    return detail::make_vector_call<float>(
        CallOp::REFLECT, std::forward<I>(i), std::forward<N>(n));
}

/// Return face forward vector.
/// faceforward orients a vector to point away from a surface as defined by its normal.
/// If dot(Nref, I) < 0 faceforward returns N, otherwise it returns -N.
template<typename N, typename I, typename NRef>
    requires any_dsl_v<N, I, NRef> && is_same_expr_v<N, I, NRef, float3>
[[nodiscard]] inline auto faceforward(N &&n, I &&i, NRef &&n_ref) noexcept {
    return def<float3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::FACEFORWARD,
            {LUISA_EXPR(n), LUISA_EXPR(i), LUISA_EXPR(n_ref)}));
}

/// Determinant of metrix
template<typename Tm>
    requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto determinant(Tm &&m) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DETERMINANT,
            {LUISA_EXPR(m)}));
}

/// Transpose of matrix
template<typename Tm>
    requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto transpose(Tm &&m) noexcept {
    using T = expr_value_t<Tm>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRANSPOSE,
            {LUISA_EXPR(m)}));
}

/// Inverse of matrix
template<typename Tm>
    requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto inverse(Tm &&m) noexcept {
    using T = expr_value_t<Tm>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::INVERSE,
            {LUISA_EXPR(m)}));
}

template<typename T>
[[nodiscard]] auto zero() noexcept {
    using X = expr_value_t<T>;
    return def<X>(detail::FunctionBuilder::current()->call(
        Type::of<X>(), CallOp::ZERO, {}));
}

template<typename T>
[[nodiscard]] auto one() noexcept {
    using X = expr_value_t<T>;
    return def<X>(detail::FunctionBuilder::current()->call(
        Type::of<X>(), CallOp::ONE, {}));
}

/// Mark that a variable requires gradient
template<typename... T>
    requires any_dsl_v<T...>
void requires_grad(T &&...args) noexcept {
    auto builder = detail::FunctionBuilder::current();
    auto do_requires_grad = [builder]<typename X>(X &&x) noexcept {
        builder->call(CallOp::REQUIRES_GRADIENT, {LUISA_EXPR(x)});
    };
    (do_requires_grad(std::forward<T>(args)), ...);
}

/// Mark that a variable does not require gradient
template<typename T>
    requires is_dsl_v<T>
auto detach(T &&x) noexcept {
    using X = expr_value_t<T>;
    return def<X>(detail::FunctionBuilder::current()->call(
        Type::of<X>(), CallOp::DETACH, {LUISA_EXPR(x)}));
}

/// Back-propagate gradient from the variable with the given gradient
template<typename T, typename G>
    requires is_dsl_v<T> && is_dsl_v<G> && is_same_expr_v<T, G>
void backward(T &&x, G &&grad) noexcept {
    auto b = detail::FunctionBuilder::current();
    auto expr_x = LUISA_EXPR(x);
    b->call(CallOp::GRADIENT_MARKER, {expr_x, LUISA_EXPR(grad)});
    b->call(CallOp::BACKWARD, {expr_x});
}

inline void discard() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RASTER_DISCARD, {});
}

/// Back-propagate gradient from the variable
template<typename T>
    requires is_dsl_v<T>
void backward(T &&x) noexcept {
    backward(std::forward<T>(x), dsl::one<T>());
}

/// Get the back-propagated gradient of the variable
template<typename T>
    requires is_dsl_v<T>
[[nodiscard]] inline auto grad(T &&x) noexcept {
    return def<expr_value_t<T>>(
        detail::FunctionBuilder::current()->call(
            Type::of<expr_value_t<T>>(), CallOp::GRADIENT,
            {LUISA_EXPR(x)}));
}

// barriers
/// Synchronize thread block.
inline void sync_block() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::SYNCHRONIZE_BLOCK, {});
}

// warp intrinsics
[[nodiscard]] inline auto warp_is_first_active_lane() noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::WARP_IS_FIRST_ACTIVE_LANE,
            {}));
}

[[nodiscard]] inline auto warp_first_active_lane() noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::WARP_FIRST_ACTIVE_LANE,
            {}));
}

template<typename X>
    requires is_scalar_expr_v<X> || is_vector_expr_v<X>
[[nodiscard]] inline auto warp_active_all_equal(X &&value) noexcept {
    using T = expr_value_t<X>;
    if constexpr (vector_dimension_v<T> == 1) {
        return def<bool>(
            detail::FunctionBuilder::current()->call(
                Type::of<bool>(), CallOp::WARP_ACTIVE_ALL_EQUAL,
                {LUISA_EXPR(value)}));
    } else {
        using Ret = Vector<bool, vector_dimension_v<T>>;
        return def<Ret>(
            detail::FunctionBuilder::current()->call(
                Type::of<Ret>(), CallOp::WARP_ACTIVE_ALL_EQUAL,
                {LUISA_EXPR(value)}));
    }
}

template<typename X>
    requires is_int_or_vector_expr_v<X> ||
             is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto warp_active_bit_and(X &&value) noexcept {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_BIT_AND,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires is_int_or_vector_expr_v<X> ||
             is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto warp_active_bit_or(X &&value) noexcept {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_BIT_OR,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires is_int_or_vector_expr_v<X> ||
             is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto warp_active_bit_xor(X &&value) noexcept {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_BIT_XOR,
            {LUISA_EXPR(value)}));
}

[[nodiscard]] inline auto warp_active_count_bits(Expr<bool> value) noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::WARP_ACTIVE_COUNT_BITS,
            {LUISA_EXPR(value)}));
}

[[nodiscard]] inline auto warp_prefix_count_bits(Expr<bool> value) noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::WARP_PREFIX_COUNT_BITS,
            {LUISA_EXPR(value)}));
}

[[nodiscard]] inline auto warp_active_all(Expr<bool> value) noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::WARP_ACTIVE_ALL,
            {LUISA_EXPR(value)}));
}

[[nodiscard]] inline auto warp_active_any(Expr<bool> value) noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::WARP_ACTIVE_ANY,
            {LUISA_EXPR(value)}));
}

[[nodiscard]] inline auto warp_active_bit_mask(Expr<bool> value) noexcept {
    return def<uint4>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint4>(), CallOp::WARP_ACTIVE_BIT_MASK,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X>) &&
            (!is_bool_or_vector_expr_v<X>)
[[nodiscard]] inline auto warp_active_min(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_MIN,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X>) &&
            (!is_bool_or_vector_expr_v<X>)
[[nodiscard]] inline auto warp_active_max(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_MAX,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X>) &&
            (!is_bool_or_vector_expr_v<X>)
[[nodiscard]] inline auto warp_active_product(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_PRODUCT,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X>) &&
            (!is_bool_or_vector_expr_v<X>)
[[nodiscard]] inline auto warp_active_sum(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_ACTIVE_SUM,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X>) &&
            (!is_bool_or_vector_expr_v<X>)
[[nodiscard]] inline auto warp_prefix_product(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_PREFIX_PRODUCT,
            {LUISA_EXPR(value)}));
}

template<typename X>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X>) &&
            (!is_bool_or_vector_expr_v<X>)
[[nodiscard]] inline auto warp_prefix_sum(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_PREFIX_SUM,
            {LUISA_EXPR(value)}));
}

template<typename X, typename Y>
    requires(is_scalar_expr_v<X> || is_vector_expr_v<X> || is_matrix_expr_v<X>) &&
            is_integral_expr_v<Y>
[[nodiscard]] inline auto warp_read_lane(X &&value, Y &&index) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_READ_LANE,
            {LUISA_EXPR(value), LUISA_EXPR(index)}));
}

template<typename X>
    requires is_scalar_expr_v<X> || is_vector_expr_v<X> || is_matrix_expr_v<X>
[[nodiscard]] inline auto warp_read_first_active_lane(X &&value) {
    using T = expr_value_t<X>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::WARP_READ_FIRST_ACTIVE_LANE,
            {LUISA_EXPR(value)}));
}

// shader execution reordering
inline void reorder_shader_execution(Expr<uint> hint, Expr<uint> hint_bits) noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::SHADER_EXECUTION_REORDER,
        {LUISA_EXPR(hint), LUISA_EXPR(hint_bits)});
}

inline void reorder_shader_execution() noexcept {
    reorder_shader_execution(0u, 0u);
}

#undef LUISA_EXPR

}// namespace dsl

}// namespace luisa::compute
