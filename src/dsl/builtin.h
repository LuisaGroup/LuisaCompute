//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#include <core/constants.h>
#include <dsl/var.h>
#include <dsl/operators.h>

namespace luisa::compute {

inline namespace dsl {

template<typename Dest, typename Src>
[[nodiscard]] inline auto cast(Src &&s) noexcept {
    Expr expr{std::forward<Src>(s)};
    return expr.template cast<Dest>();
}

template<typename Dest, typename Src>
[[nodiscard]] inline auto as(Src &&s) noexcept {
    Expr expr{std::forward<Src>(s)};
    return expr.template as<Dest>();
}

inline void assume(Expr<bool> pred) noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::ASSUME, {pred.expression()});
}

inline void unreachable() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::ASSUME, {});
}

[[nodiscard]] inline auto thread_id() noexcept {
    return def<uint3>(detail::FunctionBuilder::current()->thread_id());
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
    return def<uint3>(detail::FunctionBuilder::current()->block_id());
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
    return def<uint3>(detail::FunctionBuilder::current()->dispatch_id());
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
    return def<uint3>(detail::FunctionBuilder::current()->dispatch_size());
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

}// namespace dsl

namespace detail {

template<size_t i, size_t n, typename I, typename S, typename B>
inline void soa_read_impl(I index, S s, B buffers) noexcept {
    if constexpr (i < n) {
        s.template get<i>() = std::get<i>(buffers).read(index);
        soa_read_impl<i + 1u, n>(index, s, buffers);
    }
}

template<size_t i, size_t n, typename I, typename S, typename B>
inline void soa_write_impl(I index, S s, B buffers) noexcept {
    if constexpr (i < n) {
        std::get<i>(buffers).write(index, s.template get<i>());
        soa_write_impl<i + 1u, n>(index, s, buffers);
    }
}

}// namespace detail

inline namespace dsl {

template<typename T, typename... Args>
    requires std::negation_v<std::disjunction<std::is_pointer<std::remove_cvref_t<Args>>...>>
[[nodiscard]] inline auto def(Args &&...args) noexcept {
    return Var<expr_value_t<T>>{std::forward<Args>(args)...};
}

template<typename T>
[[nodiscard]] inline auto def(T &&x) noexcept -> Var<expr_value_t<T>> {
    return Var{Expr{std::forward<T>(x)}};
}

template<typename T>
[[nodiscard]] inline auto def(const Expression *expr) noexcept -> Var<expr_value_t<T>> {
    return Var{Expr<expr_value_t<T>>{expr}};
}

template<typename T, typename SExpr>
[[nodiscard]] inline auto eval(SExpr &&s_expr) noexcept {
    static_assert(is_basic_v<T>, "only basic types are allowed in meta-values");
    auto type = Type::of<T>();
    auto meta_value = LiteralExpr::MetaValue{
        type, luisa::string{std::forward<SExpr>(s_expr)}};
    auto expr = detail::FunctionBuilder::current()->literal(
        type, std::move(meta_value));
    return Var{Expr<T>{expr}};
}

template<typename S, typename Index, typename... Buffers>
    requires concepts::integral<expr_value_t<Index>> && std::conjunction_v<is_buffer_expr<Buffers>...>
[[nodiscard]] inline auto soa_read(Index &&index, Buffers &&...buffers) noexcept {
    Var i = std::forward<Index>(index);
    return Var<S>{std::forward<Buffers>(buffers).read(i)...};
}

template<typename Index, typename... Buffers>
    requires concepts::integral<expr_value_t<Index>> && std::conjunction_v<is_buffer_expr<Buffers>...>
[[nodiscard]] inline auto soa_read(Index &&index, Buffers &&...buffers) noexcept {
    Var i = std::forward<Index>(index);
    return compose(std::forward<Buffers>(buffers).read(i)...);
}

template<typename S, typename Index, typename... Buffers>
    requires concepts::integral<expr_value_t<Index>> && std::conjunction_v<is_buffer_expr<Buffers>...>
inline void soa_write(Index &&index, S &&s, Buffers &&...bs) noexcept {
    Var i = std::forward<Index>(index);
    Var v = std::forward<S>(s);
    detail::soa_write_impl<0u, sizeof...(bs)>(
        i, v, std::make_tuple(Expr{std::forward<Buffers>(bs)}...));
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
    requires is_dsl_v<Ts> && is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector2(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 2>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(s)}));
}

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

template<typename T, typename Tv>
    requires is_dsl_v<Tv> && is_vector_expr_v<Tv>
[[nodiscard]] inline auto make_vector2(Tv &&v) noexcept {
    using V = Vector<T, 2>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(v)}));
}

template<typename Ts>
    requires is_dsl_v<Ts> && is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector3(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 3>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(),
            {LUISA_EXPR(s)}));
}

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

template<typename Ts>
    requires is_dsl_v<Ts> && is_scalar_expr_v<Ts>
[[nodiscard]] inline auto make_vector4(Ts &&s) noexcept {
    using V = Vector<expr_value_t<Ts>, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(s)}));
}

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

template<typename T, typename Tv>
    requires is_dsl_v<Tv> && is_vector4_expr_v<Tv>
[[nodiscard]] inline auto make_vector4(Tv &&v) noexcept {
    using V = Vector<T, 4>;
    return def<V>(
        FunctionBuilder::current()->call(
            Type::of<V>(), make_vector_tag<V>(), {LUISA_EXPR(v)}));
}

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

// vectorized call
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

#define LUISA_MAKE_VECTOR(type)                                  \
    template<typename S>                                         \
        requires is_dsl_v<S> && is_same_expr_v<S, type>          \
    [[nodiscard]] inline auto make_##type##2(S && s) noexcept {  \
        return detail::make_vector2(std::forward<S>(s));         \
    }                                                            \
    template<typename X, typename Y>                             \
        requires any_dsl_v<X, Y> &&                              \
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<Y, type>                              \
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
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<Y, type> &&                           \
            is_same_expr_v<Z, type>                              \
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
            is_same_expr_v<V, type##2> &&                        \
            is_same_expr_v<Z, type>                              \
    [[nodiscard]] inline auto make_##type##3(                    \
        V && v, Z && z) noexcept {                               \
        return detail::make_vector3(                             \
            std::forward<V>(v),                                  \
            std::forward<Z>(z));                                 \
    }                                                            \
    template<typename X, typename V>                             \
        requires any_dsl_v<X, V> &&                              \
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<V, type##2>                           \
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
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<Y, type> &&                           \
            is_same_expr_v<Z, type> &&                           \
            is_same_expr_v<W, type>                              \
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
            is_same_expr_v<V, type##2> &&                        \
            is_same_expr_v<Z, type> &&                           \
            is_same_expr_v<W, type>                              \
    [[nodiscard]] inline auto make_##type##4(                    \
        V && v, Z && z, W && w) noexcept {                       \
        return detail::make_vector4(                             \
            std::forward<V>(v),                                  \
            std::forward<Z>(z),                                  \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename X, typename YZ, typename W>                \
        requires any_dsl_v<X, YZ, W> &&                          \
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<YZ, type##2> &&                       \
            is_same_expr_v<W, type>                              \
    [[nodiscard]] inline auto make_##type##4(                    \
        X && x, YZ && yz, W && w) noexcept {                     \
        return detail::make_vector4(                             \
            std::forward<X>(x),                                  \
            std::forward<YZ>(yz),                                \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename X, typename Y, typename ZW>                \
        requires any_dsl_v<X, Y, ZW> &&                          \
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<Y, type> &&                           \
            is_same_expr_v<ZW, type##2>                          \
    [[nodiscard]] inline auto make_##type##4(                    \
        X && x, Y && y, ZW && zw) noexcept {                     \
        return detail::make_vector4(                             \
            std::forward<X>(x),                                  \
            std::forward<Y>(y),                                  \
            std::forward<ZW>(zw));                               \
    }                                                            \
    template<typename XY, typename ZW>                           \
        requires any_dsl_v<XY, ZW> &&                            \
            is_same_expr_v<XY, type##2> &&                       \
            is_same_expr_v<ZW, type##2>                          \
    [[nodiscard]] inline auto make_##type##4(                    \
        XY && xy, ZW && zw) noexcept {                           \
        return detail::make_vector4(                             \
            std::forward<XY>(xy),                                \
            std::forward<ZW>(zw));                               \
    }                                                            \
    template<typename XYZ, typename W>                           \
        requires any_dsl_v<XYZ, W> &&                            \
            is_same_expr_v<XYZ, type##3> &&                      \
            is_same_expr_v<W, type>                              \
    [[nodiscard]] inline auto make_##type##4(                    \
        XYZ && xyz, W && w) noexcept {                           \
        return detail::make_vector4(                             \
            std::forward<XYZ>(xyz),                              \
            std::forward<W>(w));                                 \
    }                                                            \
    template<typename X, typename YZW>                           \
        requires any_dsl_v<X, YZW> &&                            \
            is_same_expr_v<X, type> &&                           \
            is_same_expr_v<YZW, type##3>                         \
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
#undef LUISA_MAKE_VECTOR

// make float2x2
template<typename S>
    requires is_dsl_v<S> && is_floating_point_expr_v<S>
[[nodiscard]] inline auto make_float2x2(S &&s) noexcept {
    return def<float2x2>(
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {LUISA_EXPR(s)}));
}

template<typename M00, typename M01, typename M10, typename M11>
    requires any_dsl_v<M00, M01, M10, M11> &&
        is_floating_point_expr_v<M00> &&
        is_floating_point_expr_v<M01> &&
        is_floating_point_expr_v<M10> &&
        is_floating_point_expr_v<M11>
[[nodiscard]] inline auto make_float2x2(
    M00 &&m00, M01 &&m01,
    M10 &&m10, M11 &&m11) noexcept {
    return def<float2x2>(
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {LUISA_EXPR(m00), LUISA_EXPR(m01),
             LUISA_EXPR(m10), LUISA_EXPR(m11)}));
}

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

template<typename M>
    requires is_dsl_v<M> && is_matrix_expr_v<M>
[[nodiscard]] inline auto make_float2x2(M &&m) noexcept {
    return def<float2x2>(
        detail::FunctionBuilder::current()->call(
            Type::of<float2x2>(), CallOp::MAKE_FLOAT2X2,
            {LUISA_EXPR(m)}));
}

// make float3x3
template<typename S>
    requires is_dsl_v<S> && is_same_expr_v<S, float>
[[nodiscard]] inline auto make_float3x3(S &&s) noexcept {
    return def<float3x3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {LUISA_EXPR(s)}));
}

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
    return def<float3x3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {LUISA_EXPR(m00), LUISA_EXPR(m01), LUISA_EXPR(m02),
             LUISA_EXPR(m10), LUISA_EXPR(m11), LUISA_EXPR(m12),
             LUISA_EXPR(m20), LUISA_EXPR(m21), LUISA_EXPR(m22)}));
}

template<typename M>
    requires is_dsl_v<M> && is_matrix_expr_v<M>
[[nodiscard]] inline auto make_float3x3(M &&m) noexcept {
    return def<float3x3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3x3>(), CallOp::MAKE_FLOAT3X3,
            {LUISA_EXPR(m)}));
}

// make float4x4
template<typename S>
    requires is_dsl_v<S> && is_same_expr_v<S, float>
[[nodiscard]] inline auto make_float4x4(S &&s) noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {LUISA_EXPR(s)}));
}

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
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {LUISA_EXPR(m00), LUISA_EXPR(m01), LUISA_EXPR(m02), LUISA_EXPR(m03),
             LUISA_EXPR(m10), LUISA_EXPR(m11), LUISA_EXPR(m12), LUISA_EXPR(m13),
             LUISA_EXPR(m20), LUISA_EXPR(m21), LUISA_EXPR(m22), LUISA_EXPR(m23),
             LUISA_EXPR(m30), LUISA_EXPR(m31), LUISA_EXPR(m32), LUISA_EXPR(m33)}));
}

template<typename M>
    requires is_dsl_v<M> && is_matrix_expr_v<M>
[[nodiscard]] inline auto make_float4x4(M &&m) noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::MAKE_FLOAT4X4,
            {LUISA_EXPR(m)}));
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto all(Tx &&x) noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ALL,
            {LUISA_EXPR(x)}));
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto any(Tx &&x) noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::ANY,
            {LUISA_EXPR(x)}));
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_bool_vector_expr_v<Tx>
[[nodiscard]] inline auto none(Tx &&x) noexcept {
    return !any(std::forward<Tx>(x));
}

// Tf: scalar or vector, Tt: scalar or vector, Tp: bool scalar or vector, vectorize
template<typename Tf, typename Tt, typename Tp>
    requires any_dsl_v<Tf, Tt, Tp> &&
        is_bool_or_vector_expr_v<Tp> &&
        std::disjunction_v<
            is_scalar_expr<Tf>,
            is_scalar_expr<Tt>,
            is_vector_expr<Tf>,
            is_vector_expr<Tt>> &&
        is_vector_expr_same_element_v<Tf, Tt>
[[nodiscard]] inline auto select(Tf &&f, Tt &&t, Tp &&p) noexcept {
    return detail::make_vector_call<vector_expr_element_t<Tf>>(
        CallOp::SELECT,
        std::forward<Tf>(f),
        std::forward<Tt>(t),
        std::forward<Tp>(p));
}

// Tf == Tt: non-scalar and non-vector, Tp: bool scalar, no vectorization
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

template<typename Tp, typename Tt, typename Tf>
    requires any_dsl_v<Tp, Tt, Tf>
[[nodiscard]] inline auto ite(Tp &&p, Tt &&t, Tf &&f) noexcept {
    return select(std::forward<Tf>(f),
                  std::forward<Tt>(t),
                  std::forward<Tp>(p));
}

template<typename Tv, typename Tl, typename Tu>
    requires any_dsl_v<Tv, Tl, Tu> && is_vector_expr_same_element_v<Tv, Tl, Tu>
[[nodiscard]] inline auto clamp(Tv &&v, Tl &&l, Tu &&u) noexcept {
    return detail::make_vector_call<vector_expr_element_t<Tv>>(
        CallOp::CLAMP,
        std::forward<Tv>(v),
        std::forward<Tl>(l),
        std::forward<Tu>(u));
}

template<typename Ta, typename Tb, typename Tt>
    requires any_dsl_v<Ta, Tb, Tt> && is_float_or_vector_expr_v<Ta> && is_vector_expr_same_element_v<Ta, Tb, Tt>
[[nodiscard]] inline auto lerp(Ta &&a, Tb &&b, Tt &&t) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LERP,
        std::forward<Ta>(a),
        std::forward<Tb>(b),
        std::forward<Tt>(t));
}

template<typename X, typename Y, typename Z>
    requires any_dsl_v<X, Y, Z> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y> && is_float_or_vector_expr_v<Z>
[[nodiscard]] inline auto fma(X &&x, Y &&y, Z &&z) noexcept {
    return detail::make_vector_call<float>(
        CallOp::FMA,
        std::forward<X>(x),
        std::forward<Y>(y),
        std::forward<Z>(z));
}

template<typename Tv>
    requires is_dsl_v<Tv> && is_float_or_vector_expr_v<Tv>
[[nodiscard]] inline auto saturate(Tv &&v) noexcept {
    return clamp(std::forward<Tv>(v), 0.0f, 1.0f);
}

template<typename E, typename X>
    requires any_dsl_v<E, X> && is_float_or_vector_expr_v<E> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto step(E &&edge, X &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::STEP,
        std::forward<E>(edge),
        std::forward<X>(x));
}

template<typename L, typename R, typename T>
    requires any_dsl_v<L, R, T> && is_float_or_vector_expr_v<L> && is_float_or_vector_expr_v<R> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto smoothstep(L &&left, R &&right, T &&x) noexcept {
    auto edge0 = def(std::forward<L>(left));
    auto edge1 = def(std::forward<R>(right));
    auto t = saturate((std::forward<T>(x) - edge0) / (edge1 - edge0));
    return t * t * fma(t, -2.0f, 3.0f);
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_float_or_vector_expr_v<Tx>
[[nodiscard]] inline auto abs(Tx &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ABS, std::forward<Tx>(x));
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_int_or_vector_expr_v<Tx>
[[nodiscard]] inline auto abs(Tx &&x) noexcept {
    return detail::make_vector_call<int>(
        CallOp::ABS, std::forward<Tx>(x));
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y>
[[nodiscard]] inline auto mod(X &&x_in, Y &&y_in) noexcept {
    auto x = def(std::forward<X>(x_in));
    auto y = def(std::forward<Y>(y_in));
    return x - y * floor(x / y);
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y>
[[nodiscard]] inline auto fmod(X &&x_in, Y &&y_in) noexcept {
    auto x = def(std::forward<X>(x_in));
    auto y = def(std::forward<Y>(y_in));
    return x - y * trunc(x / y);
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_vector_expr_same_element_v<X, Y>
[[nodiscard]] inline auto min(X &&x, Y &&y) noexcept {
    return detail::make_vector_call<vector_expr_element_t<X>>(
        CallOp::MIN,
        std::forward<X>(x),
        std::forward<Y>(y));
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_vector_expr_same_element_v<X, Y>
[[nodiscard]] inline auto max(X &&x, Y &&y) noexcept {
    return detail::make_vector_call<vector_expr_element_t<X>>(
        CallOp::MAX,
        std::forward<X>(x),
        std::forward<Y>(y));
}

template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto clz(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::CLZ, std::forward<X>(x));
}

template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto ctz(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::CTZ, std::forward<X>(x));
}

template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto popcount(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::POPCOUNT, std::forward<X>(x));
}

template<typename X>
    requires is_dsl_v<X> && is_uint_or_vector_expr_v<X>
[[nodiscard]] inline auto reverse(X &&x) noexcept {
    return detail::make_vector_call<uint>(
        CallOp::REVERSE, std::forward<X>(x));
}

template<typename X>
    requires is_dsl_v<X> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto isinf(X &&x) noexcept {
    return detail::make_vector_call<bool>(
        CallOp::ISINF, std::forward<X>(x));
}

template<typename X>
    requires is_dsl_v<X> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto isnan(X &&x) noexcept {
    return detail::make_vector_call<bool>(
        CallOp::ISNAN, std::forward<X>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto acos(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ACOS, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto acosh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ACOSH, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto asin(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ASIN, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto asinh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ASINH, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto atan(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ATAN, std::forward<T>(x));
}

template<typename Y, typename X>
    requires any_dsl_v<Y, X> && is_float_or_vector_expr_v<Y> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto atan2(Y &&y, X &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ATAN2,
        std::forward<Y>(y),
        std::forward<X>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto atanh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ATANH, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto cos(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::COS, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto cosh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::COSH, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto sin(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SIN, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto sinh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SINH, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto tan(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::TAN, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto tanh(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::TANH, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto exp(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::EXP, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto exp2(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::EXP2, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto exp10(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::EXP10, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto log(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LOG, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto log2(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LOG2, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto log10(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::LOG10, std::forward<T>(x));
}

template<typename X, typename A>
    requires any_dsl_v<X, A> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<A>
[[nodiscard]] inline auto pow(X &&x, A &&a) noexcept {
    return detail::make_vector_call<float>(
        CallOp::POW,
        std::forward<X>(x),
        std::forward<A>(a));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto sqrt(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::SQRT, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto rsqrt(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::RSQRT, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto ceil(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::CEIL, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto floor(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::FLOOR, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto fract(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::FRACT, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto trunc(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::TRUNC, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto round(T &&x) noexcept {
    return detail::make_vector_call<float>(
        CallOp::ROUND, std::forward<T>(x));
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto degrees(T &&x) noexcept {
    return std::forward<T>(x) * (180.0f * constants::inv_pi);
}

template<typename T>
    requires is_dsl_v<T> && is_float_or_vector_expr_v<T>
[[nodiscard]] inline auto radians(T &&x) noexcept {
    return std::forward<T>(x) * (constants::pi / 180.0f);
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_or_vector_expr_v<X> && is_float_or_vector_expr_v<Y>
[[nodiscard]] inline auto copysign(X &&x, Y &&y) noexcept {
    return detail::make_vector_call<float>(
        CallOp::COPYSIGN,
        std::forward<X>(x),
        std::forward<Y>(y));
}

template<typename X>
    requires is_dsl_v<X> && is_float_or_vector_expr_v<X>
[[nodiscard]] inline auto sign(X &&x) noexcept {
    return copysign(1.0f, std::forward<X>(x));
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_same_expr_v<X, Y, float3>
[[nodiscard]] inline auto cross(X &&x, Y &&y) noexcept {
    return def<float3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::CROSS,
            {LUISA_EXPR(x), LUISA_EXPR(y)}));
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_vector_expr_v<X> && is_float_vector_expr_v<Y> && is_vector_expr_same_dimension_v<X, Y>
[[nodiscard]] inline auto dot(X &&x, Y &&y) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DOT,
            {LUISA_EXPR(x), LUISA_EXPR(y)}));
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_float_vector_expr_v<Tx>
[[nodiscard]] inline auto length(Tx &&x) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH,
            {LUISA_EXPR(x)}));
}

template<typename Tx>
    requires is_dsl_v<Tx> && is_float_vector_expr_v<Tx>
[[nodiscard]] inline auto length_squared(Tx &&x) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::LENGTH_SQUARED,
            {LUISA_EXPR(x)}));
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_vector_expr_v<X> && is_float_vector_expr_v<Y> && is_vector_expr_same_dimension_v<X, Y>
[[nodiscard]] inline auto distance(X &&x, Y &&y) noexcept {
    return length(std::forward<X>(x) - std::forward<Y>(y));
}

template<typename X, typename Y>
    requires any_dsl_v<X, Y> && is_float_vector_expr_v<X> && is_float_vector_expr_v<Y> && is_vector_expr_same_dimension_v<X, Y>
[[nodiscard]] inline auto distance_squared(X &&x, Y &&y) noexcept {
    return length_squared(std::forward<X>(x) - std::forward<Y>(y));
}

template<typename T>
    requires is_dsl_v<T> && is_float_vector_expr_v<T>
[[nodiscard]] inline auto normalize(T &&x) noexcept {
    return detail::make_vector_call<float>(CallOp::NORMALIZE, std::forward<T>(x));
}

template<typename N, typename I, typename NRef>
    requires any_dsl_v<N, I, NRef> && is_same_expr_v<N, I, NRef, float3>
[[nodiscard]] inline auto faceforward(N &&n, I &&i, NRef &&n_ref) noexcept {
    return def<float3>(
        detail::FunctionBuilder::current()->call(
            Type::of<float3>(), CallOp::FACEFORWARD,
            {LUISA_EXPR(n), LUISA_EXPR(i), LUISA_EXPR(n_ref)}));
}

template<typename Tm>
    requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto determinant(Tm &&m) noexcept {
    return def<float>(
        detail::FunctionBuilder::current()->call(
            Type::of<float>(), CallOp::DETERMINANT,
            {LUISA_EXPR(m)}));
}

template<typename Tm>
    requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto transpose(Tm &&m) noexcept {
    using T = expr_value_t<Tm>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::TRANSPOSE,
            {LUISA_EXPR(m)}));
}

template<typename Tm>
    requires is_dsl_v<Tm> && is_matrix_expr_v<Tm>
[[nodiscard]] inline auto inverse(Tm &&m) noexcept {
    using T = expr_value_t<Tm>;
    return def<T>(
        detail::FunctionBuilder::current()->call(
            Type::of<T>(), CallOp::INVERSE,
            {LUISA_EXPR(m)}));
}

// barriers
inline void sync_block() noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::SYNCHRONIZE_BLOCK, {});
}

#undef LUISA_EXPR

}// namespace dsl
}// namespace luisa::compute
