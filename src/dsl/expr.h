//
// Created by Mike Smith on 2020/12/2.
//
#pragma once

#include <array>
#include <string_view>

#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/buffer.h>
#include <runtime/bindless_array.h>
#include <ast/function_builder.h>
#include <dsl/expr_traits.h>
#include <dsl/arg.h>

namespace luisa::compute {

inline namespace dsl {

template<typename T>
[[nodiscard]] auto def(const Expression *expr) noexcept -> Var<expr_value_t<T>>;

template<typename T>
[[nodiscard]] auto def(T &&x) noexcept -> Var<expr_value_t<T>>;

}// namespace dsl

namespace detail {

/// Extract or construct expression from given data
template<typename T>
[[nodiscard]] inline auto extract_expression(T &&v) noexcept {
    if constexpr (is_dsl_v<T>) {
        return std::forward<T>(v).expression();
    } else {
        static_assert(concepts::basic<T>);
        return FunctionBuilder::current()->literal(Type::of<T>(), std::forward<T>(v));
    }
}

}// namespace detail

/// Expr common definition
#define LUISA_EXPR_COMMON(...)                                                   \
private:                                                                         \
    const Expression *_expression;                                               \
                                                                                 \
public:                                                                          \
    explicit Expr(const Expression *expr) noexcept : _expression{expr} {}        \
    [[nodiscard]] auto expression() const noexcept { return this->_expression; } \
    Expr(Expr &&another) noexcept = default;                                     \
    Expr(const Expr &another) noexcept = default;                                \
    Expr &operator=(Expr) noexcept = delete;

/// Construct Expr from literal value
#define LUISA_EXPR_FROM_LITERAL(...)                                             \
    template<typename U>                                                         \
        requires std::same_as<U, __VA_ARGS__>                                    \
    Expr(U literal) noexcept : Expr{detail::FunctionBuilder::current()->literal( \
                                   Type::of<U>(), literal)} {}

namespace detail {

template<typename T>
struct ExprEnableStaticCast;

template<typename T>
struct ExprEnableBitwiseCast;

template<typename T>
struct ExprEnableSubscriptAccess;

template<typename T>
struct ExprEnableGetMemberByIndex;

template<typename... T, size_t... i>
[[nodiscard]] inline auto compose_impl(std::tuple<T...> t, std::index_sequence<i...>) noexcept {
    return Var<std::tuple<expr_value_t<T>...>>{Expr{std::get<i>(t)}...};
}

template<typename T, size_t... i>
[[nodiscard]] inline auto decompose_impl(Expr<T> x, std::index_sequence<i...>) noexcept {
    return std::make_tuple(x.template get<i>()...);
}

}// namespace detail

inline namespace dsl {

/// Compose tuple of values to var of tuple
template<typename... T>
[[nodiscard]] inline auto compose(std::tuple<T...> t) noexcept {
    return detail::compose_impl(t, std::index_sequence_for<T...>{});
}

/// Do nothing
template<typename T>
[[nodiscard]] inline auto compose(T &&v) noexcept {
    return std::forward<T>(v);
}

/// Compose values to var of tuple
template<typename... T>
[[nodiscard]] inline auto compose(T &&...v) noexcept {
    return compose(std::make_tuple(Expr{std::forward<T>(v)}...));
}

/// Decompose var of tuple to tuple
template<typename T>
[[nodiscard]] inline auto decompose(T &&t) noexcept {
    using member_tuple = struct_member_tuple_t<expr_value_t<T>>;
    using index_sequence = std::make_index_sequence<std::tuple_size_v<member_tuple>>;
    return detail::decompose_impl(Expr{std::forward<T>(t)}, index_sequence{});
}

}// namespace dsl

/**
 * @brief Class of Expr<T>.
 * 
 * Member function's tparam must be 0.
 * 
 * @tparam T needs to be scalar 
 */
template<typename T>
struct Expr
    : detail::ExprEnableStaticCast<Expr<T>>,
      detail::ExprEnableBitwiseCast<Expr<T>> {
    static_assert(is_scalar_v<T> || is_custom_struct_v<T>);
    LUISA_EXPR_COMMON(T)
    LUISA_EXPR_FROM_LITERAL(T)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i == 0u);
        return *this;
    }
};

/// Class of Expr<std::array<T, N>>
template<typename T, size_t N>
struct Expr<std::array<T, N>>
    : detail::ExprEnableSubscriptAccess<Expr<std::array<T, N>>>,
      detail::ExprEnableGetMemberByIndex<Expr<std::array<T, N>>> {
    LUISA_EXPR_COMMON(std::array<T, N>)
};

/// Class of Expr<T[N]>
template<typename T, size_t N>
struct Expr<T[N]>
    : detail::ExprEnableSubscriptAccess<Expr<T[N]>>,
      detail::ExprEnableGetMemberByIndex<Expr<T[N]>> {
    LUISA_EXPR_COMMON(T[N])
};

/// Class of Expr<Matrix><N>>. Can be constructed from Matrix<N>
template<size_t N>
struct Expr<Matrix<N>>
    : detail::ExprEnableSubscriptAccess<Expr<Matrix<N>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Matrix<N>>> {
    LUISA_EXPR_COMMON(Matrix<N>)
    LUISA_EXPR_FROM_LITERAL(Matrix<N>)
};

/// Class of Expr<std::tuple<T...>>.
/// get<i>() will return Expr of the ith member of tuple
template<typename... T>
struct Expr<std::tuple<T...>> {
    LUISA_EXPR_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Expr<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    };
};

/// Class of Expr<Vector<T, 2>>. Can be constructed from Vector<T, 2>.
template<typename T>
struct Expr<Vector<T, 2>>
    : detail::ExprEnableStaticCast<Expr<Vector<T, 2>>>,
      detail::ExprEnableBitwiseCast<Expr<Vector<T, 2>>>,
      detail::ExprEnableSubscriptAccess<Expr<Vector<T, 2>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Vector<T, 2>>> {
    LUISA_EXPR_COMMON(Vector<T, 2>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 2>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

/// Class of Expr<Vector<T, 3>>. Can be constructed from Vector<T, 3>.
template<typename T>
struct Expr<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Expr<Vector<T, 3>>>,
      detail::ExprEnableBitwiseCast<Expr<Vector<T, 3>>>,
      detail::ExprEnableSubscriptAccess<Expr<Vector<T, 3>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Vector<T, 3>>> {
    LUISA_EXPR_COMMON(Vector<T, 3>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 3>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

/// Class of Expr<Vector<T, 4>>. Can be constructed from Vector<T, 4>.
template<typename T>
struct Expr<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Expr<Vector<T, 4>>>,
      detail::ExprEnableBitwiseCast<Expr<Vector<T, 4>>>,
      detail::ExprEnableSubscriptAccess<Expr<Vector<T, 4>>>,
      detail::ExprEnableGetMemberByIndex<Expr<Vector<T, 4>>> {
    LUISA_EXPR_COMMON(Vector<T, 4>)
    LUISA_EXPR_FROM_LITERAL(Vector<T, 4>)
    Expr<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Expr<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Expr<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Expr<T> w{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

#undef LUISA_EXPR_COMMON
#undef LUISA_EXPR_FROM_LITERAL

namespace detail {

/// Enable static cast to type Dest
template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
        requires concepts::static_convertible<
            expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using TrueDest = expr_value_t<Dest>;
        return def<TrueDest>(
            FunctionBuilder::current()->cast(
                Type::of<TrueDest>(),
                CastOp::STATIC,
                src.expression()));
    }
};

/// Enable bitwise cast to type Dest
template<typename T>
struct ExprEnableBitwiseCast {
    template<typename Dest>
        requires concepts::bitwise_convertible<
            expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto as() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using TrueDest = expr_value_t<Dest>;
        return def<TrueDest>(
            FunctionBuilder::current()->cast(
                Type::of<TrueDest>(),
                CastOp::BITWISE,
                src.expression()));
    }
};

/// Enable subscript access
template<typename T>
struct ExprEnableSubscriptAccess {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&index) const noexcept {
        auto self = def<T>(static_cast<const T *>(this)->expression());
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Elem>(FunctionBuilder::current()->access(
            Type::of<Elem>(), self.expression(),
            extract_expression(std::forward<I>(index))));
    }
};

/// Enable get member by index
template<typename T>
struct ExprEnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        return static_cast<const T *>(this)->operator[](static_cast<uint>(i));
    }
};

}// namespace detail

// deduction guides
template<typename T>
Expr(Expr<T>) -> Expr<T>;

template<typename T>
Expr(const Var<T> &) -> Expr<T>;

template<typename T>
Expr(detail::Ref<T>) -> Expr<T>;

template<concepts::basic T>
Expr(T) -> Expr<T>;

#define LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(Class) \
    Class() noexcept = delete;                         \
    Class(Class &&) noexcept = delete;                 \
    Class(const Class &) noexcept = delete;            \
    Class &operator=(Class &&) noexcept = delete;      \
    Class &operator=(const Class &) noexcept = delete; \
    ~Class() noexcept = delete;

}// namespace luisa::compute
