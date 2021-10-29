//
// Created by Mike Smith on 2021/8/26.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

inline namespace dsl {
template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// defined in dsl/stmt.h
}

namespace detail {

template<typename T>
struct RefEnableSubscriptAccess {

public:
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto operator[](I &&index) const &noexcept {
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Elem>(FunctionBuilder::current()->access(
            Type::of<Elem>(),
            static_cast<const T *>(this)->expression(),
            extract_expression(std::forward<I>(index))));
    }
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto &operator[](I &&index) &noexcept {
        auto i = def(std::forward<I>(index));
        using Elem = std::remove_cvref_t<
            decltype(std::declval<expr_value_t<T>>()[0])>;
        auto f = FunctionBuilder::current();
        auto expr = f->access(
            Type::of<Elem>(),
            static_cast<const T *>(this)->expression(),
            i.expression());
        return *new_with_allocator<Var<Elem>>(expr);// TODO: store in function
    }
};

template<typename T>
struct RefEnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        auto self = const_cast<T *>(static_cast<const T *>(this));
        return Ref{self->operator[](static_cast<uint>(i))};
    }
};

#define LUISA_REF_COMMON(...)                                              \
private:                                                                   \
    const Expression *_expression;                                         \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(const Ref &) noexcept = default;                                   \
    template<typename Rhs>                                                 \
    void operator=(Rhs &&rhs) &noexcept {                                  \
        dsl::assign(*this, std::forward<Rhs>(rhs));                        \
    }                                                                      \
    [[nodiscard]] operator Expr<__VA_ARGS__>() const noexcept {            \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref rhs) &noexcept { (*this) = Expr<__VA_ARGS__>{rhs}; }

template<typename T>
struct Ref
    : detail::ExprEnableStaticCast<Ref<T>>,
      detail::ExprEnableBitwiseCast<Ref<T>> {
    static_assert(concepts::scalar<T>);
    LUISA_REF_COMMON(T)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i == 0u);
        return *this;
    }
};

template<typename T, size_t N>
struct Ref<std::array<T, N>>
    : detail::RefEnableSubscriptAccess<Ref<std::array<T, N>>>,
      detail::RefEnableGetMemberByIndex<Ref<std::array<T, N>>> {
    LUISA_REF_COMMON(std::array<T, N>)
};

template<size_t N>
struct Ref<Matrix<N>>
    : detail::RefEnableSubscriptAccess<Ref<Matrix<N>>>,
      detail::RefEnableGetMemberByIndex<Ref<Matrix<N>>> {
    LUISA_REF_COMMON(Matrix<N>)
};

template<typename... T>
struct Ref<std::tuple<T...>> {
    LUISA_REF_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using M = std::tuple_element_t<i, std::tuple<T...>>;
        return Ref<M>{detail::FunctionBuilder::current()->member(
            Type::of<M>(), this->expression(), i)};
    }
};

template<typename T>
struct Ref<Vector<T, 2>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 2>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 2>>>,
      detail::RefEnableSubscriptAccess<Ref<Vector<T, 2>>>,
      detail::RefEnableGetMemberByIndex<Ref<Vector<T, 2>>> {
    LUISA_REF_COMMON(Vector<T, 2>)
    Var<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Var<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Ref<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 3>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 3>>>,
      detail::RefEnableSubscriptAccess<Ref<Vector<T, 3>>>,
      detail::RefEnableGetMemberByIndex<Ref<Vector<T, 3>>> {
    LUISA_REF_COMMON(Vector<T, 3>)
    Var<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Var<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Var<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Ref<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 4>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 4>>>,
      detail::RefEnableSubscriptAccess<Ref<Vector<T, 4>>>,
      detail::RefEnableGetMemberByIndex<Ref<Vector<T, 4>>> {
    LUISA_REF_COMMON(Vector<T, 4>)
    Var<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Var<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Var<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Var<T> w{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

#undef LUISA_REF_COMMON

}// namespace detail
}// namespace luisa::compute
