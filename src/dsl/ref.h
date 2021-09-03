//
// Created by Mike Smith on 2021/8/26.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

inline namespace dsl {
template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// defined in dsl/func.h
}

#define LUISA_REF_COMMON(...)                                              \
private:                                                                   \
    const Expression *_expression;                                         \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    explicit Ref(detail::ArgumentCreation) noexcept                        \
        : Ref{detail::FunctionBuilder::current()                           \
                  ->reference(Type::of<__VA_ARGS__>())} {}                 \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(const Ref &) noexcept = default;                                   \
    template<typename Rhs>                                                 \
    void operator=(Rhs &&rhs) noexcept {                                   \
        dsl::assign(*this, std::forward<Rhs>(rhs));                        \
    }                                                                      \
    [[nodiscard]] operator Expr<__VA_ARGS__>() const noexcept {            \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref rhs) noexcept { (*this) = Expr<__VA_ARGS__>{rhs}; } \
    template<typename Value>                                               \
    void assign(Value &&v) noexcept { (*this) = std::forward<Value>(v); }

template<typename T>
struct Ref
    : detail::ExprEnableStaticCast<Ref<T>>,
      detail::ExprEnableBitwiseCast<Ref<T>>,
      detail::ExprEnableArithmeticAssign<Ref<T>> {
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
    : detail::ExprEnableSubscriptAccess<Ref<std::array<T, N>>>,
      detail::ExprEnableArithmeticAssign<Ref<std::array<T, N>>>,
      detail::ExprEnableGetMemberByIndex<Ref<std::array<T, N>>> {
    LUISA_REF_COMMON(std::array<T, N>)
};

template<size_t N>
struct Ref<Matrix<N>>
    : detail::ExprEnableSubscriptAccess<Ref<Matrix<N>>>,
      detail::ExprEnableArithmeticAssign<Ref<Matrix<N>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Matrix<N>>> {
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
      detail::ExprEnableSubscriptAccess<Ref<Vector<T, 2>>>,
      detail::ExprEnableArithmeticAssign<Ref<Vector<T, 2>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Vector<T, 2>>> {
    LUISA_REF_COMMON(Vector<T, 2>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
#include <dsl/swizzle_2.inl.h>
};

template<typename T>
struct Ref<Vector<T, 3>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 3>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 3>>>,
      detail::ExprEnableSubscriptAccess<Ref<Vector<T, 3>>>,
      detail::ExprEnableArithmeticAssign<Ref<Vector<T, 3>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Vector<T, 3>>> {
    LUISA_REF_COMMON(Vector<T, 3>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Ref<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
#include <dsl/swizzle_3.inl.h>
};

template<typename T>
struct Ref<Vector<T, 4>>
    : detail::ExprEnableStaticCast<Ref<Vector<T, 4>>>,
      detail::ExprEnableBitwiseCast<Ref<Vector<T, 4>>>,
      detail::ExprEnableSubscriptAccess<Ref<Vector<T, 4>>>,
      detail::ExprEnableArithmeticAssign<Ref<Vector<T, 4>>>,
      detail::ExprEnableGetMemberByIndex<Ref<Vector<T, 4>>> {
    LUISA_REF_COMMON(Vector<T, 4>)
    Ref<T> x{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x0u)};
    Ref<T> y{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x1u)};
    Ref<T> z{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x2u)};
    Ref<T> w{detail::FunctionBuilder::current()->swizzle(Type::of<T>(), this->expression(), 1u, 0x3u)};
#include <dsl/swizzle_4.inl.h>
};

#undef LUISA_REF_COMMON

}// namespace luisa::compute