//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>
#include <dsl/arg.h>

namespace luisa::compute {

template<typename T>
class Var : public detail::Expr<T> {

    static_assert(std::is_trivially_destructible_v<T>);

public:
    // for local variables
    template<typename... Args>
    requires concepts::constructible<T, detail::expr_value_t<Args>...>
    Var(Args &&...args) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->local(
            Type::of<T>(),
            {detail::extract_expression(std::forward<Args>(args))...})} {}

    // for internal use only...
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->argument(Type::of<T>())} {}

    Var(Var &&) noexcept = default;
    Var(const Var &another) noexcept : Var{detail::Expr{another}} {}
    void operator=(Var &&rhs) noexcept { detail::ExprBase<T>::operator=(rhs); }
    void operator=(const Var &rhs) noexcept { detail::ExprBase<T>::operator=(rhs); }
};

template<typename T>
Var(detail::Expr<T>) -> Var<T>;

template<typename T>
Var(T &&) -> Var<T>;

template<typename T, size_t N>
using VarArray = Var<std::array<T, N>>;

}// namespace luisa::compute
