//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute::dsl {

template<typename T>
struct Var : public detail::Expr<T> {

    template<typename... Args>
    requires concepts::Constructible<T, detail::expr_value_t<Args>...>
    Var(Args &&...args) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->local(
            Type::of<T>(),
            {detail::extract_expression(std::forward<Args>(args))...})} {}

    Var(Var &&) noexcept = default;
    Var(const Var &another) noexcept : Var{detail::Expr{another}} {}
    void operator=(Var &&rhs) const noexcept { detail::ExprBase<T>::operator=(rhs); }
    void operator=(const Var &rhs) const noexcept { detail::ExprBase<T>::operator=(rhs); }
};

template<typename T>
Var(detail::Expr<T>) -> Var<T>;

template<concepts::Native T>
Var(T) -> Var<T>;

// TODO: support buffer etc.

}// namespace luisa::compute::dsl
