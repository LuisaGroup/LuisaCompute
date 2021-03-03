//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>
#include <dsl/argument.h>

namespace luisa::compute::dsl {

template<typename T>
class Var : public detail::Expr<T> {

    static_assert(std::conjunction_v<
                  std::is_trivially_copyable<T>,
                  std::is_trivially_destructible<T>>);

private:
    // for making function arguments...
    template<typename U>
    friend class Kernel;

    template<typename U>
    friend class Callable;

    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->uniform(Type::of<T>())} {}

public:
    // for local variables
    template<typename... Args>
    requires concepts::Constructible<T, detail::expr_value_t<Args>...>
    Var(Args &&...args) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->local(
            Type::of<T>(),
            {detail::extract_expression(std::forward<Args>(args))...})} {}

    Var(Var &&) noexcept = default;
    Var(const Var &another) noexcept : Var{detail::Expr{another}} {}
    void operator=(Var &&rhs) noexcept { detail::ExprBase<T>::operator=(rhs); }
    void operator=(const Var &rhs) noexcept { detail::ExprBase<T>::operator=(rhs); }
};

template<typename T>
Var(detail::Expr<T>) -> Var<T>;

template<typename T>
Var(T &&) -> Var<T>;

}// namespace luisa::compute::dsl
