//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute::dsl {

namespace detail {

struct ArgumentCreation {};

template<typename T>
[[nodiscard]] inline auto create_argument() noexcept {
    return T{ArgumentCreation{}};
}

}// namespace detail

template<typename T>
struct Var : public detail::Expr<T> {

    // for making function arguments...
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->uniform(Type::of<T>())} {}

    // for local variables
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
struct Var<Buffer<T>> : public detail::Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Buffer<T>>{
            FunctionBuilder::current()->buffer(Type::of<Buffer<T>>())} {}
    Var(Var &&another) noexcept = default;
};

template<typename T>
Var(detail::Expr<T>) -> Var<T>;

template<concepts::Native T>
Var(T) -> Var<T>;

template<typename T>
struct Shared : public detail::Expr<std::array<T, 1>> {
};

template<typename T>
struct Constant : public detail::Expr<std::array<T, 1>> {
};

template<typename T>
struct is_var : std::false_type {};

template<typename T>
struct is_var<Var<T>> : std::true_type {};

template<typename T>
constexpr auto is_var_v = is_var<T>::value;

}// namespace luisa::compute::dsl
