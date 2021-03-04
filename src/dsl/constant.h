//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute::dsl {

template<typename T>
class Constant {

    static_assert(concepts::Basic<T>);

private:
    const Expression *_expression;

public:
    Constant(std::span<const T> data) noexcept
        : _expression{FunctionBuilder::current()->constant(data.data(), data.size())} {}
    Constant(std::initializer_list<T> data) noexcept
        : _expression{FunctionBuilder::current()->constant(data)} {}
    Constant(const T *data, size_t size) noexcept
        : Constant{std::span{data, size}} {}
    template<typename U>
    Constant(U &&data) noexcept : Constant{std::span<const T>{std::forward<U>(data)}} {}
    
    Constant(Constant &&) noexcept = default;
    Constant(const Constant &) noexcept = delete;
    Constant &operator=(Constant &&) noexcept = delete;
    Constant &operator=(const Constant &) noexcept = delete;
    
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    
    template<concepts::Integral U>
    [[nodiscard]] auto operator[](detail::Expr<U> index) const noexcept {
        return detail::Expr<T>{FunctionBuilder::current()->access(
            Type::of<T>(), _expression, index.expression())};
    }
    
    template<concepts::Integral U>
    [[nodiscard]] auto operator[](U index) const noexcept { return (*this)[detail::Expr{index}]; }
};

template<typename T>
Constant(std::span<T> data) -> Constant<T>;

template<typename T>
Constant(std::span<const T> data) -> Constant<T>;

template<typename T>
Constant(std::initializer_list<T>) -> Constant<T>;

template<concepts::Container T>
Constant(T &&) -> Constant<std::remove_const_t<typename std::remove_cvref_t<T>::value_type>>;

}// namespace luisa::compute::dsl
