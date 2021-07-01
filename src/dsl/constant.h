//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <ast/constant_data.h>
#include <dsl/expr.h>

namespace luisa::compute {

template<typename T>
class Constant {

    static_assert(concepts::basic<T>);

private:
    const Type *_type;
    uint64_t _hash;

public:
    Constant(std::span<const T> data) noexcept
        : _type{Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), data.size()))},
          _hash{ConstantData::create(data)} {}

    Constant(const T *data, size_t size) noexcept
        : Constant{std::span{data, size}} {}

    template<typename U>
    Constant(U &&data) noexcept : Constant{std::span<const T>{std::forward<U>(data)}} {}

    Constant(std::initializer_list<T> init) noexcept : Constant{std::vector<T>{init}} {}

    Constant(Constant &&) noexcept = default;
    Constant(const Constant &) noexcept = delete;
    Constant &operator=(Constant &&) noexcept = delete;
    Constant &operator=(const Constant &) noexcept = delete;

    template<concepts::integral U>
    [[nodiscard]] auto operator[](detail::Expr<U> index) const noexcept {
        return detail::Expr<T>{detail::FunctionBuilder::current()->access(
            Type::of<T>(),
            detail::FunctionBuilder::current()->constant(_type, _hash),
            index.expression())};
    }

    template<concepts::integral U>
    [[nodiscard]] auto operator[](U index) const noexcept { return (*this)[detail::Expr{index}]; }
};

template<typename T>
Constant(std::span<T> data) -> Constant<T>;

template<typename T>
Constant(std::span<const T> data) -> Constant<T>;

template<typename T>
Constant(std::initializer_list<T>) -> Constant<T>;

template<concepts::container T>
Constant(T &&) -> Constant<std::remove_const_t<typename std::remove_cvref_t<T>::value_type>>;

}// namespace luisa::compute
