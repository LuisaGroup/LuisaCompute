//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <ast/constant_data.h>
#include <dsl/expr.h>
#include <serialize/key_value_pair.h>

namespace luisa::compute {

/// Constant class
template<typename T>
class Constant {

    static_assert(concepts::basic<T>);

private:
    const Type *_type;
    ConstantData _data;

public:
    /// Default constructor for serialization
    Constant() noexcept = default;

    /// Construct constant from span
    Constant(luisa::span<const T> data) noexcept
        : _type{Type::from(luisa::format("array<{},{}>", Type::of<T>()->description(), data.size()))},
          _data{ConstantData::create(data)} {}

    /// Construct constant from array
    Constant(const T *data, size_t size) noexcept
        : Constant{luisa::span{data, size}} {}

    /// Construct constant from array-like data
    template<typename U>
    Constant(U &&data) noexcept : Constant{luisa::span<const T>{std::forward<U>(data)}} {}

    /// Construct constant from initializer list
    Constant(std::initializer_list<T> init) noexcept : Constant{luisa::vector<T>{init}} {}

    Constant(Constant &&) noexcept = default;
    Constant(const Constant &) noexcept = delete;
    Constant &operator=(Constant &&) noexcept = delete;
    Constant &operator=(const Constant &) noexcept = delete;

    /// Access member of constant
    template<typename U>
    requires is_integral_expr_v<U>
    [[nodiscard]] auto operator[](U &&index) const noexcept {
        return def<T>(detail::FunctionBuilder::current()->access(
            Type::of<T>(),
            detail::FunctionBuilder::current()->constant(_type, _data),
            detail::extract_expression(std::forward<U>(index))));
    }

    /// Read at index. Same as operator[]
    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return (*this)[std::forward<I>(index)];
    }

    template<typename S>
    void save(S& s) {
        luisa::string description(_type->description());
        s.serialize(
            MAKE_NAME_PAIR(_data),
            KeyValuePair{"_type", description}
        );
    }

    template<typename S>
    void load(S& s) {
        luisa::string description;
        s.serialize(
            MAKE_NAME_PAIR(_data),
            KeyValuePair{"_type", description}
        );
        _type = Type::from(description);
    }
};

template<typename T>
Constant(luisa::span<T> data) -> Constant<T>;

template<typename T>
Constant(luisa::span<const T> data) -> Constant<T>;

template<typename T>
Constant(std::initializer_list<T>) -> Constant<T>;

template<concepts::container T>
Constant(T &&) -> Constant<std::remove_const_t<typename std::remove_cvref_t<T>::value_type>>;

}// namespace luisa::compute
