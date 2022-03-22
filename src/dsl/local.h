//
// Created by Mike Smith on 2022/3/23.
//

#pragma once

#include <dsl/expr.h>
#include <core/stl.h>

namespace luisa::compute {

template<typename T>
class Local {

private:
    const RefExpr *_expression;
    size_t _size;

public:
    explicit Local(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->local(
              Type::from(luisa::format("array<{},{}>", Type::of<T>()->description(), n)))},
          _size{n} {}

    template<typename U>
        requires is_array_expr_v<U> Local(U &&array)
    noexcept
        : _expression{detail::extract_expression(def(std::forward<U>(array)))},
          _size{array_expr_dimension_v<U>} {}

    Local(Local &&) noexcept = default;
    Local(const Local &another) noexcept
        : _expression{another._expression}, _size{another._size} {
        for (auto i = 0u; i < _size; i++) {
            (*this)[i] = another[i];
        }
    }
    Local &operator=(const Local &rhs) noexcept {
        if (&rhs != this) [[likely]] {
            LUISA_ASSERT(
                _size == rhs._size,
                "Incompatible sizes ({} and {}).",
                _size, rhs._size);
            for (auto i = 0u; i < _size; i++) {
                (*this)[i] = rhs[i];
            }
        }
        return *this;
    }
    Local &operator=(Local &&rhs) noexcept {
        *this = static_cast<const Local &>(rhs);
        return *this;
    }

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &operator[](U &&index) const noexcept {
        auto i = def(std::forward<U>(index));
        auto f = detail::FunctionBuilder::current();
        auto expr = f->access(
            Type::of<T>(), _expression, i.expression());
        return *f->create_temporary<Var<T>>(expr);
    }

    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept { return (*this)[std::forward<I>(index)]; }

    template<typename I, typename U>
    void write(I &&i, U &&u) const noexcept { (*this)[std::forward<I>(i)] = std::forward<U>(u); }
};

}// namespace luisa::compute