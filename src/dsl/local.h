//
// Created by Mike Smith on 2022/3/23.
//

#pragma once
#ifndef LC_DISABLE_DSL
#include <dsl/expr.h>
#include <core/stl.h>

namespace luisa::compute {

namespace detail {
LC_DSL_API void local_array_error_sizes_missmatch(size_t lhs, size_t rhs) noexcept;
}

template<typename T>
class Local {

private:
    const RefExpr *_expression;
    size_t _size;

public:
    explicit Local(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->local(
              Type::array(Type::of<T>(), n))},
          _size{n} {}

    template<typename U>
        requires is_array_expr_v<U>
    Local(U &&array) noexcept
        : _expression{detail::extract_expression(def(std::forward<U>(array)))},
          _size{array_expr_dimension_v<U>} {}

    Local(Local &&) noexcept = default;
    Local(const Local &another) noexcept
        : _expression{another._expression}, _size{another._size} {
        detail::FunctionBuilder::current()->assign(
            _expression, another._expression);
    }
    Local &operator=(const Local &rhs) noexcept {
        if (&rhs != this) [[likely]] {
            if (_size != rhs._size) [[unlikely]] {
                detail::local_array_error_sizes_missmatch(_size, rhs._size);
            }
            detail::FunctionBuilder::current()->assign(
                _expression, rhs._expression);
        }
        return *this;
    }
    Local &operator=(Local &&rhs) noexcept {
        *this = static_cast<const Local &>(rhs);
        return *this;
    }

    template<typename U>
        requires is_array_expr_v<U>
    Local &operator=(U &&rhs) noexcept {
        constexpr auto n = array_expr_dimension_v<U>;
        if (_size != n) [[unlikely]] {
            detail::local_array_error_sizes_missmatch(_size, n);
        }
        detail::FunctionBuilder::current()->assign(
            _expression, rhs._expression);
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
#endif