#pragma once

#include <luisa/dsl/expr.h>
#include <luisa/dsl/operators.h>

namespace luisa::compute {

namespace detail {

LC_DSL_API void local_array_error_sizes_missmatch(size_t lhs, size_t rhs) noexcept;

template<typename T>
[[nodiscard]] inline auto local_array_choose_type(size_t n) noexcept {
    auto elem = Type::of<T>();
    if (n <= 1u) { return elem; }
    return elem->is_scalar() && n <= 4u ?
               Type::vector(elem, n) :
               Type::array(elem, n);
}

}// namespace detail

template<typename T>
class Local {

private:
    const RefExpr *_expression;
    size_t _size;

public:
    explicit Local(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->local(
              detail::local_array_choose_type<T>(n))},
          _size{n} {}

    Local(Local &&) noexcept = default;
    Local(const Local &another) noexcept
        : _size{another._size} {
        auto fb = detail::FunctionBuilder::current();
        _expression = fb->local(detail::local_array_choose_type<T>(_size));
        fb->assign(_expression, another._expression);
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

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] Var<T> &operator[](U &&index) const noexcept {
        auto f = detail::FunctionBuilder::current();
        if (_size == 1u) { return *f->create_temporary<Var<T>>(_expression); }
        auto i = def(std::forward<U>(index));
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

// disable address-of operators
template<typename T>
[[nodiscard]] inline ::luisa::compute::Local<T> *operator&(::luisa::compute::Local<T> &) noexcept {
    static_assert(::luisa::always_false_v<T>,
                  LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE);
    std::abort();
}

template<typename T>
[[nodiscard]] inline const ::luisa::compute::Local<T> *operator&(const ::luisa::compute::Local<T> &) noexcept {
    static_assert(::luisa::always_false_v<T>,
                  LUISA_DISABLE_DSL_ADDRESS_OF_MESSAGE);
    std::abort();
}

