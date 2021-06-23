//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

namespace detail {

template<typename>
struct SharedAsAtomic {};

}

template<typename T>
class Shared : public detail::SharedAsAtomic<T> {

private:
    const RefExpr *_expression;

public:
    explicit Shared(size_t n) noexcept
        : _expression{FunctionBuilder::current()->shared(
            Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), n)))} {}

    Shared(Shared &&) noexcept = default;
    Shared(const Shared &) noexcept = delete;
    Shared &operator=(Shared &&) noexcept = delete;
    Shared &operator=(const Shared &) noexcept = delete;

    [[nodiscard]] auto expression() const noexcept { return _expression; }

    template<concepts::integral U>
    [[nodiscard]] auto operator[](detail::Expr<U> index) const noexcept {
        return detail::Expr<T>{FunctionBuilder::current()->access(
            Type::of<T>(), _expression, index.expression())};
    }

    template<concepts::integral U>
    [[nodiscard]] auto operator[](U index) const noexcept { return (*this)[detail::Expr{index}]; }
};

}// namespace luisa::compute
