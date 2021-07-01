//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <dsl/expr.h>

namespace luisa::compute {

namespace detail {

template<typename>
struct SharedAsAtomic {};

}// namespace detail

template<typename T>
class Shared : public detail::SharedAsAtomic<T> {

private:
    const RefExpr *_expression;

public:
    explicit Shared(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->shared(
            Type::from(fmt::format("array<{},{}>", Type::of<T>()->description(), n)))} {}

    Shared(Shared &&) noexcept = default;
    Shared(const Shared &) noexcept = delete;
    Shared &operator=(Shared &&) noexcept = delete;
    Shared &operator=(const Shared &) noexcept = delete;

    [[nodiscard]] auto expression() const noexcept { return _expression; }

    template<concepts::integral U>
    [[nodiscard]] auto operator[](detail::Expr<U> index) const noexcept {
        return detail::Expr<T>{detail::FunctionBuilder::current()->access(
            Type::of<T>(), _expression, index.expression())};
    }

    template<concepts::integral U>
    [[nodiscard]] auto operator[](U index) const noexcept { return (*this)[detail::Expr{index}]; }
};

namespace detail {

template<>
struct SharedAsAtomic<int> {
    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return Expr<Atomic<int>>{static_cast<const Shared<int> &>(*this)[std::forward<I>(i)].expression()};
    }
};

template<>
struct SharedAsAtomic<uint> {
    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return Expr<Atomic<uint>>{static_cast<const Shared<uint> &>(*this)[std::forward<I>(i)].expression()};
    }
};

}// namespace detail

}// namespace luisa::compute
