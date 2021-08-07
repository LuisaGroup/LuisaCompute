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

    template<typename U>
    requires is_integral_expr_v<U>
    [[nodiscard]] auto operator[](U &&index) const noexcept {
        return Ref<T>{detail::FunctionBuilder::current()->access(
            Type::of<T>(), _expression,
            detail::extract_expression(std::forward<U>(index)))};
    }
};

namespace detail {

template<>
struct SharedAsAtomic<int> {
    [[nodiscard]] auto atomic(Expr<int> i) const noexcept {
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Shared<int> *>(this)->expression(),
            i.expression())};
    }
    [[nodiscard]] auto atomic(Expr<uint> i) const noexcept {
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Shared<int> *>(this)->expression(),
            i.expression())};
    }
};

template<>
struct SharedAsAtomic<uint> {
    [[nodiscard]] auto atomic(Expr<int> i) const noexcept {
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Shared<uint> *>(this)->expression(),
            i.expression())};
    }
    [[nodiscard]] auto atomic(Expr<uint> i) const noexcept {
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Shared<uint> *>(this)->expression(),
            i.expression())};
    }
};

}// namespace detail

}// namespace luisa::compute
