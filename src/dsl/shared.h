//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <dsl/expr.h>
#include <core/stl.h>

namespace luisa::compute {

namespace detail {

template<typename>
struct SharedAsAtomic {};

}// namespace detail

/// Shared class
template<typename T>
class Shared : public detail::SharedAsAtomic<T> {

private:
    const RefExpr *_expression;

public:
    /// Create a shared array of size n
    explicit Shared(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->shared(
              Type::from(luisa::format("array<{},{}>", Type::of<T>()->description(), n)))} {}

    Shared(Shared &&) noexcept = default;
    Shared(const Shared &) noexcept = delete;
    Shared &operator=(Shared &&) noexcept = delete;
    Shared &operator=(const Shared &) noexcept = delete;

    [[nodiscard]] auto expression() const noexcept { return _expression; }

    /// Access at index
    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] auto &operator[](U &&index) const noexcept {
        auto i = def(std::forward<U>(index));
        auto f = detail::FunctionBuilder::current();
        auto expr = f->access(
            Type::of<T>(), _expression, i.expression());
        return *f->create_temporary<Var<T>>(expr);
    }

    /// Read index
    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept { return (*this)[std::forward<I>(index)]; }

    /// Write index
    template<typename I, typename U>
    void write(I &&i, U &&u) const noexcept { (*this)[std::forward<I>(i)] = std::forward<U>(u); }
};

namespace detail {

template<>
struct SharedAsAtomic<int> {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<int>{FunctionBuilder::current()->access(
            Type::of<int>(),
            static_cast<const Shared<int> *>(this)->expression(),
            index.expression())};
    }
};

template<>
struct SharedAsAtomic<uint> {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        return AtomicRef<uint>{FunctionBuilder::current()->access(
            Type::of<uint>(),
            static_cast<const Shared<uint> *>(this)->expression(),
            index.expression())};
    }
};

}// namespace detail

}// namespace luisa::compute
