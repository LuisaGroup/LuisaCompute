#pragma once

#include <luisa/dsl/expr.h>
#include <luisa/dsl/atomic.h>

namespace luisa::compute {

template<typename T>
class Shared;

namespace detail {

template<typename T>
struct SharedAsAtomic {
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto atomic(I &&i) const noexcept {
        auto index = def(std::forward<I>(i));
        auto shared = static_cast<const Shared<T> *>(this)->expression();
        return AtomicRef<T>{AtomicRefNode::create(shared)
                                  ->access(index.expression())};
    }
};

// no-op for non-atomic types
template<typename T>
    requires is_custom_struct_v<T>
struct SharedAsAtomic<T> {};

}// namespace detail

/// Shared class
template<typename T>
class Shared : public detail::SharedAsAtomic<T> {

private:
    const RefExpr *_expression;
    size_t _size;

public:
    /// Create a shared array of size n
    explicit Shared(size_t n) noexcept
        : _expression{detail::FunctionBuilder::current()->shared(
              Type::array(Type::of<T>(), n))},
          _size{n} {}

    Shared(Shared &&) noexcept = default;
    Shared(const Shared &) noexcept = delete;
    Shared &operator=(Shared &&) noexcept = delete;
    Shared &operator=(const Shared &) noexcept = delete;

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }

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
    [[nodiscard]] auto read(I &&index) const noexcept {
        return (*this)[std::forward<I>(index)];
    }

    /// Write index
    template<typename I, typename U>
    void write(I &&i, U &&u) const noexcept {
        (*this)[std::forward<I>(i)] = std::forward<U>(u);
    }
};

}// namespace luisa::compute

