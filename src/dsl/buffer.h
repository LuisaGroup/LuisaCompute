#pragma once

#ifndef LC_DISABLE_DSL

#include <runtime/buffer.h>
#include <dsl/expr.h>

namespace luisa::compute {

template<typename T>
class BufferExprProxy {

private:
    T _buffer;

public:
    template<typename I>
        requires is_integral_expr_v<I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return Expr<Buffer<T>>{_buffer}.read(std::forward<I>(index));
    }
    template<typename I>
        requires is_integral_expr_v<I>
    void write(I &&index, Expr<T> value) const noexcept {
        return Expr<Buffer<T>>{_buffer}.write(std::forward<I>(index), value);
    }
};

}// namespace luisa::compute

#endif
