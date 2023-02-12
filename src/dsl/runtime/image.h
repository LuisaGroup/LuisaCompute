#pragma once
#include <runtime/image.h>
#include <dsl/expr.h>
#ifndef LC_DISABLE_DSL
namespace luisa::compute {
template<typename T>
class ImageExprProxy {
private:
    T _img;

public:
    [[nodiscard]] auto read(Expr<uint2> uv) const noexcept {
        return Expr<T>{_img}.read(uv);
    }
    void write(Expr<uint2> uv, Expr<Vector<T, 4>> value) const noexcept {
        Expr<T>{_img}.write(uv, value);
    }
};
}// namespace luisa::compute
#endif
