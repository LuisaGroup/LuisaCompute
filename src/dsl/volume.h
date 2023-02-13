#pragma once

#include <runtime/volume.h>
#include <dsl/expr.h>

#ifndef LC_DISABLE_DSL

namespace luisa::compute {

template<typename T>
class VolumeExprProxy {

private:
    T _img;

public:
    [[nodiscard]] auto read(Expr<uint3> uv) const noexcept {
        return Expr<T>{_img}.read(uv);
    }
    void write(Expr<uint3> uv, Expr<Vector<T, 4>> value) const noexcept {
        Expr<T>{_img}.write(uv, value);
    }
};

}// namespace luisa::compute

#endif
