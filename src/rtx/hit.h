//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <dsl/syntax.h>

namespace luisa::compute {

struct alignas(16) Hit {
    uint inst{0u};
    uint prim{0u};
    float2 uv;
};

[[nodiscard]] Var<bool> miss(Expr<Hit> hit) noexcept;
[[nodiscard]] Var<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept;
[[nodiscard]] Var<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept;
[[nodiscard]] Var<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept;

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Hit, inst, prim, uv){
    [[nodiscard]] auto miss() const noexcept {
        return luisa::compute::miss(*this);
    }
    template<typename A, typename B, typename C>
    [[nodiscard]] auto interpolate(A &&a, B &&b, C &&c) const noexcept {
        return luisa::compute::interpolate(
            *this,
            std::forward<A>(a),
            std::forward<B>(b),
            std::forward<C>(c));
    }
};
