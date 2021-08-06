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

[[nodiscard]] Expr<bool> miss(Expr<Hit> hit) noexcept;
[[nodiscard]] Expr<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept;
[[nodiscard]] Expr<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept;
[[nodiscard]] Expr<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept;

}

LUISA_STRUCT(luisa::compute::Hit, inst, prim, uv)
