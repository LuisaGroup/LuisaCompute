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

[[nodiscard]] detail::Expr<bool> miss(detail::Expr<Hit> hit) noexcept;
[[nodiscard]] detail::Expr<float> interpolate(detail::Expr<Hit> hit, detail::Expr<float> a, detail::Expr<float> b, detail::Expr<float> c) noexcept;
[[nodiscard]] detail::Expr<float2> interpolate(detail::Expr<Hit> hit, detail::Expr<float2> a, detail::Expr<float2> b, detail::Expr<float2> c) noexcept;
[[nodiscard]] detail::Expr<float3> interpolate(detail::Expr<Hit> hit, detail::Expr<float3> a, detail::Expr<float3> b, detail::Expr<float3> c) noexcept;

}

LUISA_STRUCT(luisa::compute::Hit, prim, inst, uv)
