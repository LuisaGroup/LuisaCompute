//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/hit.h>

namespace luisa::compute {

Var<bool> miss(Expr<Hit> hit) noexcept {
    return hit.inst == std::numeric_limits<uint>::max();
}

Var<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept {
    return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
}

Var<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept {
    return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
}

Var<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept {
    return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
}

}// namespace luisa::compute
