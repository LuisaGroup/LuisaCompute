//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/hit.h>

namespace luisa::compute {

Expr<bool> miss(Expr<Hit> hit) noexcept {
    static Callable _miss = [](Var<Hit> hit) noexcept {
        return hit.inst == std::numeric_limits<uint>::max();
    };
    return _miss(hit);
}

Expr<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept {
    static Callable _interpolate = [](Var<Hit> hit, Float a, Float b, Float c) noexcept {
        return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
    };
    return _interpolate(hit, a, b, c);
}

Expr<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept {
    static Callable _interpolate = [](Var<Hit> hit, Float2 a, Float2 b, Float2 c) noexcept {
        return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
    };
    return _interpolate(hit, a, b, c);
}

Expr<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept {
    static Callable _interpolate = [](Var<Hit> hit, Float3 a, Float3 b, Float3 c) noexcept {
        return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
    };
    return _interpolate(hit, a, b, c);
}

}// namespace luisa::compute
