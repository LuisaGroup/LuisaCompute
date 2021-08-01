//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/hit.h>

namespace luisa::compute {

detail::Expr<bool> miss(detail::Expr<Hit> hit) noexcept {
    static Callable _miss = [](Var<Hit> hit) noexcept {
        return hit.inst == std::numeric_limits<uint>::max();
    };
    return _miss(hit);
}

detail::Expr<float> interpolate(detail::Expr<Hit> hit, detail::Expr<float> a, detail::Expr<float> b, detail::Expr<float> c) noexcept {
    static Callable _interpolate = [](Var<Hit> hit, Float a, Float b, Float c) noexcept {
        return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
    };
    return _interpolate(hit, a, b, c);
}

detail::Expr<float2> interpolate(detail::Expr<Hit> hit, detail::Expr<float2> a, detail::Expr<float2> b, detail::Expr<float2> c) noexcept {
    static Callable _interpolate = [](Var<Hit> hit, Float2 a, Float2 b, Float2 c) noexcept {
        return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
    };
    return _interpolate(hit, a, b, c);
}

detail::Expr<float3> interpolate(detail::Expr<Hit> hit, detail::Expr<float3> a, detail::Expr<float3> b, detail::Expr<float3> c) noexcept {
    static Callable _interpolate = [](Var<Hit> hit, Float3 a, Float3 b, Float3 c) noexcept {
        return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
    };
    return _interpolate(hit, a, b, c);
}

}// namespace luisa::compute
