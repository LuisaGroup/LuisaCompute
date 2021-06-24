//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/hit.h>

namespace luisa::compute {

detail::Expr<bool> miss(detail::Expr<Hit> hit) noexcept {
    return hit.prim == std::numeric_limits<uint>::max();
}

detail::Expr<float> interpolate(detail::Expr<Hit> hit, detail::Expr<float> a, detail::Expr<float> b, detail::Expr<float> c) noexcept {
    return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
}

detail::Expr<float2> interpolate(detail::Expr<Hit> hit, detail::Expr<float2> a, detail::Expr<float2> b, detail::Expr<float2> c) noexcept {
    return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
}

detail::Expr<float3> interpolate(detail::Expr<Hit> hit, detail::Expr<float3> a, detail::Expr<float3> b, detail::Expr<float3> c) noexcept {
    return (1.0f - hit.uv.x - hit.uv.y) * a + hit.uv.x * b + hit.uv.y * c;
}

}// namespace luisa::compute
