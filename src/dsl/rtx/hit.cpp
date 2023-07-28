#include <luisa/dsl/builtin.h>
#include <luisa/dsl/stmt.h>
#include <luisa/dsl/rtx/hit.h>

namespace luisa::compute {

Var<float> interpolate(Expr<float2> bary, Expr<float> a, Expr<float> b, Expr<float> c) noexcept {
    return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

Var<float2> interpolate(Expr<float2> bary, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept {
    return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

Var<float3> interpolate(Expr<float2> bary, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept {
    return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}
Var<float4> interpolate(Expr<float2> bary, Expr<float4> a, Expr<float4> b, Expr<float4> c) noexcept {
    return (1.0f - bary.x - bary.y) * a + bary.x * b + bary.y * c;
}

}// namespace luisa::compute

