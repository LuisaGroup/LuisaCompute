#pragma once

#include <luisa/runtime/rtx/hit.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/var.h>

namespace luisa::compute {

[[nodiscard]] LC_DSL_API Var<float> triangle_interpolate(Expr<float2> bary, Expr<float> a, Expr<float> b, Expr<float> c) noexcept;
[[nodiscard]] LC_DSL_API Var<float2> triangle_interpolate(Expr<float2> bary, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept;
[[nodiscard]] LC_DSL_API Var<float3> triangle_interpolate(Expr<float2> bary, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept;
[[nodiscard]] LC_DSL_API Var<float4> triangle_interpolate(Expr<float2> bary, Expr<float4> a, Expr<float4> b, Expr<float4> c) noexcept;

// legacy APIs
#define LUISA_INTERPOLATE_DEPRECATED \
    [[deprecated("Use triangle_interpolate instead.")]]

LUISA_INTERPOLATE_DEPRECATED [[nodiscard]] LC_DSL_API inline Var<float>
interpolate(Expr<float2> bary, Expr<float> a, Expr<float> b, Expr<float> c) noexcept {
    return triangle_interpolate(bary, a, b, c);
}
LUISA_INTERPOLATE_DEPRECATED [[nodiscard]] LC_DSL_API inline Var<float2>
interpolate(Expr<float2> bary, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept {
    return triangle_interpolate(bary, a, b, c);
}
LUISA_INTERPOLATE_DEPRECATED [[nodiscard]] LC_DSL_API inline Var<float3>
interpolate(Expr<float2> bary, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept {
    return triangle_interpolate(bary, a, b, c);
}
LUISA_INTERPOLATE_DEPRECATED [[nodiscard]] LC_DSL_API inline Var<float4>
interpolate(Expr<float2> bary, Expr<float4> a, Expr<float4> b, Expr<float4> c) noexcept {
    return triangle_interpolate(bary, a, b, c);
}

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::CommittedHit, inst, prim, bary, hit_type, committed_ray_t) {
    [[nodiscard]] auto miss() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Miss);
    }
    [[nodiscard]] auto is_triangle() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Surface) & bary.y >= 0.f;
    }
    [[nodiscard]] auto is_curve() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Surface) & bary.y < 0.f;
    }
    [[nodiscard]] auto is_procedural() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Procedural);
    }
    [[nodiscard]] auto distance() const noexcept { return committed_ray_t; }
    [[nodiscard]] auto triangle_barycentric_coord() const noexcept { return bary; }
    [[nodiscard]] auto curve_parameter() const noexcept { return bary.x; }

    template<typename A, typename B, typename C>
    LUISA_INTERPOLATE_DEPRECATED [[nodiscard]] auto interpolate(const A &a, const B &b, const C &c) const noexcept {
        return luisa::compute::interpolate(this->bary, a, b, c);
    }
    template<typename A, typename B, typename C>
    [[nodiscard]] auto triangle_interpolate(const A &a, const B &b, const C &c) const noexcept {
        return luisa::compute::triangle_interpolate(this->bary, a, b, c);
    }
};

LUISA_STRUCT(luisa::compute::SurfaceHit, inst, prim, bary, committed_ray_t) {

    [[nodiscard]] auto miss() const noexcept { return inst == std::numeric_limits<uint32_t>::max(); }
    [[nodiscard]] auto distance() const noexcept { return committed_ray_t; }
    [[nodiscard]] auto is_curve() const noexcept { return !miss() & bary.y < 0.f; }
    [[nodiscard]] auto is_triangle() const noexcept { return !miss() & !is_curve(); }
    [[nodiscard]] auto triangle_barycentric_coord() const noexcept { return bary; }
    [[nodiscard]] auto curve_parameter() const noexcept { return bary.x; }

    template<typename A, typename B, typename C>
    LUISA_INTERPOLATE_DEPRECATED [[nodiscard]] auto interpolate(const A &a, const B &b, const C &c) const noexcept {
        return luisa::compute::interpolate(this->bary, a, b, c);
    }
    template<typename A, typename B, typename C>
    [[nodiscard]] auto triangle_interpolate(const A &a, const B &b, const C &c) const noexcept {
        return luisa::compute::triangle_interpolate(this->bary, a, b, c);
    }
};

#undef LUISA_INTERPOLATE_DEPRECATED

LUISA_STRUCT(luisa::compute::ProceduralHit, inst, prim) {};
