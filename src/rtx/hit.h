//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <dsl/syntax.h>
#include <dsl/struct.h>
#include <core/mathematics.h>
#include <core/stl/format.h>

namespace luisa::compute {

enum class HitType : uint8_t {
    Miss = 0,
    Triangle = 1,
    Procedural = 2
};

struct Hit {
    uint inst{0u};
    uint prim{0u};
    float2 bary;
    uint hit_type;
    float committed_ray_t;
};

#ifndef LC_DISABLE_DSL
[[nodiscard]] LC_RUNTIME_API Var<float> interpolate(Expr<Hit> hit, Expr<float> a, Expr<float> b, Expr<float> c) noexcept;
[[nodiscard]] LC_RUNTIME_API Var<float2> interpolate(Expr<Hit> hit, Expr<float2> a, Expr<float2> b, Expr<float2> c) noexcept;
[[nodiscard]] LC_RUNTIME_API Var<float3> interpolate(Expr<Hit> hit, Expr<float3> a, Expr<float3> b, Expr<float3> c) noexcept;
[[nodiscard]] LC_RUNTIME_API Var<float4> interpolate(Expr<Hit> hit, Expr<float4> a, Expr<float4> b, Expr<float4> c) noexcept;
#endif

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Hit, inst, prim, bary, hit_type, committed_ray_t)

#ifndef LC_DISABLE_DSL
// clang-format off
LUISA_STRUCT_EXT(luisa::compute::Hit) {
    [[nodiscard]] auto miss() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Miss);
    }
    [[nodiscard]] auto hit_triangle() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Triangle);
    }
    [[nodiscard]] auto hit_procedural() const noexcept {
        return hit_type == static_cast<uint32_t>(luisa::compute::HitType::Procedural);
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
// clang-format on
#endif
