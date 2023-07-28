#pragma once

#include <luisa/runtime/rtx/ray.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/builtin.h>
#include <luisa/dsl/var.h>

LUISA_STRUCT(luisa::compute::Ray,
             compressed_origin,
             compressed_t_min,
             compressed_direction,
             compressed_t_max) {
    [[nodiscard]] auto origin() const noexcept { return luisa::compute::def<luisa::float3>(compressed_origin); }
    [[nodiscard]] auto direction() const noexcept { return luisa::compute::def<luisa::float3>(compressed_direction); }
    [[nodiscard]] auto t_min() const noexcept { return compressed_t_min; }
    [[nodiscard]] auto t_max() const noexcept { return compressed_t_max; }
    void set_origin(luisa::compute::Expr<luisa::float3> origin) noexcept { compressed_origin = origin; }
    void set_direction(luisa::compute::Expr<luisa::float3> direction) noexcept { compressed_direction = direction; }
    void set_t_min(luisa::compute::Expr<float> t_min) noexcept { compressed_t_min = t_min; }
    void set_t_max(luisa::compute::Expr<float> t_max) noexcept { compressed_t_max = t_max; }
};

namespace luisa::compute {

[[nodiscard]] LC_DSL_API Float3 offset_ray_origin(
    Expr<float3> p, Expr<float3> n) noexcept;

[[nodiscard]] LC_DSL_API Float3 offset_ray_origin(
    Expr<float3> p, Expr<float3> n, Expr<float3> w) noexcept;

[[nodiscard]] LC_DSL_API Var<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction,
    Expr<float> t_min,
    Expr<float> t_max) noexcept;

[[nodiscard]] LC_DSL_API Var<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction) noexcept;

}// namespace luisa::compute
