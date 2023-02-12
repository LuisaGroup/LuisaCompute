#pragma once
#ifndef LC_DISABLE_DSL
#include <dsl/struct.h>
#include <dsl/builtin.h>
#include <dsl/var.h>
#include <rtx/ray.h>
// clang-format off
LUISA_STRUCT_EXT(luisa::compute::Ray) {
    [[nodiscard]] auto origin() const noexcept { return luisa::compute::def<luisa::float3>(compressed_origin); }
    [[nodiscard]] auto direction() const noexcept { return luisa::compute::def<luisa::float3>(compressed_direction); }
    [[nodiscard]] auto t_min() const noexcept { return compressed_t_min; }
    [[nodiscard]] auto t_max() const noexcept { return compressed_t_max; }
    void set_origin(luisa::compute::Expr<luisa::float3> origin) noexcept { compressed_origin = origin; }
    void set_direction(luisa::compute::Expr<luisa::float3> direction) noexcept { compressed_direction = direction; }
    void set_t_min(luisa::compute::Expr<float> t_min) noexcept { compressed_t_min = t_min; }
    void set_t_max(luisa::compute::Expr<float> t_max) noexcept { compressed_t_max = t_max; }
};
// clang-format on

namespace luisa::compute {

[[nodiscard]] LC_RUNTIME_API Float3 offset_ray_origin(
    Expr<float3> p, Expr<float3> n) noexcept;

[[nodiscard]] LC_RUNTIME_API Float3 offset_ray_origin(
    Expr<float3> p, Expr<float3> n, Expr<float3> w) noexcept;

[[nodiscard]] LC_RUNTIME_API Var<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction,
    Expr<float> t_min,
    Expr<float> t_max) noexcept;

[[nodiscard]] LC_RUNTIME_API Var<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction) noexcept;
}// namespace luisa::compute

#endif