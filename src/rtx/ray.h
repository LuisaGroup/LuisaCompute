//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <dsl/struct.h>
#include <dsl/syntax.h>

namespace luisa::compute {

struct alignas(16) Ray {
    float origin[3];
    float t_min;
    float direction[3];
    float t_max;
};

}

LUISA_STRUCT(luisa::compute::Ray, origin, t_min, direction, t_max)

namespace luisa::compute {

[[nodiscard]] detail::Expr<float3> origin(detail::Expr<Ray> ray) noexcept;
[[nodiscard]] detail::Expr<float3> direction(detail::Expr<Ray> ray) noexcept;

void set_origin(detail::Expr<Ray> ray, detail::Expr<float3> origin) noexcept;
void set_direction(detail::Expr<Ray> ray, detail::Expr<float3> direction) noexcept;

[[nodiscard]] detail::Expr<Ray> make_ray(
    detail::Expr<float3> origin,
    detail::Expr<float3> direction,
    detail::Expr<float> t_min,
    detail::Expr<float> t_max) noexcept;

[[nodiscard]] detail::Expr<Ray> make_ray(
    detail::Expr<float3> origin,
    detail::Expr<float3> direction) noexcept;

// ray from p with surface normal ng, with self intersections avoidance
[[nodiscard]] detail::Expr<Ray> make_ray_robust(
    detail::Expr<float3> p,
    detail::Expr<float3> ng,
    detail::Expr<float3> direction,
    detail::Expr<float> t_min,
    detail::Expr<float> t_max) noexcept;

[[nodiscard]] detail::Expr<Ray> make_ray_robust(
    detail::Expr<float3> p,
    detail::Expr<float3> ng,
    detail::Expr<float3> direction) noexcept;

}// namespace luisa::compute
