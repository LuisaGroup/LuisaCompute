//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <dsl/struct.h>
#include <dsl/syntax.h>

namespace luisa::compute {

struct alignas(16) Ray {
    std::array<float, 3> origin;
    float t_min;
    std::array<float, 3> direction;
    float t_max;
};

}

LUISA_STRUCT(luisa::compute::Ray, origin, t_min, direction, t_max)

namespace luisa::compute {

[[nodiscard]] Expr<float3> origin(Expr<Ray> ray) noexcept;
[[nodiscard]] Expr<float3> direction(Expr<Ray> ray) noexcept;

void set_origin(Expr<Ray> ray, Expr<float3> origin) noexcept;
void set_direction(Expr<Ray> ray, Expr<float3> direction) noexcept;

[[nodiscard]] Expr<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction,
    Expr<float> t_min,
    Expr<float> t_max) noexcept;

[[nodiscard]] Expr<Ray> make_ray(
    Expr<float3> origin,
    Expr<float3> direction) noexcept;

// ray from p with surface normal ng, with self intersections avoidance
[[nodiscard]] Expr<Ray> make_ray_robust(
    Expr<float3> p,
    Expr<float3> ng,
    Expr<float3> direction,
    Expr<float> t_min,
    Expr<float> t_max) noexcept;

[[nodiscard]] Expr<Ray> make_ray_robust(
    Expr<float3> p,
    Expr<float3> ng,
    Expr<float3> direction) noexcept;

}// namespace luisa::compute
