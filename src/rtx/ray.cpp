//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/ray.h>

namespace luisa::compute {

detail::Expr<float3> origin(detail::Expr<Ray> ray) noexcept {
    return make_float3(ray.origin[0], ray.origin[1], ray.origin[2]);
}

detail::Expr<float3> direction(detail::Expr<Ray> ray) noexcept {
    return make_float3(ray.direction[0], ray.direction[1], ray.direction[2]);
}

void set_origin(detail::Expr<Ray> ray, detail::Expr<float3> origin) noexcept {
    ray.origin[0] = origin.x;
    ray.origin[1] = origin.y;
    ray.origin[2] = origin.z;
}

void set_direction(detail::Expr<Ray> ray, detail::Expr<float3> direction) noexcept {
    ray.direction[0] = direction.x;
    ray.direction[1] = direction.y;
    ray.direction[2] = direction.z;
}

detail::Expr<Ray> make_ray(detail::Expr<float3> origin, detail::Expr<float3> direction, detail::Expr<float> t_min, detail::Expr<float> t_max) noexcept {
    static Callable f = [](Float3 origin, Float3 direction, Float t_min, Float t_max) noexcept {
        Var<Ray> ray;
        ray.origin[0] = origin.x;
        ray.origin[1] = origin.y;
        ray.origin[2] = origin.z;
        ray.t_min = t_min;
        ray.direction[0] = direction.x;
        ray.direction[1] = direction.y;
        ray.direction[2] = direction.z;
        ray.t_max = t_max;
        return ray;
    };
    return f(origin, direction, t_min, t_max);
}

detail::Expr<Ray> make_ray(detail::Expr<float3> origin, detail::Expr<float3> direction) noexcept {
    return make_ray(origin, direction, 0.0f, std::numeric_limits<float>::max());
}

detail::Expr<Ray> make_ray_robust(
    detail::Expr<float3> p, detail::Expr<float3> ng,
    detail::Expr<float3> direction, detail::Expr<float> t_min, detail::Expr<float> t_max) noexcept {

    static Callable offset_origin = [](Float3 p, Float3 d, Float3 ng) noexcept {
        constexpr auto origin = 1.0f / 32.0f;
        constexpr auto float_scale = 1.0f / 65536.0f;
        constexpr auto int_scale = 256.0f;
        Var n = ite(dot(ng, d) < 0.0f, -ng, ng);
        Var of_i = make_int3(int_scale * n);
        Var p_i = as<float3>(as<int3>(p) + ite(p < 0.0f, -of_i, of_i));
        return ite(abs(p) < origin, p + float_scale * n, p_i);
    };
    return make_ray(offset_origin(p, direction, ng), direction, t_min, t_max);
}

detail::Expr<Ray> make_ray_robust(detail::Expr<float3> p, detail::Expr<float3> ng, detail::Expr<float3> direction) noexcept {
    return make_ray_robust(p, ng, direction, 0.0f, std::numeric_limits<float>::max());
}

}// namespace luisa::compute
