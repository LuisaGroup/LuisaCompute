//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/ray.h>

namespace luisa::compute {

Var<float3> origin(Expr<Ray> ray) noexcept {
    return ray.origin;
}

Var<float3> direction(Expr<Ray> ray) noexcept {
    return ray.direction;
}

void set_origin(Var<Ray> &ray, Expr<float3> origin) noexcept {
    ray.origin = origin;
}

void set_direction(Var<Ray> &ray, Expr<float3> direction) noexcept {
    ray.direction = direction;
}

Var<Ray> make_ray(Expr<float3> origin, Expr<float3> direction, Expr<float> t_min, Expr<float> t_max) noexcept {
    Var<Ray> ray{origin, t_min, direction, t_max};
    return ray;
}

Var<Ray> make_ray(Expr<float3> origin, Expr<float3> direction) noexcept {
    return make_ray(origin, direction, 0.0f, std::numeric_limits<float>::max());
}

Var<Ray> make_ray_robust(
    Expr<float3> p, Expr<float3> ng,
    Expr<float3> direction, Expr<float> t_max) noexcept {

    static Callable _make_ray_robust = [](Float3 p, Float3 d, Float3 ng, Float t_max) noexcept {
        constexpr auto origin = 1.0f / 32.0f;
        constexpr auto float_scale = 1.0f / 65536.0f;
        constexpr auto int_scale = 256.0f;
        auto n = faceforward(ng, -d, ng);
        auto of_i = make_int3(int_scale * n);
        auto p_i = as<float3>(as<int3>(p) + ite(p < 0.0f, -of_i, of_i));
        auto ro = ite(abs(p) < origin, p + float_scale * n, p_i);
        return make_ray(ro, d, 0.0f, t_max);
    };
    return _make_ray_robust(p, direction, ng, t_max);
}

Var<Ray> make_ray_robust(Expr<float3> p, Expr<float3> ng, Expr<float3> direction) noexcept {
    return make_ray_robust(p, ng, direction, std::numeric_limits<float>::max());
}

}// namespace luisa::compute
