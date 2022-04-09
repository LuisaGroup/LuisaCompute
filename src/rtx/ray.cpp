//
// Created by Mike Smith on 2021/6/24.
//

#include <rtx/ray.h>

namespace luisa::compute {

Var<Ray> make_ray(Expr<float3> origin, Expr<float3> direction, Expr<float> t_min, Expr<float> t_max) noexcept {
    Var<Ray> ray{origin, t_min, direction, t_max};
    return ray;
}

Var<Ray> make_ray(Expr<float3> origin, Expr<float3> direction) noexcept {
    return make_ray(origin, direction, 0.0f, std::numeric_limits<float>::max());
}

Float3 offset_ray_origin(Expr<float3> p, Expr<float3> n) noexcept {
    constexpr auto origin = 1.0f / 32.0f;
    constexpr auto float_scale = 1.0f / 65536.0f;
    constexpr auto int_scale = 256.0f;
    auto of_i = make_int3(int_scale * n);
    auto p_i = as<float3>(as<int3>(p) + ite(p < 0.0f, -of_i, of_i));
    return ite(abs(p) < origin, p + float_scale * n, p_i);
}

Float3 compute::offset_ray_origin(Expr<float3> p, Expr<float3> n, Expr<float3> w) noexcept {
    return offset_ray_origin(p, faceforward(n, -w, n));
}

}// namespace luisa::compute
