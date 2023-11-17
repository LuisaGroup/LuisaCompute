#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/rtx/curve.h>

namespace luisa::compute {

std::pair<Float3, Float3>
CurveInterpolator::surface_position_and_normal(Expr<float> u, Expr<float3> ps_in) const noexcept {
    auto ps = def(ps_in);
    auto normal = def(make_float3());
    $outline {
        $if (u == 0.f) {
            normal = -derivative(0.f).xyz();
        }
        $elif (u == 1.f) {
            normal = derivative(1.f).xyz();
        }
        $else {
            auto p4 = position(u);
            auto p = p4.xyz();
            auto r = p4.w;
            auto d4 = derivative(u);
            auto d = d4.xyz();
            auto dr = d4.w;
            auto dd = dot(d, d);
            auto o1 = ps - p;
            o1 -= dot(o1, d) / dd * d;
            o1 *= r / length(o1);
            ps = p + o1;
            dd -= dot(second_derivative(u).xyz(), o1);
            normal = dd * o1 - (dr * r) * d;
        };
    };
    return std::make_pair(ps, normalize(normal));
}

Float3 CurveInterpolator::tangent(Expr<float> u) const noexcept {
    return normalize(derivative(u).xyz());
}

PiecewiseLinearCurve::PiecewiseLinearCurve(Expr<float4> q0, Expr<float4> q1) noexcept
    : _p0{q0}, _p1{q1 - q0} {}

Float4 PiecewiseLinearCurve::position(Expr<float> u) const noexcept {
    return _p0 + u * _p1;
}

Float4 PiecewiseLinearCurve::derivative(Expr<float> u) const noexcept {
    return _p1;
}

Float4 PiecewiseLinearCurve::second_derivative(Expr<float> u) const noexcept {
    return make_float4(0.f);
}

std::pair<Float3, Float3>
PiecewiseLinearCurve::surface_position_and_normal(Expr<float> u, Expr<float3> ps_in) const noexcept {
    auto ps = def(ps_in);
    auto normal = def(make_float3());
    $outline {
        $if (u == 0.f) {
            normal = ps - _p0.xyz();
        }
        $elif (u >= 1.f) {
            auto p1 = _p1.xyz() + _p0.xyz();
            normal = ps - p1;
        }
        $else {
            auto p4 = position(u);
            auto p = p4.xyz();
            auto r = p4.w;
            auto d4 = derivative(u);
            auto d = d4.xyz();
            auto dr = d4.w;
            auto dd = dot(d, d);
            auto o1 = ps - p;
            o1 -= dot(o1, d) / dd * d;
            o1 *= r / length(o1);
            ps = p + o1;
            normal = dd * o1 - (dr * r) * d;
        };
    };
    return std::make_pair(ps, normalize(normal));
}

CubicCurve::CubicCurve(Float4 p0, Float4 p1, Float4 p2, Float4 p3) noexcept
    : _p0{std::move(p0)},
      _p1{std::move(p1)},
      _p2{std::move(p2)},
      _p3{std::move(p3)} {}

Float4 CubicCurve::position(Expr<float> u) const noexcept {
    return (((_p0 * u) + _p1) * u + _p2) * u + _p3;
}

Float4 CubicCurve::derivative(Expr<float> u_in) const noexcept {
    auto u = clamp(u_in, 1e-6f, 1.f - 1e-6f);
    return ((3.f * _p0 * u) + 2.f * _p1) * u + _p2;
}

Float4 CubicCurve::second_derivative(Expr<float> u) const noexcept {
    return 6.f * _p0 * u + 2.f * _p1;
}

CubicBSplineCurve::CubicBSplineCurve(Expr<float4> q0,
                                     Expr<float4> q1,
                                     Expr<float4> q2,
                                     Expr<float4> q3) noexcept
    : CubicCurve{(q0 * (-1.f) + q1 * (3.f) + q2 * (-3.f) + q3) / 6.f,
                 (q0 * (3.f) + q1 * (-6.f) + q2 * (3.f)) / 6.f,
                 (q0 * (-3.f) + q2 * (3.f)) / 6.f,
                 (q0 * (1.f) + q1 * (4.f) + q2 * (1.f)) / 6.f} {}

CatmullRomCurve::CatmullRomCurve(Expr<float4> q0,
                                 Expr<float4> q1,
                                 Expr<float4> q2,
                                 Expr<float4> q3) noexcept
    : CubicCurve{(-1.f * q0 + (3.f) * q1 + (-3.f) * q2 + (1.f) * q3) / 2.f,
                 (2.f * q0 + (-5.f) * q1 + (4.f) * q2 + (-1.f) * q3) / 2.f,
                 (-1.f * q0 + (1.f) * q2) / 2.f,
                 ((2.f) * q1) / 2.f} {}

}// namespace luisa::compute
