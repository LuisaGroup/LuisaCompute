#pragma once

#include <luisa/dsl/var.h>

namespace luisa::compute {

struct CurveInterpolator {

protected:
    ~CurveInterpolator() noexcept = default;

public:
    [[nodiscard]] virtual Float4 position(Expr<float> u) const noexcept = 0;
    [[nodiscard]] virtual Float4 derivative(Expr<float> u) const noexcept = 0;
    [[nodiscard]] virtual Float4 second_derivative(Expr<float> u) const noexcept = 0;
    [[nodiscard]] virtual std::pair<Float3, Float3> surface_position_and_normal(Expr<float> u, Expr<float3> ps) const noexcept;
    [[nodiscard]] virtual Float3 tangent(Expr<float> u) const noexcept;
};

class PiecewiseLinearCurve final : public CurveInterpolator {

private:
    Float4 _p0;
    Float4 _p1;

public:
    PiecewiseLinearCurve(Expr<float4> q0, Expr<float4> q1) noexcept;
    [[nodiscard]] Float4 position(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 derivative(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 second_derivative(Expr<float> u) const noexcept override;
    [[nodiscard]] std::pair<Float3, Float3> surface_position_and_normal(Expr<float> u, Expr<float3> ps) const noexcept override;
};

class CubicCurve : public CurveInterpolator {

private:
    Float4 _p0;
    Float4 _p1;
    Float4 _p2;
    Float4 _p3;

protected:
    CubicCurve(Float4 p0, Float4 p1, Float4 p2, Float4 p3) noexcept;
    ~CubicCurve() noexcept = default;

public:
    [[nodiscard]] Float4 position(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 derivative(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 second_derivative(Expr<float> u) const noexcept override;
};

class CubicBSplineCurve final : public CubicCurve {
public:
    CubicBSplineCurve(Expr<float4> q0,
                      Expr<float4> q1,
                      Expr<float4> q2,
                      Expr<float4> q3) noexcept;
};

class CatmullRomCurve final : public CubicCurve {
public:
    CatmullRomCurve(Expr<float4> q0,
                    Expr<float4> q1,
                    Expr<float4> q2,
                    Expr<float4> q3) noexcept;
};

}// namespace luisa::compute
