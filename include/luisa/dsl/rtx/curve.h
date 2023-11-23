#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/dsl/var.h>

namespace luisa::compute {

struct LC_DSL_API CurveEvaluation {

    Float3 position;
    Float3 normal;
    Float3 tangent;

    [[nodiscard]] Float h(Expr<float3> w) const noexcept;
    [[nodiscard]] Float v(Expr<float3> w) const noexcept { return h(w) * .5f + .5f; }
};

class LC_DSL_API CurveEvaluator {

public:
    using Evaluation = CurveEvaluation;

public:
    virtual ~CurveEvaluator() noexcept = default;
    [[nodiscard]] virtual Float4 position(Expr<float> u) const noexcept = 0;
    [[nodiscard]] virtual Float4 derivative(Expr<float> u) const noexcept = 0;
    [[nodiscard]] virtual Float4 second_derivative(Expr<float> u) const noexcept = 0;
    [[nodiscard]] virtual Float3 tangent(Expr<float> u) const noexcept;
    [[nodiscard]] virtual CurveEvaluation evaluate(Expr<float> u, Expr<float3> ps) const noexcept;

public:
    template<typename... P>
        requires std::conjunction_v<std::is_same<expr_value_t<P>, float4>...>
    [[nodiscard]] static luisa::unique_ptr<CurveEvaluator> create(CurveBasis basis, P &&...p) noexcept;
};

class LC_DSL_API PiecewiseLinearCurve final : public CurveEvaluator {

private:
    Float4 _p0;
    Float4 _p1;

public:
    PiecewiseLinearCurve(Expr<float4> q0, Expr<float4> q1) noexcept;
    [[nodiscard]] Float4 position(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 derivative(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 second_derivative(Expr<float> u) const noexcept override;
    [[nodiscard]] CurveEvaluation evaluate(Expr<float> u, Expr<float3> ps) const noexcept override;
};

class LC_DSL_API CubicCurve : public CurveEvaluator {

private:
    Float4 _p0;
    Float4 _p1;
    Float4 _p2;
    Float4 _p3;

protected:
    CubicCurve(Float4 p0, Float4 p1, Float4 p2, Float4 p3) noexcept;

public:
    [[nodiscard]] Float4 position(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 derivative(Expr<float> u) const noexcept override;
    [[nodiscard]] Float4 second_derivative(Expr<float> u) const noexcept override;
};

class LC_DSL_API CubicBSplineCurve final : public CubicCurve {
public:
    CubicBSplineCurve(Expr<float4> q0,
                      Expr<float4> q1,
                      Expr<float4> q2,
                      Expr<float4> q3) noexcept;
};

class LC_DSL_API CatmullRomCurve final : public CubicCurve {
public:
    CatmullRomCurve(Expr<float4> q0,
                    Expr<float4> q1,
                    Expr<float4> q2,
                    Expr<float4> q3) noexcept;
};

class LC_DSL_API BezierCurve final : public CubicCurve {

public:
    BezierCurve(Expr<float4> q0,
                Expr<float4> q1,
                Expr<float4> q2,
                Expr<float4> q3) noexcept;
};

template<typename... P>
    requires std::conjunction_v<std::is_same<expr_value_t<P>, float4>...>
luisa::unique_ptr<CurveEvaluator> CurveEvaluator::create(CurveBasis basis, P &&...p) noexcept {
    if constexpr (sizeof...(P) == 2u) {
        switch (basis) {
            case CurveBasis::PIECEWISE_LINEAR:
                return luisa::make_unique<PiecewiseLinearCurve>(std::forward<P>(p)...);
            default: break;
        }
        return nullptr;
    } else if constexpr (sizeof...(P) == 4u) {
        switch (basis) {
            case CurveBasis::CUBIC_BSPLINE:
                return luisa::make_unique<CubicBSplineCurve>(std::forward<P>(p)...);
            case CurveBasis::CATMULL_ROM:
                return luisa::make_unique<CatmullRomCurve>(std::forward<P>(p)...);
            case CurveBasis::BEZIER:
                return luisa::make_unique<BezierCurve>(std::forward<P>(p)...);
            default: break;
        }
        return nullptr;
    } else {
        return nullptr;
    }
}

}// namespace luisa::compute
