#pragma once

#include <luisa/runtime/rtx/accel.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/dsl/rtx/hit.h>
#include <luisa/dsl/rtx/ray.h>
#include <luisa/dsl/rtx/motion.h>

namespace luisa::compute {

struct AccelTraceOptions {
    CurveBasisSet curve_bases{CurveBasisSet::make_none()};
    UInt visibility_mask{0xffu};
};

#define LUISA_ACCEL_TRACE_DEPRECATED                                                                 \
    [[deprecated(                                                                                    \
        "\n\n"                                                                                       \
        "Accel::trace_*(ray, vis_mask) and query_*(ray, vis_mask) are deprecated.\n"                 \
        "Please use Accel::intersect_*/traverse_*(ray, const AccelTraceOptions &options) instead.\n" \
        "\n"                                                                                         \
        "Note: curve tracing is disabled by default for performance reasons. If you would\n"         \
        "      like to enable it, please specify the required curve bases in the options.\n"         \
        "\n")]]

template<>
struct LC_DSL_API Expr<Accel> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept;
    explicit Expr(const Accel &accel) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<TriangleHit> trace_closest(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAll query_all(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAny query_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;

    [[nodiscard]] Var<SurfaceHit> intersect(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<bool> intersect_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAll traverse(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAny traverse_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;

    // motion blur versions: note that these overloads have significant overhead even if the accel is not actually built with motion blur
    [[nodiscard]] Var<SurfaceHit> intersect_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<bool> intersect_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAll traverse_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAny traverse_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;

    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<int> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_user_id(Expr<int> instance_id, Expr<uint> id) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_user_id(Expr<uint> instance_id, Expr<uint> id) const noexcept;

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    [[nodiscard]] auto instance_motion_matrix(I0 &&instance_id, I1 &&keyframe_id) const noexcept {
        auto expr_inst_id = detail::extract_expression(std::forward<I0>(instance_id));
        auto expr_keyframe_id = detail::extract_expression(std::forward<I1>(keyframe_id));
        return def<float4x4>(
            detail::FunctionBuilder::current()->call(
                Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX,
                {_expression, expr_inst_id, expr_keyframe_id}));
    }

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    [[nodiscard]] auto instance_motion_srt(I0 &&instance_id, I1 &&keyframe_id) const noexcept {
        auto expr_inst_id = detail::extract_expression(std::forward<I0>(instance_id));
        auto expr_keyframe_id = detail::extract_expression(std::forward<I1>(keyframe_id));
        return def<MotionInstanceTransformSRT>(
            detail::FunctionBuilder::current()->call(
                Type::of<MotionInstanceTransformSRT>(), CallOp::RAY_TRACING_INSTANCE_MOTION_SRT,
                {_expression, expr_inst_id, expr_keyframe_id}));
    }

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    void set_instance_motion_matrix(I0 &&instance_id, I1 &&keyframe_id, Expr<float4x4> m) const noexcept {
        auto expr_inst_id = detail::extract_expression(std::forward<I0>(instance_id));
        auto expr_keyframe_id = detail::extract_expression(std::forward<I1>(keyframe_id));
        auto expr_m = m.expression();
        detail::FunctionBuilder::current()->call(
            CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX,
            {_expression, expr_inst_id, expr_keyframe_id, expr_m});
    }

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    void set_instance_motion_srt(I0 &&instance_id, I1 &&keyframe_id, Expr<MotionInstanceTransformSRT> srt) const noexcept {
        auto expr_inst_id = detail::extract_expression(std::forward<I0>(instance_id));
        auto expr_keyframe_id = detail::extract_expression(std::forward<I1>(keyframe_id));
        auto expr_srt = srt.expression();
        detail::FunctionBuilder::current()->call(
            CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT,
            {_expression, expr_inst_id, expr_keyframe_id, expr_srt});
    }

    [[nodiscard]] auto operator->() const noexcept { return this; }
};

Expr(const Accel &) noexcept -> Expr<Accel>;

template<>
struct Var<Accel> : public Expr<Accel> {
    explicit Var(detail::ArgumentCreation) noexcept
        : Expr<Accel>{detail::FunctionBuilder::current()->accel()} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

using AccelVar = Var<Accel>;

namespace detail {

class LC_DSL_API AccelExprProxy {

private:
    Accel _accel;

public:
    LUISA_RESOURCE_PROXY_AVOID_CONSTRUCTION(AccelExprProxy)

public:
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<TriangleHit> trace_closest(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAll query_all(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;
    LUISA_ACCEL_TRACE_DEPRECATED [[nodiscard]] RayQueryAny query_any(Expr<Ray> ray, Expr<uint> vis_mask = 0xffu) const noexcept;

    [[nodiscard]] Var<SurfaceHit> intersect(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<bool> intersect_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAll traverse(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAny traverse_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept;

    // motion blur versions: note that these overloads have significant overhead even if the accel is not actually built with motion blur
    [[nodiscard]] Var<SurfaceHit> intersect_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] Var<bool> intersect_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAll traverse_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;
    [[nodiscard]] RayQueryAny traverse_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept;

    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_user_id(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<uint> instance_visibility_mask(Expr<int> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept;
    void set_instance_user_id(Expr<int> instance_id, Expr<uint> id) const noexcept;
    void set_instance_user_id(Expr<uint> instance_id, Expr<uint> id) const noexcept;

    // motion blur support
    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    [[nodiscard]] auto instance_motion_matrix(I0 &&instance_id, I1 &&keyframe_id) const noexcept {
        return Expr<Accel>{_accel}.instance_motion_matrix(
            std::forward<I0>(instance_id),
            std::forward<I1>(keyframe_id));
    }

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    [[nodiscard]] auto instance_motion_srt(I0 &&instance_id, I1 &&keyframe_id) const noexcept {
        return Expr<Accel>{_accel}.instance_motion_srt(
            std::forward<I0>(instance_id),
            std::forward<I1>(keyframe_id));
    }

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    void set_instance_motion_matrix(I0 &&instance_id, I1 &&keyframe_id, Expr<float4x4> m) const noexcept {
        Expr<Accel>{_accel}.set_instance_motion_matrix(
            std::forward<I0>(instance_id),
            std::forward<I1>(keyframe_id), m);
    }

    template<typename I0, typename I1>
        requires is_integral_expr_v<I0> && is_integral_expr_v<I1>
    void set_instance_motion_srt(I0 &&instance_id, I1 &&keyframe_id, Expr<MotionInstanceTransformSRT> srt) const noexcept {
        Expr<Accel>{_accel}.set_instance_motion_srt(
            std::forward<I0>(instance_id),
            std::forward<I1>(keyframe_id), srt);
    }
};

}// namespace detail

#undef LUISA_ACCEL_TRACE_DEPRECATED

}// namespace luisa::compute
