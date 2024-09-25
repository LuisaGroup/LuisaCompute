#include <luisa/dsl/stmt.h>
#include <luisa/dsl/rtx/accel.h>

namespace luisa::compute {

namespace detail {

Var<TriangleHit> AccelExprProxy::trace_closest(Expr<Ray> ray, Expr<uint> vis_mask) const noexcept {
    return intersect(ray, {.visibility_mask = vis_mask});
}

Var<bool> AccelExprProxy::trace_any(Expr<Ray> ray, Expr<uint> vis_mask) const noexcept {
    return intersect_any(ray, {.visibility_mask = vis_mask});
}

RayQueryAll AccelExprProxy::query_all(Expr<Ray> ray, Expr<uint> vis_mask) const noexcept {
    return traverse(ray, {.visibility_mask = vis_mask});
}

RayQueryAny AccelExprProxy::query_any(Expr<Ray> ray, Expr<uint> vis_mask) const noexcept {
    return traverse_any(ray, {.visibility_mask = vis_mask});
}

Var<TriangleHit> AccelExprProxy::intersect(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.intersect(ray, options);
}

Var<bool> AccelExprProxy::intersect_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.intersect_any(ray, options);
}

RayQueryAll AccelExprProxy::traverse(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.traverse(ray, options);
}

RayQueryAny AccelExprProxy::traverse_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.traverse_any(ray, options);
}

Var<SurfaceHit> AccelExprProxy::intersect_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.intersect_motion(ray, time, options);
}

Var<bool> AccelExprProxy::intersect_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.intersect_any_motion(ray, time, options);
}

RayQueryAll AccelExprProxy::traverse_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.traverse_motion(ray, time, options);
}

RayQueryAny AccelExprProxy::traverse_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    return Expr<Accel>{_accel}.traverse_any_motion(ray, time, options);
}

Var<float4x4> AccelExprProxy::instance_transform(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_transform(instance_id);
}

Var<float4x4> AccelExprProxy::instance_transform(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_transform(instance_id);
}

Var<uint> AccelExprProxy::instance_user_id(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_user_id(instance_id);
}

Var<uint> AccelExprProxy::instance_user_id(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_user_id(instance_id);
}
Var<uint> AccelExprProxy::instance_visibility_mask(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_visibility_mask(instance_id);
}

Var<uint> AccelExprProxy::instance_visibility_mask(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_visibility_mask(instance_id);
}

void AccelExprProxy::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{_accel}.set_instance_transform(instance_id, mat);
}

void AccelExprProxy::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{_accel}.set_instance_transform(instance_id, mat);
}

void AccelExprProxy::set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept {
    Expr<Accel>{_accel}.set_instance_visibility(instance_id, vis_mask);
}

void AccelExprProxy::set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept {
    Expr<Accel>{_accel}.set_instance_visibility(instance_id, vis_mask);
}

void AccelExprProxy::set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{_accel}.set_instance_opaque(instance_id, opaque);
}

void AccelExprProxy::set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{_accel}.set_instance_opaque(instance_id, opaque);
}

void AccelExprProxy::set_instance_user_id(Expr<int> instance_id, Expr<uint> id) const noexcept {
    Expr<Accel>{_accel}.set_instance_user_id(instance_id, id);
}

void AccelExprProxy::set_instance_user_id(Expr<uint> instance_id, Expr<uint> id) const noexcept {
    Expr<Accel>{_accel}.set_instance_user_id(instance_id, id);
}

}// namespace detail

Expr<Accel>::Expr(const RefExpr *expr) noexcept
    : _expression{expr} {}

Expr<Accel>::Expr(const Accel &accel) noexcept
    : _expression{detail::FunctionBuilder::current()->accel_binding(
          accel.handle())} {}

Var<TriangleHit> Expr<Accel>::trace_closest(Expr<Ray> ray, Expr<uint> mask) const noexcept {
    return intersect(ray, {.visibility_mask = mask});
}

Var<bool> Expr<Accel>::trace_any(Expr<Ray> ray, Expr<uint> mask) const noexcept {
    return intersect_any(ray, {.visibility_mask = mask});
}

RayQueryAll Expr<Accel>::query_all(Expr<Ray> ray, Expr<uint> mask) const noexcept {
    return traverse(ray, {.visibility_mask = mask});
}

RayQueryAny Expr<Accel>::query_any(Expr<Ray> ray, Expr<uint> mask) const noexcept {
    return traverse_any(ray, {.visibility_mask = mask});
}

Var<TriangleHit> Expr<Accel>::intersect(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return def<TriangleHit>(
        detail::FunctionBuilder::current()->call(
            Type::of<TriangleHit>(), CallOp::RAY_TRACING_TRACE_CLOSEST,
            {_expression, ray.expression(), options.visibility_mask.expression()}));
}

Var<bool> Expr<Accel>::intersect_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::RAY_TRACING_TRACE_ANY,
            {_expression, ray.expression(), options.visibility_mask.expression()}));
}

RayQueryAll Expr<Accel>::traverse(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return {_expression, ray.expression(), options.visibility_mask.expression()};
}

RayQueryAny Expr<Accel>::traverse_any(Expr<Ray> ray, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return {_expression, ray.expression(), options.visibility_mask.expression()};
}

Var<SurfaceHit> Expr<Accel>::intersect_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return def<TriangleHit>(
        detail::FunctionBuilder::current()->call(
            Type::of<TriangleHit>(), CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR,
            {_expression, ray.expression(), time.expression(), options.visibility_mask.expression()}));
}

Var<bool> Expr<Accel>::intersect_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR,
            {_expression, ray.expression(), time.expression(), options.visibility_mask.expression()}));
}

RayQueryAll Expr<Accel>::traverse_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return {_expression, ray.expression(), time.expression(), options.visibility_mask.expression()};
}

RayQueryAny Expr<Accel>::traverse_any_motion(Expr<Ray> ray, Expr<float> time, const AccelTraceOptions &options) const noexcept {
    require_curve_basis_set(options.curve_bases);
    return {_expression, ray.expression(), time.expression(), options.visibility_mask.expression()};
}

Var<float4x4> Expr<Accel>::instance_transform(Expr<uint> instance_id) const noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression()}));
}

Var<uint> Expr<Accel>::instance_user_id(Expr<uint> instance_id) const noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::RAY_TRACING_INSTANCE_USER_ID,
            {_expression, instance_id.expression()}));
}

Var<uint> Expr<Accel>::instance_visibility_mask(Expr<uint> instance_id) const noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK,
            {_expression, instance_id.expression()}));
}

Var<float4x4> Expr<Accel>::instance_transform(Expr<int> instance_id) const noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression()}));
}

Var<uint> Expr<Accel>::instance_user_id(Expr<int> instance_id) const noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::RAY_TRACING_INSTANCE_USER_ID,
            {_expression, instance_id.expression()}));
}

Var<uint> Expr<Accel>::instance_visibility_mask(Expr<int> instance_id) const noexcept {
    return def<uint>(
        detail::FunctionBuilder::current()->call(
            Type::of<uint>(), CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK,
            {_expression, instance_id.expression()}));
}

void Expr<Accel>::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}

void Expr<Accel>::set_instance_visibility(Expr<int> instance_id, Expr<uint> vis_mask) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis_mask.expression()});
}

void Expr<Accel>::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}

void Expr<Accel>::set_instance_visibility(Expr<uint> instance_id, Expr<uint> vis_mask) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis_mask.expression()});
}

void Expr<Accel>::set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_OPACITY,
        {_expression, instance_id.expression(), opaque.expression()});
}

void Expr<Accel>::set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_OPACITY,
        {_expression, instance_id.expression(), opaque.expression()});
}
void Expr<Accel>::set_instance_user_id(Expr<int> instance_id, Expr<uint> id) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_USER_ID,
        {_expression, instance_id.expression(), id.expression()});
}

void Expr<Accel>::set_instance_user_id(Expr<uint> instance_id, Expr<uint> id) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_USER_ID,
        {_expression, instance_id.expression(), id.expression()});
}

}// namespace luisa::compute
