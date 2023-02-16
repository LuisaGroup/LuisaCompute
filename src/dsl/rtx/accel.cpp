#include <dsl/rtx/accel.h>

namespace luisa::compute {

Var<Hit> AccelExprProxy::trace_closest(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{_accel}.trace_closest(ray);
}

Var<bool> AccelExprProxy::trace_any(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{_accel}.trace_any(ray);
}

RayQuery AccelExprProxy::trace_all(Expr<Ray> ray) const noexcept {
    return Expr<Accel>{_accel}.trace_all(ray);
}

Var<float4x4> AccelExprProxy::instance_transform(Expr<int> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_transform(instance_id);
}

Var<float4x4> AccelExprProxy::instance_transform(Expr<uint> instance_id) const noexcept {
    return Expr<Accel>{_accel}.instance_transform(instance_id);
}

void AccelExprProxy::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{_accel}.set_instance_transform(instance_id, mat);
}

void AccelExprProxy::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    Expr<Accel>{_accel}.set_instance_transform(instance_id, mat);
}

void AccelExprProxy::set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
    Expr<Accel>{_accel}.set_instance_visibility(instance_id, vis);
}

void AccelExprProxy::set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
    Expr<Accel>{_accel}.set_instance_visibility(instance_id, vis);
}
void AccelExprProxy::set_instance_opaque(Expr<int> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{_accel}.set_instance_opaque(instance_id, opaque);
}

void AccelExprProxy::set_instance_opaque(Expr<uint> instance_id, Expr<bool> opaque) const noexcept {
    Expr<Accel>{_accel}.set_instance_opaque(instance_id, opaque);
}

Expr<Accel>::Expr(const RefExpr *expr) noexcept
    : _expression{expr} {}

Expr<Accel>::Expr(const Accel &accel) noexcept
    : _expression{detail::FunctionBuilder::current()->accel_binding(
          accel.handle())} {}

Var<Hit> Expr<Accel>::trace_closest(Expr<Ray> ray) const noexcept {
    return def<Hit>(
        detail::FunctionBuilder::current()->call(
            Type::of<Hit>(), CallOp::RAY_TRACING_TRACE_CLOSEST,
            {_expression, ray.expression()}));
}

Var<bool> Expr<Accel>::trace_any(Expr<Ray> ray) const noexcept {
    return def<bool>(
        detail::FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::RAY_TRACING_TRACE_ANY,
            {_expression, ray.expression()}));
}

RayQuery Expr<Accel>::trace_all(Expr<Ray> ray) const noexcept {
    return RayQuery(
        detail::FunctionBuilder::current()->call(
            Type::of<RayQuery>(), CallOp::RAY_TRACING_TRACE_ALL,
            {_expression, ray.expression()}));
}

Var<float4x4> Expr<Accel>::instance_transform(Expr<uint> instance_id) const noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression()}));
}

Var<float4x4> Expr<Accel>::instance_transform(Expr<int> instance_id) const noexcept {
    return def<float4x4>(
        detail::FunctionBuilder::current()->call(
            Type::of<float4x4>(), CallOp::RAY_TRACING_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression()}));
}

void Expr<Accel>::set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}

void Expr<Accel>::set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis.expression()});
}

void Expr<Accel>::set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM,
        {_expression, instance_id.expression(), mat.expression()});
}

void Expr<Accel>::set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
    detail::FunctionBuilder::current()->call(
        CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY,
        {_expression, instance_id.expression(), vis.expression()});
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

}// namespace luisa::compute
