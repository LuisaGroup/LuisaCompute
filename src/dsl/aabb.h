#pragma once

#ifndef LC_DISABLE_DSL
#include <dsl/expr.h>
#include <dsl/var.h>
#include <dsl/struct.h>
#include <runtime/custom_struct.h>
LUISA_CUSTOM_STRUCT(AABB);
namespace luisa::compute {
inline void set_aabb(Expr<Buffer<AABB>> buffer, Expr<uint> idx, Expr<float3> min_axis, Expr<float3> max_axis) {
    detail::FunctionBuilder::current()->call(
        CallOp::SET_AABB,
        {buffer.expression(), idx.expression(), min_axis.expression(), max_axis.expression()});
}
struct AABBValue {
    Expr<float3> min_axis;
    Expr<float3> max_axis;
};
inline AABBValue get_aabb(Expr<Buffer<AABB>> buffer, Expr<uint> idx) {
    Var<float3> min_axis, max_axis;
    detail::FunctionBuilder::current()->call(
        CallOp::GET_AABB,
        {buffer.expression(), idx.expression(), min_axis.expression(), max_axis.expression()});
    return {min_axis, max_axis};
}
}// namespace luisa::compute
#endif