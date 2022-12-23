#pragma once
#ifndef LC_DISABLE_DSL
#include <raster/raster_kernel.h>
#include <ast/function_builder.h>
#include <dsl/var.h>
#include <dsl/struct.h>
namespace luisa::compute {
struct VertexData {
    Float3 position;
    Float3 normal;
    Float4 tangent;
    Float4 color;
    Float2 uv[4];
    UInt vertex_id;
    UInt instance_id;
};
#define LUISA_EXPR(value) \
    detail::extract_expression(std::forward<decltype(value)>(value))
inline VertexData get_vertex_data() {
    VertexData data;
    detail::FunctionBuilder::current()->call(
        CallOp::GET_VERTEX_DATA,
        {LUISA_EXPR(data.position),
         LUISA_EXPR(data.normal),
         LUISA_EXPR(data.tangent),
         LUISA_EXPR(data.color),
         LUISA_EXPR(data.uv[0]),
         LUISA_EXPR(data.uv[1]),
         LUISA_EXPR(data.uv[2]),
         LUISA_EXPR(data.uv[3]),
         LUISA_EXPR(data.vertex_id),
         LUISA_EXPR(data.instance_id)});
    return data;
}
inline auto object_id(){
    return Var<uint>{detail::FunctionBuilder::current()->object_id()};
}
#undef LUISA_EXPR
}// namespace luisa::compute
#endif