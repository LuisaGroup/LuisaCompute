//
// Created by Mike Smith on 2021/4/12.
//

#include <dsl/syntax.h>

namespace luisa::compute {

Expr<float4> HeapTexture2D::sample(Expr<float2> uv) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D,
        {_heap, _index, uv.expression()}));
}

Expr<float4> HeapTexture2D::sample(Expr<float2> uv, Expr<float> mip) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_LEVEL,
        {_heap, _index, uv.expression(), mip.expression()}));
}

Expr<float4> HeapTexture2D::sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE2D_GRAD,
        {_heap, _index, uv.expression(), dpdx.expression(), dpdy.expression()}));
}

Expr<uint2> HeapTexture2D::size() const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<uint2>(f->call(
        Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D,
        {_heap, _index}));
}

Expr<uint2> HeapTexture2D::size(Expr<int> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<uint2>(f->call(
        Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
        {_heap, _index, level.expression()}));
}

Expr<uint2> HeapTexture2D::size(Expr<uint> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<uint2>(f->call(
        Type::of<uint2>(), CallOp::TEXTURE_HEAP_SIZE2D_LEVEL,
        {_heap, _index, level.expression()}));
}

Expr<float4> HeapTexture2D::read(Expr<uint2> coord) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_READ2D,
        {_heap, _index, coord.expression()}));
}

Expr<float4> HeapTexture3D::sample(Expr<float3> uvw) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D,
        {_heap, _index, uvw.expression()}));
}

Expr<float4> HeapTexture3D::sample(Expr<float3> uvw, Expr<float> mip) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_LEVEL,
        {_heap, _index, uvw.expression(), mip.expression()}));
}

Expr<float4> HeapTexture3D::sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_SAMPLE3D_GRAD,
        {_heap, _index, uvw.expression(), dpdx.expression(), dpdy.expression()}));
}

Expr<uint3> HeapTexture3D::size() const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<uint3>(f->call(
        Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D,
        {_heap, _index}));
}

Expr<uint3> HeapTexture3D::size(Expr<int> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<uint3>(f->call(
        Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
        {_heap, _index, level.expression()}));
}

Expr<uint3> HeapTexture3D::size(Expr<uint> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<uint3>(f->call(
        Type::of<uint3>(), CallOp::TEXTURE_HEAP_SIZE3D_LEVEL,
        {_heap, _index, level.expression()}));
}

Expr<float4> HeapTexture3D::read(Expr<uint3> coord) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return detail::make_var_expr<float4>(f->call(
        Type::of<float4>(), CallOp::TEXTURE_HEAP_READ3D,
        {_heap, _index, coord.expression()}));
}

}// namespace luisa::compute