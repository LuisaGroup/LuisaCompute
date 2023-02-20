//
// Created by Mike Smith on 2023/2/18.
//

#include <dsl/builtin.h>
#include <dsl/stmt.h>
#include <dsl/resource.h>

namespace luisa::compute::detail {

Var<float4> BindlessTexture2D::sample(Expr<float2> uv) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE2D_SAMPLE,
        {_array, _index, uv.expression()}));
}

Var<float4> BindlessTexture2D::sample(Expr<float2> uv, Expr<float> mip) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL,
        {_array, _index, uv.expression(), mip.expression()}));
}

Var<float4> BindlessTexture2D::sample(Expr<float2> uv, Expr<float2> dpdx, Expr<float2> dpdy) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD,
        {_array, _index, uv.expression(), dpdx.expression(), dpdy.expression()}));
}

Var<uint2> BindlessTexture2D::size() const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<uint2>(f->call(
        Type::of<uint2>(), CallOp::BINDLESS_TEXTURE2D_SIZE,
        {_array, _index}));
}

Var<uint2> BindlessTexture2D::size(Expr<int> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<uint2>(f->call(
        Type::of<uint2>(), CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL,
        {_array, _index, level.expression()}));
}

Var<uint2> BindlessTexture2D::size(Expr<uint> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<uint2>(f->call(
        Type::of<uint2>(), CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL,
        {_array, _index, level.expression()}));
}

Var<float4> BindlessTexture2D::read(Expr<uint2> coord) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE2D_READ,
        {_array, _index, coord.expression()}));
}

Var<float4> BindlessTexture3D::sample(Expr<float3> uvw) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_SAMPLE,
        {_array, _index, uvw.expression()}));
}

Var<float4> BindlessTexture3D::sample(Expr<float3> uvw, Expr<float> mip) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL,
        {_array, _index, uvw.expression(), mip.expression()}));
}

Var<float4> BindlessTexture3D::sample(Expr<float3> uvw, Expr<float3> dpdx, Expr<float3> dpdy) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD,
        {_array, _index, uvw.expression(), dpdx.expression(), dpdy.expression()}));
}

Var<uint3> BindlessTexture3D::size() const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<uint3>(f->call(
        Type::of<uint3>(), CallOp::BINDLESS_TEXTURE3D_SIZE,
        {_array, _index}));
}

Var<uint3> BindlessTexture3D::size(Expr<int> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<uint3>(f->call(
        Type::of<uint3>(), CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL,
        {_array, _index, level.expression()}));
}

Var<uint3> BindlessTexture3D::size(Expr<uint> level) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<uint3>(f->call(
        Type::of<uint3>(), CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL,
        {_array, _index, level.expression()}));
}

Var<float4> BindlessTexture3D::read(Expr<uint3> coord) const noexcept {
    auto f = detail::FunctionBuilder::current();
    return def<float4>(f->call(
        Type::of<float4>(), CallOp::BINDLESS_TEXTURE3D_READ,
        {_array, _index, coord.expression()}));
}

}// namespace luisa::compute::detail
