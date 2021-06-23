//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <dsl/expr.h>
#include <dsl/arg.h>

namespace luisa::compute {

template<typename T>
struct Var : public detail::Expr<T> {

    static_assert(std::is_trivially_destructible_v<T>);

    // for local variables
    template<typename... Args>
    requires concepts::constructible<T, detail::expr_value_t<Args>...>
    Var(Args &&...args)
    noexcept
        : detail::Expr<T>{FunctionBuilder::current()->local(
            Type::of<T>(),
            {detail::extract_expression(std::forward<Args>(args))...})} {}

    // for internal use only...
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<T>{FunctionBuilder::current()->argument(Type::of<T>())} {}

    Var(Var &&) noexcept = default;
    Var(const Var &another) noexcept : Var{detail::Expr{another}} {}
    void operator=(Var &&rhs) noexcept { detail::ExprBase<T>::operator=(rhs); }
    void operator=(const Var &rhs) noexcept { detail::ExprBase<T>::operator=(rhs); }
};

template<typename T>
struct Var<Buffer<T>> : public detail::Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Buffer<T>>{
            FunctionBuilder::current()->buffer(Type::of<Buffer<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

template<typename T>
struct Var<BufferView<T>> : public detail::Expr<Buffer<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Buffer<T>>{
            FunctionBuilder::buffer(Type::of<Buffer<T>>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

template<typename T>
struct Var<Image<T>> : public detail::Expr<Image<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Image<T>>{
            FunctionBuilder::current()->texture(Type::of<Image<T>>()),
            FunctionBuilder::current()->argument(Type::of<uint2>())} {
    }
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

template<typename T>
struct Var<ImageView<T>> : public detail::Expr<Image<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Image<T>>{
            FunctionBuilder::texture(Type::of<Image<T>>()),
            FunctionBuilder::current()->argument(Type::of<uint2>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

template<typename T>
struct Var<Volume<T>> : public detail::Expr<Volume<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Volume<T>>{
            FunctionBuilder::current()->texture(Type::of<Volume<T>>()),
            FunctionBuilder::current()->argument(Type::of<uint3>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

template<typename T>
struct Var<VolumeView<T>> : public detail::Expr<Volume<T>> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Volume<T>>{
            FunctionBuilder::texture(Type::of<Volume<T>>()),
            FunctionBuilder::current()->argument(Type::of<uint3>())} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

template<typename T>
Var(detail::Expr<T>) -> Var<T>;

template<typename T>
Var(T &&) -> Var<T>;

template<typename T, size_t N>
using ArrayVar = Var<std::array<T, N>>;

template<typename T>
using BufferVar = Var<Buffer<T>>;

template<typename T>
using ImageVar = Var<Image<T>>;

template<typename T>
using VolumeVar = Var<Volume<T>>;

using Int = Var<int>;
using Int2 = Var<int2>;
using Int3 = Var<int3>;
using Int4 = Var<int4>;
using UInt = Var<uint>;
using UInt2 = Var<uint2>;
using UInt3 = Var<uint3>;
using UInt4 = Var<uint4>;
using Float = Var<float>;
using Float2 = Var<float2>;
using Float3 = Var<float3>;
using Float4 = Var<float4>;
using Bool = Var<bool>;
using Bool2 = Var<bool2>;
using Bool3 = Var<bool3>;
using Bool4 = Var<bool4>;

template<size_t N>
using ArrayInt = ArrayVar<int, N>;
template<size_t N>
using ArrayInt2 = ArrayVar<int2, N>;
template<size_t N>
using ArrayInt3 = ArrayVar<int3, N>;
template<size_t N>
using ArrayInt4 = ArrayVar<int4, N>;
template<size_t N>
using ArrayUInt = ArrayVar<uint, N>;
template<size_t N>
using ArrayUInt2 = ArrayVar<uint2, N>;
template<size_t N>
using ArrayUInt3 = ArrayVar<uint3, N>;
template<size_t N>
using ArrayUInt4 = ArrayVar<uint4, N>;
template<size_t N>
using ArrayFloat = ArrayVar<float, N>;
template<size_t N>
using ArrayFloat2 = ArrayVar<float2, N>;
template<size_t N>
using ArrayFloat3 = ArrayVar<float3, N>;
template<size_t N>
using ArrayFloat4 = ArrayVar<float4, N>;
template<size_t N>
using ArrayBool = ArrayVar<bool, N>;
template<size_t N>
using ArrayBool2 = ArrayVar<bool2, N>;
template<size_t N>
using ArrayBool3 = ArrayVar<bool3, N>;
template<size_t N>
using ArrayBool4 = ArrayVar<bool4, N>;

using BufferInt = BufferVar<int>;
using BufferInt2 = BufferVar<int2>;
using BufferInt3 = BufferVar<int3>;
using BufferInt4 = BufferVar<int4>;
using BufferUInt = BufferVar<uint>;
using BufferUInt2 = BufferVar<uint2>;
using BufferUInt3 = BufferVar<uint3>;
using BufferUInt4 = BufferVar<uint4>;
using BufferFloat = BufferVar<float>;
using BufferFloat2 = BufferVar<float2>;
using BufferFloat3 = BufferVar<float3>;
using BufferFloat4 = BufferVar<float4>;
using BufferBool = BufferVar<bool>;
using BufferBool2 = BufferVar<bool2>;
using BufferBool3 = BufferVar<bool3>;
using BufferBool4 = BufferVar<bool4>;

using ImageInt = ImageVar<int>;
using ImageInt2 = ImageVar<int2>;
using ImageInt3 = ImageVar<int3>;
using ImageInt4 = ImageVar<int4>;
using ImageUInt = ImageVar<uint>;
using ImageUInt2 = ImageVar<uint2>;
using ImageUInt3 = ImageVar<uint3>;
using ImageUInt4 = ImageVar<uint4>;
using ImageFloat = ImageVar<float>;
using ImageFloat2 = ImageVar<float2>;
using ImageFloat3 = ImageVar<float3>;
using ImageFloat4 = ImageVar<float4>;

using VolumeInt = VolumeVar<int>;
using VolumeInt2 = VolumeVar<int2>;
using VolumeInt3 = VolumeVar<int3>;
using VolumeInt4 = VolumeVar<int4>;
using VolumeUInt = VolumeVar<uint>;
using VolumeUInt2 = VolumeVar<uint2>;
using VolumeUInt3 = VolumeVar<uint3>;
using VolumeUInt4 = VolumeVar<uint4>;
using VolumeFloat = VolumeVar<float>;
using VolumeFloat2 = VolumeVar<float2>;
using VolumeFloat3 = VolumeVar<float3>;
using VolumeFloat4 = VolumeVar<float4>;

}// namespace luisa::compute
