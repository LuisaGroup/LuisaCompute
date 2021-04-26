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
    Var(Args &&...args) noexcept
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

}// namespace luisa::compute
