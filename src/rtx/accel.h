//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <core/basic_types.h>
#include <runtime/device.h>
#include <runtime/buffer.h>
#include <rtx/ray.h>
#include <rtx/hit.h>
#include <rtx/mesh.h>

namespace luisa::compute {

class Accel : concepts::Noncopyable {

private:
    Device::Handle _device;
    uint64_t _handle;
    bool _built{false};

private:
    friend class Device;
    explicit Accel(Device::Handle device) noexcept;

    void _destroy() noexcept;
    void _mark_dirty() noexcept;
    void _mark_should_rebuild() noexcept;

public:
    ~Accel() noexcept;
    Accel(Accel &&) noexcept = default;
    Accel &operator=(Accel &&rhs) noexcept;
    [[nodiscard]] Command *update(
        size_t first,
        size_t count,
        const float4x4 *transforms) noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] Command *build(
        AccelBuildHint mode,
        std::span<const uint64_t> mesh_handles,
        std::span<const float4x4> transforms) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }

    // shader functions
    [[nodiscard]] detail::Expr<Hit> trace_closest(detail::Expr<Ray> ray) const noexcept;
    [[nodiscard]] detail::Expr<bool> trace_any(detail::Expr<Ray> ray) const noexcept;
};

namespace detail {

template<>
struct Expr<Accel> {

public:
    using ValueType = TextureHeap;

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    explicit Expr(const Accel &accel) noexcept
        : _expression{FunctionBuilder::current()->accel_binding(accel.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto trace_closest(Expr<Ray> ray) const noexcept {
        return Expr<Hit>{FunctionBuilder::current()->call(
            Type::of<Hit>(), CallOp::TRACE_CLOSEST,
            {_expression, ray.expression()})};
    }
    [[nodiscard]] auto trace_any(Expr<Ray> ray) const noexcept {
        return Expr<bool>{FunctionBuilder::current()->call(
            Type::of<bool>(), CallOp::TRACE_ANY,
            {_expression, ray.expression()})};
    }
};

}// namespace detail

template<>
struct Var<Accel> : public detail::Expr<Accel> {
    explicit Var(detail::ArgumentCreation) noexcept
        : detail::Expr<Accel>{
            detail::FunctionBuilder::current()->accel()} {}
    Var(Var &&) noexcept = default;
    Var(const Var &) noexcept = delete;
    Var &operator=(Var &&) noexcept = delete;
    Var &operator=(const Var &) noexcept = delete;
};

using AccelVar = Var<Accel>;

}// namespace luisa::compute
