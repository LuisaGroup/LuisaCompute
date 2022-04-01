//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl.h>
#include <core/observer.h>
#include <dsl/expr.h>
#include <rtx/ray.h>
#include <rtx/hit.h>
#include <rtx/mesh.h>

namespace luisa::compute {

class LC_RTX_API Accel final : public Resource {

public:
    class RebuildObserver : public Observer {

    private:
        bool _requires_rebuild{true};

    public:
        RebuildObserver() noexcept = default;
        void clear() noexcept { _requires_rebuild = false; }
        void notify() noexcept override { _requires_rebuild = true; }
        [[nodiscard]] auto requires_rebuild() const noexcept { return _requires_rebuild; }
    };

private:
    luisa::unique_ptr<RebuildObserver> _rebuild_observer;

private:
    friend class Device;
    friend class Mesh;
    explicit Accel(Device::Interface *device, AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept;

public:
    Accel() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _rebuild_observer->size(); }
    Accel &emplace_back(Mesh const &mesh, float4x4 transform = make_float4x4(1.f), bool visible = true) noexcept;
    Accel &set(size_t index, const Mesh &mesh, float4x4 transform = make_float4x4(1.f), bool visible = true) noexcept;
    Accel &pop_back() noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] Command *build() noexcept;

    // shader functions
    [[nodiscard]] Var<Hit> trace_closest(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    void set_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept;
    void set_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept;
};

template<>
struct Expr<Accel> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    explicit Expr(const Accel &accel) noexcept
        : _expression{detail::FunctionBuilder::current()->accel_binding(accel.handle())} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto trace_closest(Expr<Ray> ray) const noexcept {
        return def<Hit>(
            detail::FunctionBuilder::current()->call(
                Type::of<Hit>(), CallOp::TRACE_CLOSEST,
                {_expression, ray.expression()}));
    }
    [[nodiscard]] auto trace_any(Expr<Ray> ray) const noexcept {
        return def<bool>(
            detail::FunctionBuilder::current()->call(
                Type::of<bool>(), CallOp::TRACE_ANY,
                {_expression, ray.expression()}));
    }
    [[nodiscard]] auto instance_transform(Expr<uint> instance_id) const noexcept {
        return def<float4x4>(
            detail::FunctionBuilder::current()->call(
                Type::of<float4x4>(), CallOp::INSTANCE_TO_WORLD_MATRIX,
                {_expression, instance_id.expression()}));
    }
    [[nodiscard]] auto instance_transform(Expr<int> instance_id) const noexcept {
        return def<float4x4>(
            detail::FunctionBuilder::current()->call(
                Type::of<float4x4>(), CallOp::INSTANCE_TO_WORLD_MATRIX,
                {_expression, instance_id.expression()}));
    }
    void set_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression(), mat.expression()});
    }
    void set_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_VISIBILITY,
            {_expression, instance_id.expression(), vis.expression()});
    }
    void set_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression(), mat.expression()});
    }
    void set_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_VISIBILITY,
            {_expression, instance_id.expression(), vis.expression()});
    }
};

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

}// namespace luisa::compute
