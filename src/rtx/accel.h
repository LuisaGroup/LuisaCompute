//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <core/basic_types.h>
#include <core/stl.h>
#include <dsl/expr.h>
#include <rtx/ray.h>
#include <rtx/hit.h>
#include <rtx/mesh.h>

namespace luisa::compute {

class LC_RTX_API Accel final : public Resource {

public:
    using UsageHint = AccelUsageHint;
    using BuildRequest = AccelBuildRequest;
    using Modification = AccelBuildCommand::Modification;

private:
    luisa::unordered_map<size_t, Modification> _modifications;
    luisa::vector<uint64_t> _mesh_handles;
    luisa::unique_ptr<std::mutex> _mutex;

private:
    friend class Device;
    friend class Mesh;
    explicit Accel(Device::Interface *device, UsageHint hint = UsageHint::FAST_TRACE) noexcept;

public:
    Accel() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _mesh_handles.size(); }

    // host interfaces
    void _emplace_back(uint64_t mesh_handle, float4x4 transform = make_float4x4(1.f), bool visible = true) noexcept;
    void emplace_back(Mesh const &mesh, float4x4 transform = make_float4x4(1.f), bool visible = true) noexcept;
    void _set(size_t index, uint64_t mesh_handle, float4x4 transform = make_float4x4(1.f), bool visible = true) noexcept;
    void set(size_t index, const Mesh &mesh, float4x4 transform = make_float4x4(1.f), bool visible = true) noexcept;
    void pop_back() noexcept;
    void set_transform_on_update(size_t index, float4x4 transform) noexcept;
    void set_visibility_on_update(size_t index, bool visible) noexcept;
    [[nodiscard]] Command *build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;

    // shader functions
    [[nodiscard]] Var<Hit> trace_closest(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept;
};

template<>
struct Expr<Accel> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept
        : _expression{expr} {}
    explicit Expr(const Accel &accel) noexcept
        : _expression{detail::FunctionBuilder::current()->accel_binding(
              accel.handle())} {}
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
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression(), mat.expression()});
    }
    void set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_VISIBILITY,
            {_expression, instance_id.expression(), vis.expression()});
    }
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept {
        detail::FunctionBuilder::current()->call(
            CallOp::SET_INSTANCE_TRANSFORM,
            {_expression, instance_id.expression(), mat.expression()});
    }
    void set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept {
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
