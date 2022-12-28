//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <core/dll_export.h>
#include <core/basic_types.h>

#ifndef LC_DISABLE_DSL
#include <dsl/expr.h>
#endif

#include <runtime/custom_pass.h>
#include <rtx/ray.h>
#include <rtx/ray_query.h>
#include <rtx/mesh.h>
#include <rtx/procedural_primitive.h>

namespace luisa::compute {

class LC_RUNTIME_API Accel final : public Resource {

public:
    using UsageHint = AccelUsageHint;
    using BuildRequest = AccelBuildRequest;
    using Modification = AccelBuildCommand::Modification;

private:
    luisa::unordered_map<size_t, Modification> _modifications;
    luisa::vector<uint64_t> _mesh_handles;

private:
    friend class Device;
    friend class Mesh;
    explicit Accel(DeviceInterface *device, AccelBuildOption const &option) noexcept;
    luisa::unique_ptr<Command> update(bool build_accel, Accel::BuildRequest request) noexcept;

public:
    Accel() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _mesh_handles.size(); }

    // host interfaces
    void emplace_back(const Mesh &mesh, float4x4 transform = make_float4x4(1.f), bool visible = true, bool opaque = true) noexcept {
        emplace_back_handle(mesh.handle(), transform, visible, opaque);
    }
    void emplace_back(const ProceduralPrimitive &prim, float4x4 transform = make_float4x4(1.f), bool visible = true, bool opaque = true) noexcept {
        emplace_back_handle(prim.handle(), transform, visible, opaque);
    }
    void emplace_back_handle(uint64_t mesh_handle, float4x4 const &transform, bool visible, bool opaque) noexcept;
    void set(size_t index, const Mesh &mesh, float4x4 transform = make_float4x4(1.f), bool visible = true, bool opaque = true) noexcept {
        set_handle(index, mesh.handle(), transform, visible, opaque);
    }
    void set(size_t index, const ProceduralPrimitive &prim, float4x4 transform = make_float4x4(1.f), bool visible = true, bool opaque = true) noexcept {
        set_handle(index, prim.handle(), transform, visible, opaque);
    }
    void set_handle(size_t index, uint64_t mesh_handle, float4x4 const &transform, bool visible, bool opaque) noexcept;
    void pop_back() noexcept;
    void set_transform_on_update(size_t index, float4x4 transform) noexcept;
    void set_visibility_on_update(size_t index, bool visible) noexcept;
    void set_opaque_on_update(size_t index, bool opaque) noexcept;
    // update top-level accel, build or only update instance data
    [[nodiscard]] luisa::unique_ptr<Command> update_instance() noexcept {
        return update(false, Accel::BuildRequest::PREFER_UPDATE);
    }
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept {
        return update(true, request);
    }
#ifndef LC_DISABLE_DSL
    // shader functions
    [[nodiscard]] Var<Hit> trace_closest(Expr<Ray> ray) const noexcept;
    [[nodiscard]] RayQuery trace_all(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> vis) const noexcept;
#endif
};

#ifndef LC_DISABLE_DSL
template<>
struct LC_RUNTIME_API Expr<Accel> {

private:
    const RefExpr *_expression{nullptr};

public:
    explicit Expr(const RefExpr *expr) noexcept;
    Expr(const Accel &accel) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] Var<Hit> trace_closest(Expr<Ray> ray) const noexcept;
    [[nodiscard]] RayQuery trace_all(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<bool> trace_any(Expr<Ray> ray) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<uint> instance_id) const noexcept;
    [[nodiscard]] Var<float4x4> instance_transform(Expr<int> instance_id) const noexcept;
    void set_instance_transform(Expr<int> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<int> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_opaque(Expr<int> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_transform(Expr<uint> instance_id, Expr<float4x4> mat) const noexcept;
    void set_instance_visibility(Expr<uint> instance_id, Expr<bool> vis) const noexcept;
    void set_instance_opaque(Expr<uint> instance_id, Expr<bool> vis) const noexcept;
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
#endif

namespace custompass_detail {

template<>
struct CustomResFilter<Accel> {
    static constexpr bool LegalType = true;
    static void emplace(luisa::string &&name, Usage usage, CustomPass *cmd, Accel const &v) {
        CustomCommand::ResourceBinding bindings;
        bindings.name = std::move(name);
        bindings.usage = usage;
        bindings.resource_view = CustomCommand::AccelView{
            .handle = v.handle()};
        cmd->_bindings.emplace_back(std::move(bindings));
    }
};

}// namespace custompass_detail

}// namespace luisa::compute
