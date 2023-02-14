//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <core/dll_export.h>
#include <core/basic_types.h>
#include <core/stl/unordered_map.h>
#include <runtime/rtx/ray.h>
#include <runtime/rtx/mesh.h>
#include <runtime/rtx/procedural_primitive.h>

namespace luisa::compute {
// DSL
class AccelExprProxy;
class LC_RUNTIME_API Accel final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;
    using Modification = AccelBuildCommand::Modification;

private:
    luisa::unordered_map<size_t, Modification> _modifications;
    luisa::vector<uint64_t> _mesh_handles;

private:
    friend class Device;
    friend class Mesh;
    explicit Accel(DeviceInterface *device, const AccelOption &option) noexcept;
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
    // shader functions
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<AccelExprProxy const *>(this);
    }
};
}// namespace luisa::compute
