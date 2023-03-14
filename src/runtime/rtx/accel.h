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

namespace detail {
// for DSL
class AccelExprProxy;
}// namespace detail

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
    Accel(Accel &&) noexcept = default;
    Accel(Accel const &) noexcept = delete;
    Accel &operator=(Accel &&) noexcept = default;
    Accel &operator=(Accel const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _mesh_handles.size(); }

    // host interfaces
    // operations is committed by update_instance() or build()
    void emplace_back(const Mesh &mesh,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true) noexcept {
        emplace_back_handle(mesh.handle(), transform, visibility_mask, opaque);
    }

    void emplace_back(const ProceduralPrimitive &prim,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true) noexcept {
        emplace_back_handle(prim.handle(), transform, visibility_mask, opaque);
    }

    void emplace_back_handle(uint64_t mesh_handle,
                             float4x4 const &transform,
                             uint8_t visibility_mask,
                             bool opaque) noexcept;

    void set(size_t index, const Mesh &mesh,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true) noexcept {
        set_handle(index, mesh.handle(), transform, visibility_mask, opaque);
    }

    void set(size_t index, const ProceduralPrimitive &prim,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true) noexcept {
        set_handle(index, prim.handle(), transform, visibility_mask, opaque);
    }

    void set_handle(size_t index, uint64_t mesh_handle,
                    float4x4 const &transform,
                    uint8_t visibility_mask, bool opaque) noexcept;

    void pop_back() noexcept;
    void set_transform_on_update(size_t index, float4x4 transform) noexcept;
    void set_visibility_on_update(size_t index, uint8_t visibility_mask) noexcept;
    void set_opaque_on_update(size_t index, bool opaque) noexcept;

    // update top-level acceleration structure's instance data without build
    [[nodiscard]] luisa::unique_ptr<Command> update_instance() noexcept {
        return update(false, Accel::BuildRequest::PREFER_UPDATE);
    }
    // update top-level acceleration structure's instance data and build
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept {
        return update(true, request);
    }

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::AccelExprProxy *>(this);
    }
};

}// namespace luisa::compute
