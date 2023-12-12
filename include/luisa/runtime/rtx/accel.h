#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/curve.h>
#include <luisa/runtime/rtx/procedural_primitive.h>

namespace luisa::compute {

namespace detail {
// for DSL
class AccelExprProxy;
}// namespace detail

// Accel is top-level acceleration structure(TLAS) for ray-tracing
class LC_RUNTIME_API Accel final : public Resource {
    friend class ManagedAccel;

public:
    using BuildRequest = AccelBuildRequest;
    using Modification = AccelBuildCommand::Modification;

private:
    luisa::unordered_map<size_t, Modification> _modifications;
    mutable luisa::spin_mutex _mtx;
    size_t _instance_count{};


private:
    friend class Device;
    friend class Mesh;
    explicit Accel(DeviceInterface *device, const AccelOption &option) noexcept;
    luisa::unique_ptr<Command> _build(Accel::BuildRequest request,
                                      bool update_instance_buffer_only) noexcept;

public:
    Accel() noexcept = default;
    ~Accel() noexcept override;
    Accel(Accel &&) noexcept;
    Accel(Accel const &) noexcept = delete;
    Accel &operator=(Accel &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Accel &operator=(Accel const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        std::lock_guard lock{_mtx};
        return _instance_count;
    }
    void emplace_back_handle(uint64_t mesh_handle,
                              float4x4 const &transform,
                              uint8_t visibility_mask,
                              bool opaque,
                              uint user_id) noexcept;
    void set_handle(size_t index, uint64_t mesh_handle,
                     float4x4 const &transform,
                     uint8_t visibility_mask, bool opaque, uint user_id) noexcept;
    void set_prim_handle(size_t index, uint64_t prim_handle) noexcept;

    // host interfaces
    // operations is committed by update_instance() or build()
    void emplace_back(const Mesh &mesh,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true,
                      uint user_id = 0) noexcept {
        emplace_back_handle(mesh.handle(), transform, visibility_mask, opaque, user_id);
    }

    void emplace_back(const Curve &curve,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true,
                      uint user_id = 0) noexcept {
        emplace_back_handle(curve.handle(), transform, visibility_mask, opaque, user_id);
    }

    void emplace_back(const ProceduralPrimitive &prim,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      uint user_id = 0) noexcept {
        emplace_back_handle(prim.handle(), transform, visibility_mask,
                             false /* procedural geometry is always non-opaque */,
                             user_id);
    }

    void emplace_back(uint64_t mesh_handle,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true,
                      uint user_id = 0) noexcept {
        emplace_back_handle(mesh_handle, transform, visibility_mask, opaque, user_id);
    }

    void set(size_t index, const Mesh &mesh,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true,
             uint user_id = 0) noexcept {
        set_handle(index, mesh.handle(), transform, visibility_mask, opaque, user_id);
    }
    void set(size_t index, const Curve &curve,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true,
             uint user_id = 0) noexcept {
        set_handle(index, curve.handle(), transform, visibility_mask, opaque, user_id);
    }
    void set(size_t index, const ProceduralPrimitive &prim,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             uint user_id = 0) noexcept {
        set_handle(index, prim.handle(), transform, visibility_mask, false, user_id);
    }
    void set_mesh(size_t index, const Mesh &mesh) noexcept {
        set_prim_handle(index, mesh.handle());
    }
    void set_curve(size_t index, const Curve &curve) noexcept {
        set_prim_handle(index, curve.handle());
    }
    void set_procedural_primitive(size_t index, const ProceduralPrimitive &prim) noexcept {
        set_prim_handle(index, prim.handle());
    }
    void pop_back() noexcept;
    void set_transform_on_update(size_t index, float4x4 transform) noexcept;
    void set_visibility_on_update(size_t index, uint8_t visibility_mask) noexcept;
    void set_opaque_on_update(size_t index, bool opaque) noexcept;
    void set_instance_user_id_on_update(size_t index, uint user_id) noexcept;

    // update top-level acceleration structure's instance data without build
    [[nodiscard]] luisa::unique_ptr<Command> update_instance_buffer() noexcept {
        return _build(Accel::BuildRequest::PREFER_UPDATE, true);
    }
    // update top-level acceleration structure's instance data and build
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept {
        return _build(request, false);
    }

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::AccelExprProxy *>(this);
    }
};

}// namespace luisa::compute
