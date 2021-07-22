//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <rtx/ray.h>
#include <rtx/hit.h>
#include <rtx/mesh.h>

namespace luisa::compute {

class Accel;

class Instance {

private:
    Accel *_geometry;
    size_t _index;

private:
    friend class Accel;
    Instance(Accel *geom, size_t index) noexcept
        : _geometry{geom}, _index{index} {}

public:
    [[nodiscard]] uint64_t mesh() const noexcept;
    void set_transform(float4x4 m) noexcept;
};

class Accel : concepts::Noncopyable {

private:
    Device::Handle _device;
    uint64_t _handle;
    std::vector<uint64_t> _instance_mesh_handles;
    std::vector<float4x4> _instance_transforms;
    bool _built{false};
    bool _dirty{false};

private:
    friend class Device;
    friend class Mesh;
    friend class Instance;

    explicit Accel(Device::Handle device) noexcept;

    void _destroy() noexcept;
    void _mark_dirty() noexcept;
    void _check_built() const noexcept;

public:
    ~Accel() noexcept;
    Accel(Accel &&) noexcept = default;
    Accel &operator=(Accel &&rhs) noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<bool> hits) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] Command *build(AccelBuildMode mode) noexcept;
    [[nodiscard]] Instance add(const Mesh &mesh, float4x4 transform) noexcept;
    [[nodiscard]] Instance instance(size_t i) noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
};

}// namespace luisa::compute
