//
// Created by Mike Smith on 2021/6/24.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <rtx/ray.h>
#include <rtx/hit.h>

namespace luisa::compute {

struct Triangle {
    uint i[3];
};

class Geometry;

namespace detail {

class Mesh {

private:
    Geometry *_geometry;
    uint _index;
    BufferView<float3> _vertices;
    BufferView<Triangle> _triangles;

public:
    Mesh(Geometry *geom, uint index, BufferView<float3> vertices, BufferView<Triangle> triangles) noexcept
        : _geometry{geom}, _index{index}, _vertices{vertices}, _triangles{triangles} {}
    [[nodiscard]] Command *build() const noexcept;
    [[nodiscard]] Command *update() const noexcept;
    [[nodiscard]] uint64_t handle() const noexcept;
    [[nodiscard]] auto vertex_buffer() const noexcept { return _vertices; }
    [[nodiscard]] auto triangle_buffer() const noexcept { return _triangles; }
};

}

class Geometry : concepts::Noncopyable {

private:
    Device::Interface *_device;
    uint64_t _handle;
    std::vector<uint64_t> _mesh_handles;
    std::vector<uint64_t> _instance_mesh_handles;
    std::vector<float4x4> _instance_transforms;
    std::vector<bool> _mesh_built;
    bool _built{false};
    bool _dirty{false};

private:
    friend class detail::Mesh;
    void _mark_mesh_built(uint mesh_index) noexcept;
    void _mark_dirty() noexcept;

public:
    ~Geometry() noexcept { _device->destroy_accel(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_closest(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<Hit> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<bool> hits) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *trace_any(BufferView<Ray> rays, BufferView<uint32_t> indices, BufferView<bool> hits, BufferView<uint> ray_count) const noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] Command *build() noexcept;
};

}

LUISA_STRUCT(luisa::compute::Triangle, i)
