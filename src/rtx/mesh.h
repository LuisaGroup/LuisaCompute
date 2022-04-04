//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#include <core/stl.h>
#include <runtime/device.h>
#include <runtime/buffer.h>
#include <dsl/syntax.h>

namespace luisa::compute {

struct Triangle {
    uint i0;
    uint i1;
    uint i2;
};

class Accel;

class LC_RTX_API Mesh {

private:
    Resource::Handle _resource;
    uint _triangle_count{};

private:
    friend class Device;

private:
    template<typename VBuffer, typename TBuffer>
        requires is_buffer_or_view_v<VBuffer> &&
            is_buffer_or_view_v<TBuffer> &&
            std::same_as<buffer_element_t<TBuffer>, Triangle>
    explicit Mesh(Device::Interface *device, VBuffer &&vertex_buffer, TBuffer &&triangle_buffer,
                  AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept
        : _triangle_count{static_cast<uint>(triangle_buffer.size())} {
        BufferView vertices{std::forward<VBuffer>(vertex_buffer)};
        BufferView triangles{std::forward<TBuffer>(triangle_buffer)};
        using vertex_type = buffer_element_t<VBuffer>;
        auto vertex_buffer_handle = vertices.handle();
        auto vertex_buffer_offset = vertices.offset_bytes();
        auto vertex_stride = sizeof(vertex_type);
        auto vertex_count = vertices.size();
        auto triangle_buffer_handle = triangles.handle();
        auto triangle_buffer_offset = triangles.offset_bytes();
        auto triangle_count = triangles.size();
        auto handle = device->create_mesh(
            vertex_buffer_handle, vertex_buffer_offset, vertex_stride, vertex_count,
            triangle_buffer_handle, triangle_buffer_offset, triangle_count, hint);
        _resource = luisa::make_shared<Resource>(device, Resource::Tag::MESH, handle);
    }

public:
    Mesh() noexcept = default;
    [[nodiscard]] Command *build() noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] auto triangle_count() const noexcept { return _triangle_count; }
    [[nodiscard]] explicit operator bool() const noexcept { return _resource != nullptr; }
    [[nodiscard]] auto resource() const noexcept { return _resource.get(); }
};

template<typename VBuffer, typename TBuffer>
Mesh Device::create_mesh(VBuffer &&vertices, TBuffer &&triangles, AccelBuildHint hint) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), std::forward<TBuffer>(triangles), hint);
}

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Triangle, i0, i1, i2){};
