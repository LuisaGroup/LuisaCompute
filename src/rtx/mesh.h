//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#include <core/allocator.h>
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

class Mesh {

private:
    Device::Handle _device;
    uint64_t _handle{0u};
    mutable luisa::unordered_set<Accel *> _observers;
    bool _requires_rebuild{true};

private:
    friend class Device;
    friend class Accel;
    void _register(Accel *accel) const noexcept;
    void _remove(Accel *accel) const noexcept;
    void _destroy() noexcept;

private:
    template<typename VBuffer, typename TBuffer>
        requires is_buffer_or_view_v<VBuffer> &&
            is_buffer_or_view_v<TBuffer> &&
            std::same_as<buffer_element_t<TBuffer>, Triangle>
    explicit Mesh(Device::Interface *device, VBuffer &&vertex_buffer, TBuffer &&triangle_buffer,
                  AccelBuildHint hint = AccelBuildHint::FAST_TRACE) noexcept
        : _device{device->shared_from_this()} {
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
        _handle = device->create_mesh(
            vertex_buffer_handle, vertex_buffer_offset, vertex_stride, vertex_count,
            triangle_buffer_handle, triangle_buffer_offset, triangle_count, hint);
    }

public:
    Mesh() noexcept = default;
    ~Mesh() noexcept;
    Mesh(Mesh &&another) noexcept;
    Mesh(const Mesh &) noexcept = delete;
    Mesh &operator=(Mesh &&rhs) noexcept;
    Mesh &operator=(const Mesh &) noexcept = delete;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto device() const noexcept { return _device.get(); }
    [[nodiscard]] Command *build() noexcept;
    [[nodiscard]] Command *update() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
};

template<typename VBuffer, typename TBuffer>
Mesh Device::create_mesh(VBuffer &&vertices, TBuffer &&triangles, AccelBuildHint hint) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), std::forward<TBuffer>(triangles), hint);
}

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Triangle, i0, i1, i2){};
