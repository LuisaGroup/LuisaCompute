//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <runtime/rtx/triangle.h>

namespace luisa::compute {

class Accel;

class LC_RUNTIME_API Mesh final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;

private:
    uint _triangle_count{};
    uint64_t _v_buffer{};
    size_t _v_buffer_offset{};
    size_t _v_buffer_size{};
    size_t _v_stride{};
    uint64_t _t_buffer{};
    size_t _t_buffer_offset{};
    size_t _t_buffer_size{};

private:
    friend class Device;

    template<typename VBuffer, typename TBuffer>
        requires is_buffer_or_view_v<VBuffer> &&
                 is_buffer_or_view_v<TBuffer> &&
                 std::same_as<buffer_element_t<TBuffer>, Triangle>
    [[nodiscard]] static ResourceCreationInfo _create_resource(
        DeviceInterface *device, const AccelOption &option,
        const VBuffer &vertex_buffer, const TBuffer &triangle_buffer) noexcept {
        return device->create_mesh(option);
    }

private:
    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device, const VBuffer &vertex_buffer, const TBuffer &triangle_buffer,
         const AccelOption &option) noexcept
        : Resource{device, Resource::Tag::MESH,
                   _create_resource(device, option, vertex_buffer, triangle_buffer)},
          _triangle_count{static_cast<uint>(triangle_buffer.size())},
          _v_buffer{BufferView{vertex_buffer}.handle()},
          _v_buffer_offset{BufferView{vertex_buffer}.offset_bytes()},
          _v_buffer_size{BufferView{vertex_buffer}.size_bytes()},
          _v_stride(vertex_buffer.stride()),
          _t_buffer{BufferView{triangle_buffer}.handle()},
          _t_buffer_offset{BufferView{triangle_buffer}.offset_bytes()},
          _t_buffer_size{BufferView{triangle_buffer}.size_bytes()} {}

public:
    Mesh() noexcept = default;
    Mesh(Mesh &&) noexcept = default;
    Mesh(Mesh const &) noexcept = delete;
    Mesh &operator=(Mesh &&) noexcept = default;
    Mesh &operator=(Mesh const &) noexcept = delete;
    using Resource::operator bool;
    // build triangle based bottom-level acceleration structure
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;
    [[nodiscard]] auto triangle_count() const noexcept { return _triangle_count; }
};

template<typename VBuffer, typename TBuffer>
Mesh Device::create_mesh(VBuffer &&vertices, TBuffer &&triangles, const AccelOption &option) noexcept {
    return this->_create<Mesh>(std::forward<VBuffer>(vertices), std::forward<TBuffer>(triangles), option);
}

}// namespace luisa::compute
