//
// Created by Mike Smith on 2021/7/22.
//

#pragma once

#include <runtime/device.h>
#include <runtime/buffer.h>
#include <dsl/syntax.h>
#include <dsl/struct.h>

namespace luisa::compute {

struct Triangle {
    uint i0;
    uint i1;
    uint i2;
};

class Accel;

class LC_RUNTIME_API Mesh final : public Resource {

public:
    using BuildRequest = AccelBuildRequest;

private:
    friend class Device;

private:
    template<typename VBuffer, typename TBuffer>
    Mesh(DeviceInterface *device, const AccelCreateOption &option,
         const VBuffer &vertex_buffer, const TBuffer &triangle_buffer) noexcept
        : Resource{device, Resource::Tag::MESH,
                   device->create_mesh(option,
                                       vertex_buffer.handle(), vertex_buffer.offset_bytes(),
                                       sizeof(buffer_element_t<VBuffer>), vertex_buffer.size(),
                                       triangle_buffer.handle(), triangle_buffer.offset_bytes(),
                                       triangle_buffer.size())} {}

public:
    Mesh() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;
};

template<typename VBuffer, typename TBuffer>
Mesh Device::create_mesh(const AccelCreateOption &option,
                         VBuffer &&vertices, TBuffer &&triangles) noexcept {
    return this->_create<Mesh>(option,
                               BufferView{std::forward<VBuffer>(vertices)},
                               BufferView{std::forward<TBuffer>(triangles)});
}

}// namespace luisa::compute

LUISA_STRUCT(luisa::compute::Triangle, i0, i1, i2)
