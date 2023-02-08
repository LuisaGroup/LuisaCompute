//
// Created by Mike Smith on 2021/7/30.
//

#include <runtime/resource.h>
#include <runtime/device.h>
#include <core/logging.h>

namespace luisa::compute {

Resource::Resource(Resource &&rhs) noexcept
    : _device{std::move(rhs._device)},
      _info{rhs._info},
      _tag{rhs._tag} { rhs._info.invalidate(); }

void Resource::_destroy() noexcept {
    if (*this) {
        switch (_tag) {
            case Tag::BUFFER: _device->destroy_buffer(_info.handle); break;
            case Tag::TEXTURE: _device->destroy_texture(_info.handle); break;
            case Tag::BINDLESS_ARRAY: _device->destroy_bindless_array(_info.handle); break;
            case Tag::MESH: _device->destroy_mesh(_info.handle); break;
            case Tag::PROCEDURAL_PRIMITIVE: _device->destroy_procedural_primitive(_info.handle); break;
            case Tag::ACCEL: _device->destroy_accel(_info.handle); break;
            case Tag::STREAM: _device->destroy_stream(_info.handle); break;
            case Tag::EVENT: _device->destroy_event(_info.handle); break;
            case Tag::SHADER: _device->destroy_shader(_info.handle); break;
            case Tag::RASTER_SHADER: _device->destroy_raster_shader(_info.handle); break;
            case Tag::SWAP_CHAIN: _device->destroy_swap_chain(_info.handle); break;
            case Tag::DEPTH_BUFFER: _device->destroy_depth_buffer(_info.handle); break;
        }
    }
}

Resource &Resource::operator=(Resource &&rhs) noexcept {
    if (this == &rhs) [[unlikely]] { return *this; }
    LUISA_ASSERT(_device == rhs._device, "Cannot move resources between different devices.");
    LUISA_ASSERT(_tag == rhs._tag, "Cannot move resources of different types.");
    _destroy();
    _device = std::move(rhs._device);
    _info = rhs._info;
    rhs._info.invalidate();
    _tag = rhs._tag;
    return *this;
}

Resource::Resource(DeviceInterface *device,
                   Resource::Tag tag,
                   const ResourceCreationInfo &info) noexcept
    : _device{device->shared_from_this()},
      _info{info}, _tag{tag} {}

}// namespace luisa::compute
