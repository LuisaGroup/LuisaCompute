//
// Created by Mike Smith on 2021/7/30.
//

#include <runtime/resource.h>
#include <runtime/device.h>
#include <core/logging.h>

namespace luisa::compute {
Resource::Resource(Resource &&rhs) noexcept
    : _device(std::move(rhs._device)),
      _handle(rhs._handle),
      _tag(rhs._tag) {
    rhs._handle = invalid_handle;
}

void Resource::_destroy() noexcept {
    if (*this) {
        switch (_tag) {
            case Tag::BUFFER: _device->destroy_buffer(_handle); break;
            case Tag::TEXTURE: _device->destroy_texture(_handle); break;
            case Tag::BINDLESS_ARRAY: _device->destroy_bindless_array(_handle); break;
            case Tag::MESH:
            case Tag::PROCEDURAL_PRIMITIVE: _device->destroy_mesh(_handle); break;
            case Tag::ACCEL: _device->destroy_accel(_handle); break;
            case Tag::STREAM: _device->destroy_stream(_handle); break;
            case Tag::EVENT: _device->destroy_event(_handle); break;
            case Tag::SHADER: _device->destroy_shader(_handle); break;
            case Tag::RASTER_SHADER: _device->destroy_raster_shader(_handle); break;
            case Tag::SWAP_CHAIN: _device->destroy_swap_chain(_handle); break;
            case Tag::DEPTH_BUFFER: _device->destroy_depth_buffer(_handle); break;
        }
    }
}

Resource &Resource::operator=(Resource &&rhs) noexcept {
    if (this == &rhs) [[unlikely]]
        return *this;
    _destroy();
    _device = std::move(rhs._device);
    _handle = rhs._handle;
    rhs._handle = invalid_handle;
    _tag = rhs._tag;
    return *this;
}

Resource::Resource(DeviceInterface *device, Resource::Tag tag, uint64_t handle) noexcept
    : _device{device->shared_from_this()}, _handle{handle}, _tag{tag} {}

void *Resource::native_handle() const noexcept {
    switch (_tag) {
        case Tag::BUFFER: return _device->buffer_native_handle(_handle);
        case Tag::TEXTURE: return _device->texture_native_handle(_handle);
        case Tag::BINDLESS_ARRAY: LUISA_ERROR_WITH_LOCATION("Native handles of bindless arrays are not obtainable yet.");
        case Tag::MESH: LUISA_ERROR_WITH_LOCATION("Native handles of meshes are not obtainable yet.");
        case Tag::PROCEDURAL_PRIMITIVE: LUISA_ERROR_WITH_LOCATION("Native handles of meshes are not obtainable yet.");
        case Tag::ACCEL: LUISA_ERROR_WITH_LOCATION("Native handles of acceleration structures are not obtainable yet.");
        case Tag::STREAM: return _device->stream_native_handle(_handle);
        case Tag::EVENT: LUISA_ERROR_WITH_LOCATION("Native handles of events are not obtainable yet.");
        case Tag::SHADER: LUISA_ERROR_WITH_LOCATION("Native handles of shaders are not obtainable yet.");
        case Tag::RASTER_SHADER: LUISA_ERROR_WITH_LOCATION("Native handles of raster shaders are not obtainable yet.");
        case Tag::SWAP_CHAIN: LUISA_ERROR_WITH_LOCATION("Native handles of swap chains are not obtainable yet.");
        case Tag::DEPTH_BUFFER: LUISA_ERROR_WITH_LOCATION("Native handles of depth buffer are not obtainable yet.");
    }
    LUISA_ERROR_WITH_LOCATION("Unknown resource tag.");
}

}// namespace luisa::compute
