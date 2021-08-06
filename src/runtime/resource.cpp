//
// Created by Mike Smith on 2021/7/30.
//

#include <runtime/resource.h>

namespace luisa::compute {

void Resource::_destroy() noexcept {
    if (*this) {
        switch (_tag) {
            case Tag::BUFFER: _device->destroy_buffer(_handle); break;
            case Tag::TEXTURE: _device->destroy_texture(_handle); break;
            case Tag::HEAP: _device->destroy_heap(_handle); break;
            case Tag::MESH: _device->destroy_mesh(_handle); break;
            case Tag::ACCEL: _device->destroy_accel(_handle); break;
            case Tag::STREAM: _device->destroy_stream(_handle); break;
            case Tag::EVENT: _device->destroy_event(_handle); break;
            case Tag::SHADER: _device->destroy_shader(_handle); break;
        }
    }
}

Resource &Resource::operator=(Resource &&rhs) noexcept {
    if (this != &rhs) [[unlikely]] {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
        _tag = rhs._tag;
    }
    return *this;
}

Resource::Resource(Device::Interface *device, Resource::Tag tag, uint64_t handle) noexcept
: _device{device->shared_from_this()}, _handle{handle}, _tag{tag} {}

}
