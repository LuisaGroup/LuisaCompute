//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/device.h>
#include <runtime/shader.h>
#include <runtime/command.h>
#include <runtime/bindless_array.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const BindlessArray &array) noexcept {
    _encode_pending_bindings();
    if (auto t = _kernel.arguments()[_argument_index].type();
        !t->is_bindless_array()) {
        LUISA_ERROR_WITH_LOCATION(
            "Expected {} but got bindless array for argument {}.",
            t->description(), _argument_index);
    }
    auto v = _kernel.arguments()[_argument_index++].uid();
    _dispatch_command()->encode_bindless_array(v, array.handle());
    return *this;
}

}// namespace detail

BindlessArray Device::create_bindless_array(size_t slots) noexcept {
    return _create<BindlessArray>(slots);
}

BindlessArray::BindlessArray(Device::Interface *device, size_t size) noexcept
    : Resource{device, Tag::BINDLESS_ARRAY, device->create_bindless_array(size)},
      _size{size} {}

void BindlessArray::_emplace_buffer(size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    device()->emplace_buffer_in_bindless_array(this->handle(), index, handle, offset_bytes);
}

void BindlessArray::_emplace_tex2d(size_t index, uint64_t handle, Sampler sampler) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    device()->emplace_tex2d_in_bindless_array(this->handle(), index, handle, sampler);
}

void BindlessArray::_emplace_tex3d(size_t index, uint64_t handle, Sampler sampler) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    device()->emplace_tex3d_in_bindless_array(this->handle(), index, handle, sampler);
}

BindlessArray &BindlessArray::remove_buffer(size_t index) noexcept {
    device()->remove_buffer_in_bindless_array(handle(), index);
    return *this;
}

BindlessArray &BindlessArray::remove_tex2d(size_t index) noexcept {
    device()->remove_tex2d_in_bindless_array(handle(), index);
    return *this;
}

BindlessArray &BindlessArray::remove_tex3d(size_t index) noexcept {
    device()->remove_tex3d_in_bindless_array(handle(), index);
    return *this;
}

Command *BindlessArray::update() noexcept {
    return BindlessArrayUpdateCommand::create(handle());
}

}// namespace luisa::compute
