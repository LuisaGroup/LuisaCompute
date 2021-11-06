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
    auto v = _kernel.arguments()[_argument_index++].uid();
    _dispatch_command()->encode_bindless_array(v, array.handle());
    return *this;
}

}// namespace detail

BindlessArray Device::create_bindless_array(size_t slots) noexcept {
    return _create<BindlessArray>(slots);
}

void BindlessArray::_mark_dirty(size_t index) noexcept {
    if (_dirty_count == 0u) {
        _dirty_begin = index;
        _dirty_count = 1u;
    } else {
        auto dirty_end = std::max(_dirty_begin + _dirty_count, index + 1u);
        _dirty_begin = std::min(_dirty_begin, index);
        _dirty_count = dirty_end - _dirty_begin;
    }
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
    _mark_dirty(index);
}

void BindlessArray::_emplace_tex2d(size_t index, uint64_t handle, Sampler sampler) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    device()->emplace_tex2d_in_bindless_array(this->handle(), index, handle, sampler);
    _mark_dirty(index);
}

void BindlessArray::_emplace_tex3d(size_t index, uint64_t handle, Sampler sampler) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    device()->emplace_tex3d_in_bindless_array(this->handle(), index, handle, sampler);
    _mark_dirty(index);
}

BindlessArray &BindlessArray::remove_buffer(size_t index) noexcept {
    device()->remove_buffer_in_bindless_array(handle(), index);
    _mark_dirty(index);
    return *this;
}

BindlessArray &BindlessArray::remove_tex2d(size_t index) noexcept {
    device()->remove_tex2d_in_bindless_array(handle(), index);
    _mark_dirty(index);
    return *this;
}

BindlessArray &BindlessArray::remove_tex3d(size_t index) noexcept {
    device()->remove_tex3d_in_bindless_array(handle(), index);
    _mark_dirty(index);
    return *this;
}

Command *BindlessArray::update() noexcept {
    if (_dirty_count == 0u) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Ignoring update command on bindless "
            "array #{} without modification.",
            handle());
        return nullptr;
    }
    auto command = BindlessArrayUpdateCommand::create(handle(), _dirty_begin, _dirty_count);
    _dirty_begin = _dirty_count = 0u;// clear dirty state
    return command;
}

}// namespace luisa::compute
