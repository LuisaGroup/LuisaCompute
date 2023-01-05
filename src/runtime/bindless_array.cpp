//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/device.h>
#include <runtime/shader.h>
#include <runtime/command.h>
#include <runtime/bindless_array.h>
#include <core/logging.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const BindlessArray &array) noexcept {
    _command->encode_bindless_array(array.handle());
    return *this;
}

}// namespace detail

BindlessArray Device::create_bindless_array(size_t slots) noexcept {
    return _create<BindlessArray>(slots);
}

BindlessArray::BindlessArray(DeviceInterface *device, size_t size) noexcept
    : Resource{device, Tag::BINDLESS_ARRAY, device->create_bindless_array(size)},
      _size{size} {}

void BindlessArray::emplace_buffer_on_update(size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->buffer = Modification::Buffer::emplace(handle, offset_bytes);
}

void BindlessArray::emplace_tex2d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->tex2d = Modification::Texture::emplace(handle, sampler);
}

void BindlessArray::emplace_tex3d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->tex3d = Modification::Texture::emplace(handle, sampler);
}

BindlessArray &BindlessArray::remove_buffer_on_update(size_t index) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->buffer = Modification::Buffer::remove();
    return *this;
}

BindlessArray &BindlessArray::remove_tex2d_on_update(size_t index) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->tex2d = Modification::Texture::remove();
    return *this;
}

BindlessArray &BindlessArray::remove_tex3d_on_update(size_t index) noexcept {
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->tex3d = Modification::Texture::remove();
    return *this;
}

luisa::unique_ptr<Command> BindlessArray::update() noexcept {
    luisa::vector<Modification> mods;
    mods.reserve(_updates.size());
    for (auto m : _updates) { mods.emplace_back(m); }
    _updates.clear();
    return BindlessArrayUpdateCommand::create(handle(), std::move(mods));
}

}// namespace luisa::compute
