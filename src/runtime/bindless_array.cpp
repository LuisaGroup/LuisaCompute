#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const BindlessArray &array) noexcept {
    array._check_is_valid();
    _encoder.encode_bindless_array(array.handle());
    return *this;
}

}// namespace detail

BindlessArray Device::create_bindless_array(size_t slots) noexcept {
    return _create<BindlessArray>(slots);
}

BindlessArray::BindlessArray(DeviceInterface *device, size_t size) noexcept
    : Resource{device, Tag::BINDLESS_ARRAY, device->create_bindless_array(size)},
      _size{size} {}

void BindlessArray::_emplace_buffer_on_update(size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    _check_is_valid();
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->buffer = Modification::Buffer::emplace(handle, offset_bytes);
}

void BindlessArray::_emplace_tex2d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    _check_is_valid();
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->tex2d = Modification::Texture::emplace(handle, sampler);
}

void BindlessArray::_emplace_tex3d_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    _check_is_valid();
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    auto [iter, _] = _updates.emplace(Modification{index});
    iter->tex3d = Modification::Texture::emplace(handle, sampler);
}

BindlessArray &BindlessArray::remove_buffer_on_update(size_t index) noexcept {
    _check_is_valid();
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
    _check_is_valid();
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
    _check_is_valid();
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
    _check_is_valid();
    if (!dirty()) {
        LUISA_WARNING_WITH_LOCATION(
            "No update to bindless array.");
        return nullptr;
    }
    luisa::vector<Modification> mods;
    mods.reserve(_updates.size());
    for (auto m : _updates) { mods.emplace_back(m); }
    _updates.clear();
    return luisa::make_unique<BindlessArrayUpdateCommand>(handle(), std::move(mods));
}

BindlessArray::~BindlessArray() noexcept {
    if (!_updates.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Bindless array #{} destroyed with {} pending updates. "
            "Did you forget to call update()?",
            this->handle(), _updates.size());
    }
    if (*this) {
        device()->destroy_bindless_array(handle());
    }
}

}// namespace luisa::compute

