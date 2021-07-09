//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/texture_heap.h>

namespace luisa::compute {

TextureHeap::TextureHeap(TextureHeap &&another) noexcept
    : _device{another._device},
      _handle{another._handle},
      _capacity{another._capacity},
      _slots{std::move(another._slots)},
      _available{std::move(another._available)} { another._device = nullptr; }

TextureHeap &TextureHeap::operator=(TextureHeap &&rhs) noexcept {
    if (&rhs != this) {
        _destroy();
        _device = rhs._device;
        _handle = rhs._handle;
        _capacity = rhs._capacity;
        _slots = std::move(rhs._slots);
        _available = std::move(rhs._available);
        rhs._device = nullptr;
    }
    return *this;
}

constexpr auto TextureHeap::_compute_mipmap_levels(uint3 size, uint requested_levels) noexcept {
    auto max_size = std::max({size.x, size.y, size.z});
    auto max_levels = 0u;
    while (max_size != 0u) {
        max_size >>= 1u;
        max_levels++;
    }
    return requested_levels == 0u
               ? max_levels
               : std::min(requested_levels, max_levels);
}

void TextureHeap::_destroy() noexcept {
    if (_device != nullptr) { _device->dispose_texture_heap(_handle); }
}

TextureHeap::~TextureHeap() noexcept { _destroy(); }

TextureHeap::TextureHeap(Device &device, size_t capacity) noexcept
    : _device{device.impl()},
      _handle{device.impl()->create_texture_heap(capacity)},
      _capacity{capacity},
      _slots(max_slot_count),
      _available(max_slot_count) {
    for (auto i = 0u; i < max_slot_count; i++) {
        _available[i] = max_slot_count - 1u - i;
    }
}

uint32_t TextureHeap::allocate(PixelStorage storage, uint2 size, TextureSampler sampler, uint mipmap_levels) noexcept {
    if (_available.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate texture from heap #{} with full slots.", _handle);
        return invalid_index;
    }
    auto valid_mipmap_levels = _compute_mipmap_levels(make_uint3(size, 1u), mipmap_levels);
    if (valid_mipmap_levels == 1) { sampler.set_mip_filter_mode(TextureSampler::MipFilterMode::NONE); }
    auto index = _available.back();
    _available.pop_back();
    auto handle = _device->create_texture(
        pixel_storage_to_format<float>(storage), 2u,
        size.x, size.y, 1u, valid_mipmap_levels,
        sampler, _handle, index);
    _slots[index] = {handle, storage, 2u, make_uint3(size, 1u), mipmap_levels};
    return index;
}

uint32_t TextureHeap::allocate(PixelStorage storage, uint3 size, TextureSampler sampler, uint mipmap_levels) noexcept {
    if (_available.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate texture from heap #{} with full slots.", _handle);
        return invalid_index;
    }
    auto valid_mipmap_levels = _compute_mipmap_levels(size, mipmap_levels);
    if (valid_mipmap_levels == 1) { sampler.set_mip_filter_mode(TextureSampler::MipFilterMode::NONE); }
    auto index = _available.back();
    _available.pop_back();
    auto handle = _device->create_texture(
        pixel_storage_to_format<float>(storage), 3u,
        size.x, size.y, size.z, valid_mipmap_levels,
        sampler, _handle, index);
    _slots[index] = {handle, storage, 3u, size, mipmap_levels};
    return index;
}

void TextureHeap::recycle(uint32_t index) noexcept {
    if (_slots[index].invalid()) {
        LUISA_WARNING_WITH_LOCATION(
            "Recycling already recycled heap texture at slot {} in heap #{}.",
            index, _handle);
        return;
    }
    _device->dispose_texture(_slots[index].handle());
    _slots[index].invalidate();
}

CommandHandle TextureHeap::emplace(uint32_t index, ImageView<float> view, uint32_t mipmap_level) noexcept {
    if (!_validate_mipmap_level<2>(index, mipmap_level)) { return nullptr; }
    auto tex_desc = _slots[index];
    auto size = make_uint2(tex_desc.size());
    auto mipmap_size = max(size >> mipmap_level, 1u);
    if (!all(size == view.size())) {
        LUISA_WARNING_WITH_LOCATION(
            "Sizes mismatch when copying from image #{} "
            "to texture #{} (mipmap level {}) at {} in heap #{} "
            "([{}, {}] vs. [{}, {}]).",
            view.handle(), tex_desc.handle(), mipmap_level, index, _handle,
            view.size().x, view.size().y, mipmap_size.x, mipmap_size.y);
        mipmap_size = min(mipmap_size, view.size());
    }
    return TextureCopyCommand::create(
        view.handle(), tex_desc.handle(),
        0u, mipmap_level,
        make_uint3(view.offset(), 0u), make_uint3(0u),
        make_uint3(mipmap_size, 1u), _handle);
}

CommandHandle TextureHeap::emplace(uint32_t index, VolumeView<float> view, uint32_t mipmap_level) noexcept {
    if (!_validate_mipmap_level<3>(index, mipmap_level)) { return nullptr; }
    auto tex_desc = _slots[index];
    auto size = tex_desc.size();
    auto mipmap_size = max(size >> mipmap_level, 1u);
    if (!all(size == view.size())) {
        LUISA_WARNING_WITH_LOCATION(
            "Sizes mismatch when copying from image #{} "
            "to texture #{} (mipmap level {}) at {} in heap #{} "
            "([{}, {}, {}] vs. [{}, {}, {}]).",
            view.handle(), tex_desc.handle(), mipmap_level, index, _handle,
            view.size().x, view.size().y, view.size().z,
            mipmap_size.x, mipmap_size.y, mipmap_size.z);
        mipmap_size = min(mipmap_size, view.size());
    }
    return TextureCopyCommand::create(
        view.handle(), tex_desc.handle(),
        0u, mipmap_level,
        view.offset(), make_uint3(0u),
        mipmap_size, _handle);
}

CommandHandle TextureHeap::emplace(uint32_t index, const void *pixels, uint32_t mipmap_level) noexcept {
    if (!_validate_mipmap_level<0u>(index, mipmap_level)) { return nullptr; }
    auto tex = _slots[index];
    auto size = tex.size();
    auto mipmap_size = max(size >> mipmap_level, 1u);
    return TextureUploadCommand::create(
        tex.handle(), tex.storage(), mipmap_level,
        make_uint3(0u), mipmap_size,
        pixels, _handle);
}

template<uint dim /* 0 for don't check */>
bool TextureHeap::_validate_mipmap_level(uint index, uint level) const noexcept {
    auto desc = _slots[index];
    if (desc.invalid()) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid texture at index {} in heap #{}.",
            index, _handle);
        return false;
    }
    if (auto d = desc.dimension(); dim != 0u && d != dim) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture dimension {} (expected {}).",
            d, dim);
    }
    if (level >= desc.mip_levels()) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid mipmap level {} (max = {}) for texture #{} at index {} in heap #{}.",
            level, desc.mip_levels() - 1u, desc.handle(), index, _handle);
        return false;
    }
    return true;
}

}// namespace luisa::compute
