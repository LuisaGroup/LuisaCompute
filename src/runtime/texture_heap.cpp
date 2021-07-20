//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/texture_heap.h>

namespace luisa::compute {

TextureHeap::TextureHeap(TextureHeap &&another) noexcept
    : _device{std::move(another._device)},
      _handle{another._handle},
      _capacity{another._capacity},
      _slots{std::move(another._slots)} {}

TextureHeap &TextureHeap::operator=(TextureHeap &&rhs) noexcept {
    if (&rhs != this) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
        _capacity = rhs._capacity;
        _slots = std::move(rhs._slots);
    }
    return *this;
}

constexpr auto TextureHeap::_compute_mip_levels(uint3 size, uint requested_levels) noexcept {
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
    if (*this) { _device->destroy_texture_heap(_handle); }
}

TextureHeap::~TextureHeap() noexcept { _destroy(); }

TextureHeap::TextureHeap(Device::Handle device, size_t capacity) noexcept
    : _device{std::move(device)},
      _handle{_device->create_texture_heap(capacity)},
      _capacity{capacity},
      _slots(slot_count, invalid_handle) {}

detail::Texture2D TextureHeap::create(uint index, PixelStorage storage, uint2 size, TextureSampler sampler, uint mip_levels) noexcept {
    if (auto h = _slots[index]; h != invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "Overwriting texture #{} at {} in heap #{}.",
            h, index, _handle);
        destroy(index);
    }
    auto valid_mip_levels = _compute_mip_levels(make_uint3(size, 1u), mip_levels);
    if (valid_mip_levels == 1u
        && (sampler.filter() == TextureSampler::Filter::TRILINEAR
            || sampler.filter() == TextureSampler::Filter::ANISOTROPIC)) {
        LUISA_WARNING_WITH_LOCATION(
            "Textures without mipmaps do not support "
            "trilinear or anisotropic sampling.");
        sampler.set_filter(TextureSampler::Filter::BILINEAR);
    }
    auto handle = _device->create_texture(
        pixel_storage_to_format<float>(storage), 2u,
        size.x, size.y, 1u, valid_mip_levels,
        sampler, _handle, index);
    _slots[index] = handle;
    return {handle, storage, valid_mip_levels, size};
}

detail::Texture3D TextureHeap::create(uint index, PixelStorage storage, uint3 size, TextureSampler sampler, uint mip_levels) noexcept {
    if (auto h = _slots[index]; h != invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "Overwriting texture #{} at {} in heap #{}.",
            h, index, _handle);
        destroy(index);
    }
    auto valid_mip_levels = _compute_mip_levels(size, mip_levels);
    if (valid_mip_levels == 1u
        && (sampler.filter() == TextureSampler::Filter::TRILINEAR
            || sampler.filter() == TextureSampler::Filter::ANISOTROPIC)) {
        LUISA_WARNING_WITH_LOCATION(
            "Textures without mipmaps do not support "
            "trilinear or anisotropic sampling.");
        sampler.set_filter(TextureSampler::Filter::BILINEAR);
    }
    auto handle = _device->create_texture(
        pixel_storage_to_format<float>(storage), 3u,
        size.x, size.y, size.z, valid_mip_levels,
        sampler, _handle, index);
    _slots[index] = handle;
    return {handle, storage, valid_mip_levels, size};
}

void TextureHeap::destroy(uint32_t index) noexcept {
    if (auto &&h = _slots[index]; h == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "Recycling already destroyed heap texture at slot {} in heap #{}.",
            index, _handle);
    } else {
        _device->destroy_texture(h);
        h = invalid_handle;
    }
}

Command *detail::Texture2D::load(const void *pixels, uint mip_level) noexcept {
    if (!validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    return TextureUploadCommand::create(
        _handle, _storage, mip_level,
        make_uint3(0u),
        make_uint3(mipmap_size, 1u),
        pixels);
}

Command *detail::Texture2D::load(ImageView<float> image, uint mip_level) noexcept {
    if (!validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    if (!all(mipmap_size == image.size())) {
        LUISA_WARNING_WITH_LOCATION(
            "Sizes mismatch when copying from image #{} "
            "to texture #{} (mipmap level {}) "
            "([{}, {}] vs. [{}, {}]).",
            image.handle(), _handle, mip_level,
            image.size().x, image.size().y, mipmap_size.x, mipmap_size.y);
        mipmap_size = min(mipmap_size, image.size());
    }
    return TextureCopyCommand::create(
        image.handle(), _handle,
        0u, mip_level,
        make_uint3(image.offset(), 0u), make_uint3(0u),
        make_uint3(mipmap_size, 1u));
}

Command *detail::Texture3D::load(const void *pixels, uint mip_level) noexcept {
    if (!validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    return TextureUploadCommand::create(
        _handle, _storage, mip_level,
        make_uint3(0u), mipmap_size,
        pixels);
}

Command *detail::Texture3D::load(VolumeView<float> volume, uint mip_level) noexcept {
    if (!validate_mip_level(*this, mip_level)) { return nullptr; }
    auto mipmap_size = max(_size >> mip_level, 1u);
    if (!all(mipmap_size == volume.size())) {
        LUISA_WARNING_WITH_LOCATION(
            "Sizes mismatch when copying from image #{} "
            "to texture #{} (mipmap level {}) "
            "([{}, {}, {}] vs. [{}, {}, {}]).",
            volume.handle(), _handle, mip_level,
            volume.size().x, volume.size().y, volume.size().z,
            mipmap_size.x, mipmap_size.y, mipmap_size.z);
        mipmap_size = min(mipmap_size, volume.size());
    }
    return TextureCopyCommand::create(
        volume.handle(), _handle, 0u, mip_level,
        volume.offset(), make_uint3(0u), mipmap_size);
}

}// namespace luisa::compute
