//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/device.h>
#include <runtime/heap.h>

namespace luisa::compute {

Heap Device::create_heap(size_t size) noexcept {
    return _create<Heap>(size);
}

Heap &Heap::operator=(Heap &&rhs) noexcept {
    if (&rhs != this) {
        _destroy();
        _device = std::move(rhs._device);
        _handle = rhs._handle;
        _capacity = rhs._capacity;
        _slots = std::move(rhs._slots);
    }
    return *this;
}

constexpr auto Heap::_compute_mip_levels(uint3 size, uint requested_levels) noexcept {
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

void Heap::_destroy() noexcept {
    if (*this) {
        for (auto texture : _slots) {
            if (texture != invalid_handle) {
                _device->destroy_texture(texture);
            }
        }
        _device->destroy_heap(_handle);
    }
}

Heap::~Heap() noexcept { _destroy(); }

Heap::Heap(Device::Handle device, size_t capacity) noexcept
    : _device{std::move(device)},
      _handle{_device->create_heap(capacity)},
      _capacity{capacity},
      _slots(slot_count, invalid_handle) {}

Texture2D Heap::create_tex2d(uint index, PixelStorage storage, uint2 size, TextureSampler sampler, uint mip_levels) noexcept {
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

Texture3D Heap::create_tex3d(uint index, PixelStorage storage, uint3 size, TextureSampler sampler, uint mip_levels) noexcept {
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

void Heap::destroy(uint32_t index) noexcept {
    if (auto &&h = _slots[index]; h == invalid_handle) {
        LUISA_WARNING_WITH_LOCATION(
            "Recycling already destroyed heap texture at slot {} in heap #{}.",
            index, _handle);
    } else {
        _device->destroy_texture(h);
        h = invalid_handle;
    }
}

}// namespace luisa::compute
