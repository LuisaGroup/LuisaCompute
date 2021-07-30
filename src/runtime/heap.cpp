//
// Created by Mike Smith on 2021/4/7.
//

#include <runtime/device.h>
#include <runtime/heap.h>

namespace luisa::compute {

Heap Device::create_heap(size_t size) noexcept {
    return _create<Heap>(size);
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

Heap::Heap(Device::Interface *device, size_t capacity) noexcept
    : Resource{device, Tag::HEAP, device->create_heap(capacity)},
      _capacity{capacity},
      _texture_slots(slot_count, invalid_handle),
      _buffer_slots(slot_count, invalid_handle) {}

TextureView2D Heap::create_texture(uint index, PixelStorage storage, uint2 size, TextureSampler sampler, uint mip_levels) noexcept {
    if (auto h = _texture_slots[index]; h != invalid_handle) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Overwriting texture #{} at {} in heap #{}.",
            h, index, handle());
        destroy_texture(index);
    }
    auto valid_mip_levels = _compute_mip_levels(make_uint3(size, 1u), mip_levels);
    if (valid_mip_levels == 1u
        && (sampler.filter() == TextureSampler::Filter::TRILINEAR
            || sampler.filter() == TextureSampler::Filter::ANISOTROPIC)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Textures without mipmaps do not support "
            "trilinear or anisotropic sampling.");
        sampler.set_filter(TextureSampler::Filter::BILINEAR);
    }
    auto handle = device()->create_texture(
        pixel_storage_to_format<float>(storage), 2u,
        size.x, size.y, 1u, valid_mip_levels,
        sampler, this->handle(), index);
    _texture_slots[index] = handle;
    return {handle, storage, valid_mip_levels, size};
}

TextureView3D Heap::create_texture(uint index, PixelStorage storage, uint3 size, TextureSampler sampler, uint mip_levels) noexcept {
    if (auto h = _texture_slots[index]; h != invalid_handle) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Overwriting texture #{} at {} in heap #{}.",
            h, index, handle());
        destroy_texture(index);
    }
    auto valid_mip_levels = _compute_mip_levels(size, mip_levels);
    if (valid_mip_levels == 1u
        && (sampler.filter() == TextureSampler::Filter::TRILINEAR
            || sampler.filter() == TextureSampler::Filter::ANISOTROPIC)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Textures without mipmaps do not support "
            "trilinear or anisotropic sampling.");
        sampler.set_filter(TextureSampler::Filter::BILINEAR);
    }
    auto handle = device()->create_texture(
        pixel_storage_to_format<float>(storage), 3u,
        size.x, size.y, size.z, valid_mip_levels,
        sampler, this->handle(), index);
    _texture_slots[index] = handle;
    return {handle, storage, valid_mip_levels, size};
}

void Heap::destroy_texture(uint32_t index) noexcept {
    if (auto &&h = _texture_slots[index]; h == invalid_handle) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Destroying already destroyed heap texture at slot {} in heap #{}.",
            index, handle());
    } else {
        device()->destroy_texture(h);
        h = invalid_handle;
    }
}

void Heap::destroy_buffer(uint32_t index) noexcept {
    if (auto &&h = _buffer_slots[index]; h == invalid_handle) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Destroying already destroyed heap buffer at slot {} in heap #{}.",
            index, handle());
    } else {
        device()->destroy_buffer(h);
        h = invalid_handle;
    }
}

}// namespace luisa::compute
