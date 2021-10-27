//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <runtime/sampler.h>
#include <runtime/mipmap.h>
#include <runtime/resource.h>

namespace luisa::compute {

class HeapTexture2D;
class HeapTexture3D;

template<typename T>
class HeapBuffer;

template<typename T>
struct Expr;

class Heap : public Resource {

public:
    static constexpr auto slot_count = 65536u;
    static constexpr auto invalid_handle = std::numeric_limits<uint64_t>::max();

private:
    size_t _capacity{};
    std::vector<uint64_t> _texture_slots;
    std::vector<uint64_t> _buffer_slots;

private:
    friend class Device;
    Heap(Device::Interface *device, size_t capacity) noexcept;

public:
    Heap() noexcept = default;
    using Resource::operator bool;

    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto allocated_size() const noexcept { return device()->query_heap_memory_usage(handle()); }

    template<typename T>
    [[nodiscard]] BufferView<T> create_buffer(uint index, size_t size) noexcept {
        if (auto h = _buffer_slots[index]; h != invalid_handle) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Overwriting buffer #{} at {} in heap #{}.",
                h, index, handle());
            destroy_buffer(index);
        }
        auto buffer_handle = device()->create_buffer(size * sizeof(T), handle(), index);
        _buffer_slots[index] = buffer_handle;
        return {buffer_handle, 0u, size};
    }
    void destroy_buffer(uint32_t index) noexcept;

    template<typename T>
    [[nodiscard]] ImageView<T> create_image(uint index, PixelStorage storage, uint2 size, Sampler sampler = Sampler{}, uint mip_levels = 1u) noexcept {
        if (auto h = _texture_slots[index]; h != invalid_handle) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Overwriting texture #{} at {} in heap #{}.",
                h, index, handle());
            destroy_texture(index);
        }
        auto valid_mip_levels = detail::max_mip_levels(make_uint3(size, 1u), mip_levels);
        if (valid_mip_levels == 1u
            && (sampler.filter() == Sampler::Filter::TRILINEAR
                || sampler.filter() == Sampler::Filter::ANISOTROPIC)) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Textures without mipmaps do not support "
                "trilinear or anisotropic sampling.");
            sampler.set_filter(Sampler::Filter::BILINEAR);
        }
        auto handle = device()->create_texture(
            pixel_storage_to_format<float>(storage), 2u,
            size.x, size.y, 1u, valid_mip_levels,
            sampler, this->handle(), index);
        _texture_slots[index] = handle;
        return {handle, storage, valid_mip_levels, make_uint2(0u), size};
    }

    template<typename T>
    [[nodiscard]] VolumeView<T> create_volume(uint index, PixelStorage storage, uint3 size, Sampler sampler = Sampler{}, uint mip_levels = 1u) noexcept {
        if (auto h = _texture_slots[index]; h != invalid_handle) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Overwriting texture #{} at {} in heap #{}.",
                h, index, handle());
            destroy_texture(index);
        }
        auto valid_mip_levels = detail::max_mip_levels(size, mip_levels);
        if (valid_mip_levels == 1u
            && (sampler.filter() == Sampler::Filter::TRILINEAR
                || sampler.filter() == Sampler::Filter::ANISOTROPIC)) [[unlikely]] {
            LUISA_WARNING_WITH_LOCATION(
                "Textures without mipmaps do not support "
                "trilinear or anisotropic sampling.");
            sampler.set_filter(Sampler::Filter::BILINEAR);
        }
        auto handle = device()->create_texture(
            pixel_storage_to_format<float>(storage), 3u,
            size.x, size.y, size.z, valid_mip_levels,
            sampler, this->handle(), index);
        _texture_slots[index] = handle;
        return {handle, storage, valid_mip_levels, make_uint3(0u), size};
    }

    void destroy_texture(uint32_t index) noexcept;

    // see implementations in dsl/expr.h
    template<typename I>
    HeapTexture2D tex2d(I &&index) const noexcept;

    template<typename I>
    HeapTexture2D tex3d(I &&index) const noexcept;

    template<typename T, typename I>
    HeapBuffer<T> buffer(I &&index) const noexcept;
};

}// namespace luisa::compute
