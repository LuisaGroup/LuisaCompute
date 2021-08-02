//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <runtime/texture.h>
#include <runtime/resource.h>

namespace luisa::compute {

namespace detail {

template<typename T>
class Expr;

class TextureRef2D;
class TextureRef3D;

template<typename T>
class BufferRef;

}

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
    [[nodiscard]] static constexpr auto _compute_mip_levels(uint3 size, uint requested_levels) noexcept;

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

    [[nodiscard]] TextureView2D create_texture(uint index, PixelStorage storage, uint2 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    [[nodiscard]] TextureView3D create_texture(uint index, PixelStorage storage, uint3 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    void destroy_texture(uint32_t index) noexcept;

    // see implementations in dsl/expr.h
    template<typename I>
    detail::TextureRef2D tex2d(I &&index) const noexcept;

    template<typename I>
    detail::TextureRef2D tex3d(I &&index) const noexcept;

    template<typename T, typename I>
    detail::BufferRef<T> buffer(I &&index) const noexcept;
};

}// namespace luisa::compute
