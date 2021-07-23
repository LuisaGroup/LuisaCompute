//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/basic_types.h>
#include <runtime/texture.h>

namespace luisa::compute {

namespace detail {

template<typename T>
class Expr;

class TextureRef2D;
class TextureRef3D;

template<typename T>
class BufferRef;

}

class Heap : concepts::Noncopyable {

public:
    static constexpr auto slot_count = 65536u;
    static constexpr auto invalid_handle = std::numeric_limits<uint64_t>::max();

private:
    Device::Handle _device;
    uint64_t _handle{};
    size_t _capacity{};
    std::vector<uint64_t> _texture_slots;
    std::vector<uint64_t> _buffer_slots;

private:
    friend class Device;
    Heap(Device::Handle device, size_t capacity) noexcept;
    [[nodiscard]] static constexpr auto _compute_mip_levels(uint3 size, uint requested_levels) noexcept;
    void _destroy() noexcept;

public:
    Heap() noexcept = default;
    Heap(Heap &&another) noexcept = default;
    Heap &operator=(Heap &&rhs) noexcept;
    ~Heap() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto allocated_size() const noexcept { return _device->query_heap_memory_usage(_handle); }

    template<typename T>
    [[nodiscard]] BufferView<T> create_buffer(uint index, size_t size) noexcept {
        if (auto h = _buffer_slots[index]; h != invalid_handle) {
            LUISA_WARNING_WITH_LOCATION(
                "Overwriting buffer #{} at {} in heap #{}.",
                h, index, _handle);
            destroy_buffer(index);
        }
        auto buffer_handle = _device->create_buffer(size, _handle, index);
        return {buffer_handle, 0u, size};
    }
    void destroy_buffer(uint32_t index) noexcept;

    [[nodiscard]] Texture2D create_texture(uint index, PixelStorage storage, uint2 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    [[nodiscard]] Texture3D create_texture(uint index, PixelStorage storage, uint3 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
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
