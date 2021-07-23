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
}

class Heap : concepts::Noncopyable {

public:
    static constexpr auto slot_count = 65536u;
    static constexpr auto invalid_handle = std::numeric_limits<uint64_t>::max();

private:
    Device::Handle _device;
    uint64_t _handle{};
    size_t _capacity{};
    std::vector<uint64_t> _slots;

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
    [[nodiscard]] Texture2D create_tex2d(uint index, PixelStorage storage, uint2 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    [[nodiscard]] Texture3D create_tex3d(uint index, PixelStorage storage, uint3 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    void destroy(uint32_t index) noexcept;

    // see implementations in dsl/expr.h
    template<typename I>
    detail::TextureRef2D tex2d(I &&index) const noexcept;

    template<typename I>
    detail::TextureRef2D tex3d(I &&index) const noexcept;
};

}// namespace luisa::compute
