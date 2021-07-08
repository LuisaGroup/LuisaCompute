//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/basic_types.h>
#include <core/mathematics.h>
#include <runtime/image.h>
#include <runtime/volume.h>
#include <runtime/buffer.h>
#include <runtime/texture_sampler.h>

namespace luisa::compute {

namespace detail {

template<typename T>
class Expr;

}

class TextureHeap : concepts::Noncopyable {

public:
    static constexpr auto max_slot_count = 65536u;
    static constexpr auto invalid_index = std::numeric_limits<uint32_t>::max();
    static constexpr auto invalid_handle = std::numeric_limits<uint64_t>::max();

private:
    class TextureDesc {

    private:
        uint64_t _handle{invalid_handle};
        PixelStorage _storage{};
        uint _mipmap_levels{0u};
        uint _dimension{};
        uint _size[3]{};

    public:
        constexpr TextureDesc() noexcept = default;
        constexpr TextureDesc(uint64_t handle, PixelStorage storage, uint dim, uint3 size, uint mipmap_levels) noexcept
            : _handle{handle},
              _storage{storage},
              _mipmap_levels{mipmap_levels},
              _dimension{3u},
              _size{size.x, size.y, size.z} {}
        [[nodiscard]] auto handle() const noexcept { return _handle; }
        [[nodiscard]] auto storage() const noexcept { return _storage; }
        [[nodiscard]] auto mip_levels() const noexcept { return _mipmap_levels; }
        [[nodiscard]] auto dimension() const noexcept { return _dimension; }
        [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
        [[nodiscard]] auto invalid() const noexcept { return _handle == invalid_handle; }
        void invalidate() noexcept { _handle = invalid_handle; }
    };

private:
    Device::Interface *_device;
    uint64_t _handle;
    size_t _capacity;
    std::vector<TextureDesc> _slots;
    std::vector<uint32_t> _available;

private:
    friend class Device;
    TextureHeap(Device &device, size_t capacity) noexcept;
    [[nodiscard]] static constexpr auto _compute_mipmap_levels(uint3 size, uint requested_levels) noexcept;
    void _destroy() noexcept;

    template<uint dim>
    [[nodiscard]] bool _validate_mipmap_level(uint index, uint level) const noexcept;

public:
    TextureHeap(TextureHeap &&another) noexcept;
    TextureHeap &operator=(TextureHeap &&rhs) noexcept;
    ~TextureHeap() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto allocated_size() const noexcept { return _device->query_texture_heap_memory_usage(_handle); }
    [[nodiscard]] uint32_t allocate(PixelStorage storage, uint2 size, TextureSampler sampler, uint mipmap_levels = 1u) noexcept;
    [[nodiscard]] uint32_t allocate(PixelStorage storage, uint3 size, TextureSampler sampler, uint mipmap_levels = 1u) noexcept;
    void recycle(uint32_t index) noexcept;

    [[nodiscard]] CommandHandle emplace(uint32_t index, ImageView<float> view, uint32_t mipmap_level = 0u) noexcept;
    [[nodiscard]] CommandHandle emplace(uint32_t index, VolumeView<float> view, uint32_t mipmap_level = 0u) noexcept;
    [[nodiscard]] CommandHandle emplace(uint32_t index, const void *pixels, uint32_t mipmap_level = 0u) noexcept;

    template<typename T>
    [[nodiscard]] CommandHandle emplace(uint32_t index, BufferView<T> buffer, uint32_t mipmap_level = 0u) noexcept {
        if (!_validate_mipmap_level<0>(index, mipmap_level)) { return nullptr; }
        auto tex = _slots[index];
        auto mipmap_size = max(tex.size() >> mipmap_level, 1u);
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            tex.handle(), tex.storage(),
            mipmap_level, uint3(0u), mipmap_size,
            _handle);
    }

    // see implementations in dsl/expr.h
    template<typename I, typename Coord>
    [[nodiscard]] detail::Expr<float4> sample(I &&index, Coord &&coord) const noexcept;

    template<typename I, typename Coord, typename Level>
    [[nodiscard]] detail::Expr<float4> sample(I &&index, Coord &&coord, Level &&level) const noexcept;

    template<typename I, typename Coord, typename DPDX, typename DPDY>
    [[nodiscard]] detail::Expr<float4> sample(I &&index, Coord &&coord, DPDX &&dpdx, DPDY &&dpdy) const noexcept;
};

}// namespace luisa::compute
