//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/basic_types.h>
#include <core/mathematics.h>
#include <runtime/image.h>
#include <runtime/buffer.h>
#include <runtime/texture_sampler.h>

namespace luisa::compute {

class TextureHeap : concepts::Noncopyable {

    struct TextureDesc {
        uint64_t handle{0u};
        PixelStorage storage{};
        uint mipmap_levels{0u};
        uint2 size;
        constexpr TextureDesc() noexcept = default;
        constexpr TextureDesc(uint64_t handle, PixelStorage storage, uint2 size, uint mipmap_levels) noexcept
            : handle{handle}, storage{storage},
              mipmap_levels{mipmap_levels}, size{size} {}
    };

public:
    static constexpr auto max_slot_count = 65536u;
    static constexpr auto invalid_index = std::numeric_limits<uint32_t>::max();
    static constexpr auto invalid_handle = std::numeric_limits<uint64_t>::max();

private:
    Device::Interface *_device;
    uint64_t _handle;
    size_t _capacity;
    std::vector<TextureDesc> _slots;
    std::vector<uint32_t> _available;

private:
    friend class Device;
    TextureHeap(Device &device, size_t capacity) noexcept;
    [[nodiscard]] static constexpr auto _compute_mipmap_levels(uint width, uint height, uint requested_levels) noexcept;
    void _destroy() noexcept;
    [[nodiscard]] bool _validate_mipmap_level(uint index, uint level) const noexcept;

public:
    TextureHeap(TextureHeap &&another) noexcept;
    TextureHeap &operator=(TextureHeap &&rhs) noexcept;
    ~TextureHeap() noexcept;
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto allocated_size() const noexcept { return _device->query_texture_heap_memory_usage(_handle); }
    [[nodiscard]] uint32_t allocate(PixelStorage storage, uint2 size, TextureSampler sampler, uint mipmap_levels = 1u) noexcept;
    void recycle(uint32_t index) noexcept;
    [[nodiscard]] CommandHandle emplace(uint32_t index, ImageView<float> view, uint32_t mipmap_level = 0u) noexcept;
    [[nodiscard]] CommandHandle emplace(uint32_t index, const void *pixels, uint32_t mipmap_level = 0u) noexcept;

    template<typename T>
    [[nodiscard]] CommandHandle emplace(uint32_t index, BufferView<T> buffer, uint32_t mipmap_level = 0u) noexcept {
        if (!_validate_mipmap_level(index, mipmap_level)) { return nullptr; }
        auto tex = _slots[index];
        auto mipmap_size = max(tex.size >> mipmap_level, 1u);
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            tex.handle, tex.storage,
            mipmap_level, uint3(0u), uint3(mipmap_size, 1u),
            _handle);
    }
};

}// namespace luisa::compute
