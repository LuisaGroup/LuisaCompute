//
// Created by Mike Smith on 2021/4/7.
//

#pragma once

#include <core/mathematics.h>
#include <runtime/image.h>
#include <runtime/buffer.h>

namespace luisa::compute {

class TextureSampler {

public:
    enum struct CoordMode : uint8_t {
        NORMALIZED,
        PIXEL
    };

    enum struct AddressMode : uint8_t {
        EDGE,
        REPEAT,
        MIRROR
    };

    enum struct FilterMode : uint8_t {
        NEAREST,
        LINEAR
    };

private:
    std::array<AddressMode, 3u> _address{AddressMode::EDGE, AddressMode::EDGE, AddressMode::EDGE};
    FilterMode _filter{FilterMode::NEAREST};
    FilterMode _mipmap_filter{FilterMode::NEAREST};
    CoordMode _coord{CoordMode::NORMALIZED};

public:
    void set_address_mode(AddressMode mode) noexcept { _address[0] = _address[1] = _address[2] = mode; }
    void set_address_mode(AddressMode u, AddressMode v, AddressMode w = AddressMode::EDGE) noexcept { _address = {u, v, w}; }
    void set_filter_mode(FilterMode mode) noexcept { _filter = mode; }
    void set_mipmap_filter_mode(FilterMode mode) noexcept { _mipmap_filter = mode; }
    void set_coord_mode(CoordMode mode) noexcept { _coord = mode; }
    [[nodiscard]] auto address_mode() const noexcept { return _address; }
    [[nodiscard]] auto filter_mode() const noexcept { return _filter; }
    [[nodiscard]] auto mipmap_filter_mode() const noexcept { return _mipmap_filter; }
    [[nodiscard]] auto coord_mode() const noexcept { return _coord; }
};

namespace detail {

}// namespace detail

class TextureHeap : concepts::Noncopyable {

    struct TextureDesc {
        uint64_t handle{0u};
        PixelStorage storage{};
        uint mipmap_levels{0u};
        uint2 size;
        TextureSampler sampler;
        constexpr TextureDesc() noexcept = default;
        constexpr TextureDesc(uint64_t handle, PixelStorage storage, uint2 size, uint mipmap_levels, TextureSampler sampler) noexcept
            : handle{handle}, storage{storage},
              mipmap_levels{mipmap_levels}, size{size},
              sampler{sampler} {}
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
    [[nodiscard]] static constexpr auto _valid_mipmap_levels(uint width, uint height, uint requested_levels) noexcept;
    void _destroy() noexcept;
    [[nodiscard]] bool _validate_mipmap_level(uint index, uint level) const noexcept;

public:
    TextureHeap(TextureHeap &&another) noexcept;
    TextureHeap &operator=(TextureHeap &&rhs) noexcept;
    ~TextureHeap() noexcept;
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto size() const noexcept { return _device->query_texture_heap_memory_usage(_handle); }
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
