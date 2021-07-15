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

template<typename Texture>
[[nodiscard]] inline auto validate_mip_level(Texture t, uint level) noexcept {
    auto valid = level < t.mip_levels();
    if (!valid) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid mipmap level {} (max = {}) for heap texture #{}.",
            level, t.mip_levels() - 1u, t.handle());
    }
    return valid;
}

class Texture2D {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _mip_levels;
    uint2 _size;

public:
    Texture2D(uint64_t handle, PixelStorage storage, uint mip_levels, uint2 size) noexcept
        : _handle{handle}, _storage{storage}, _mip_levels{mip_levels}, _size{size} {}

    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    [[nodiscard]] PixelStorage storage() const noexcept { return _storage; }
    [[nodiscard]] uint mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] uint2 size() const noexcept { return _size; }

    [[nodiscard]] CommandHandle load(const void *pixels, uint mip_level = 0u) noexcept;
    [[nodiscard]] CommandHandle load(ImageView<float> image, uint mip_level = 0u) noexcept;

    template<typename T>
    [[nodiscard]] CommandHandle load(BufferView<T> buffer, uint mip_level = 0u) noexcept {
        if (!validate_mip_level(*this, mip_level)) { return nullptr; }
        auto mipmap_size = max(_size >> mip_level, 1u);
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage,
            mip_level, make_uint3(0u), make_uint3(mipmap_size, 1u));
    }
};

class Texture3D {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _mip_levels;
    uint3 _size;

public:
    Texture3D(uint64_t handle, PixelStorage storage, uint mip_levels, uint3 size) noexcept
        : _handle{handle},
          _storage{storage},
          _mip_levels{mip_levels},
          _size{size} {}

    [[nodiscard]] uint64_t handle() const noexcept { return _handle; }
    [[nodiscard]] PixelStorage storage() const noexcept { return _storage; }
    [[nodiscard]] uint mip_levels() const noexcept { return _mip_levels; }
    [[nodiscard]] uint3 size() const noexcept { return _size; }

    [[nodiscard]] CommandHandle load(const void *pixels, uint mip_level = 0u) noexcept;
    [[nodiscard]] CommandHandle load(VolumeView<float> image, uint mip_level = 0u) noexcept;

    template<typename T>
    [[nodiscard]] CommandHandle load(BufferView<T> buffer, uint mip_level = 0u) noexcept {
        if (!validate_mip_level(*this, mip_level)) { return nullptr; }
        auto mipmap_size = max(_size >> mip_level, 1u);
        return BufferToTextureCopyCommand::create(
            buffer.handle(), buffer.offset_bytes(),
            _handle, _storage,
            mip_level, make_uint3(0u), mipmap_size);
    }
};

}// namespace detail

class TextureHeap : concepts::Noncopyable {

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
    TextureHeap(Device::Handle device, size_t capacity) noexcept;
    [[nodiscard]] static constexpr auto _compute_mip_levels(uint3 size, uint requested_levels) noexcept;
    void _destroy() noexcept;

public:
    TextureHeap() noexcept = default;
    TextureHeap(TextureHeap &&another) noexcept;
    TextureHeap &operator=(TextureHeap &&rhs) noexcept;
    ~TextureHeap() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return _device != nullptr; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] auto allocated_size() const noexcept { return _device->query_texture_heap_memory_usage(_handle); }
    [[nodiscard]] detail::Texture2D create2d(uint index, PixelStorage storage, uint2 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    [[nodiscard]] detail::Texture3D create3d(uint index, PixelStorage storage, uint3 size, TextureSampler sampler = TextureSampler{}, uint mip_levels = 1u) noexcept;
    void destroy(uint32_t index) noexcept;

    // see implementations in dsl/expr.h
    template<typename I>
    [[nodiscard]] detail::Expr<float4> sample2d(I &&index, detail::Expr<float2> uv) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<float4> sample2d(I &&index, detail::Expr<float2> uv, detail::Expr<float> level) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<float4> sample2d(I &&index, detail::Expr<float2> uv, detail::Expr<float2> dpdx, detail::Expr<float2> dpdy) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<float4> sample3d(I &&index, detail::Expr<float3> uvw) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<float4> sample3d(I &&index, detail::Expr<float3> uvw, detail::Expr<float> level) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<float4> sample3d(I &&index, detail::Expr<float3> uvw, detail::Expr<float3> dpdx, detail::Expr<float3> dpdy) const noexcept;

    template<typename I>
    [[nodiscard]] detail::Expr<uint2> size2d(I &&index) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<uint3> size3d(I &&index) const noexcept;
    template<typename I, typename Level>
    [[nodiscard]] detail::Expr<uint2> size2d(I &&index, Level &&level) const noexcept;
    template<typename I, typename Level>
    [[nodiscard]] detail::Expr<uint3> size3d(I &&index, Level &&level) const noexcept;

    template<typename I>
    [[nodiscard]] detail::Expr<uint2> read2d(I &&index, detail::Expr<uint2> coord) const noexcept;
    template<typename I>
    [[nodiscard]] detail::Expr<uint3> read3d(I &&index, detail::Expr<uint3> coord) const noexcept;
    template<typename I, typename Level>
    [[nodiscard]] detail::Expr<uint2> read2d(I &&index, detail::Expr<uint2> coord, Level &&level) const noexcept;
    template<typename I, typename Level>
    [[nodiscard]] detail::Expr<uint3> read3d(I &&index, detail::Expr<uint3> coord, Level &&level) const noexcept;
};

}// namespace luisa::compute
