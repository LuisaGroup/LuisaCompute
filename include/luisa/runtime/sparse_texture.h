#pragma once

#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/tile_modification.h>

namespace luisa::compute {
class SparseTextureHeap;
namespace detail {
struct SparseTextureTileRegion {
    uint3 offset;
    uint3 extent;
};

LUISA_RUNTIME_API SparseTextureTileRegion check_sparse_texture_tile_region(
    uint32_t dimension,
    uint3 size,
    uint3 tile_size,
    uint32_t mip_levels,
    uint32_t mip_level,
    uint3 start_tile,
    uint3 tile_count) noexcept;
LUISA_RUNTIME_API void check_sparse_texture_copy_buffer_size(
    PixelStorage storage,
    uint3 extent,
    size_t buffer_size) noexcept;
}// namespace detail

template<typename T>
class Buffer;

template<typename T>
class BufferView;

class LUISA_RUNTIME_API SparseTexture : public Resource {
public:
protected:
    size_t _tile_size_bytes;
    uint3 _tile_size;
    SparseTexture(DeviceInterface *device, const SparseTextureCreationInfo &info) noexcept;
    SparseTexture(SparseTexture &&) noexcept = default;
    ~SparseTexture() noexcept override;

public:
    SparseTexture() noexcept : _tile_size_bytes{0}, _tile_size{0} {}
    // deleted members should be public
    SparseTexture(const SparseTexture &) noexcept = delete;
    SparseTexture &operator=(SparseTexture &&) noexcept = delete;// use _move_from in derived classes
    SparseTexture &operator=(const SparseTexture &) noexcept = delete;
    using Resource::operator bool;
    using Resource::release;
    [[nodiscard]] auto tile_size_bytes() const noexcept {
        _check_is_valid();
        return _tile_size_bytes;
    }
};

}// namespace luisa::compute
