#pragma once

#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/stream_event.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/tile_modification.h>

namespace luisa::compute {
class SparseTextureHeap;
namespace detail {
LC_RUNTIME_API void check_sparse_tex2d_map(uint2 size, uint2 tile_size, uint2 start_tile, uint2 tile_count);
LC_RUNTIME_API void check_sparse_tex2d_unmap(uint2 size, uint2 tile_size, uint2 start_tile);
LC_RUNTIME_API void check_sparse_tex3d_map(uint3 size, uint3 tile_size, uint3 start_tile, uint3 tile_count);
LC_RUNTIME_API void check_sparse_tex3d_unmap(uint3 size, uint3 tile_size, uint3 start_tile);
LC_RUNTIME_API void check_tex_heap_match(PixelStorage storage, SparseTextureHeap const& heap);
}// namespace detail

template<typename T>
class Buffer;

template<typename T>
class BufferView;

class LC_RUNTIME_API SparseTexture : public Resource {
public:
protected:
    size_t _tile_size_bytes;
    uint3 _tile_size;
    SparseTexture(DeviceInterface *device, const SparseTextureCreationInfo &info) noexcept;
    SparseTexture(SparseTexture &&) noexcept = default;
    ~SparseTexture() noexcept override;

public:
    // deleted members should be public
    SparseTexture(const SparseTexture &) noexcept = delete;
    SparseTexture &operator=(SparseTexture &&) noexcept = delete;// use _move_from in derived classes
    SparseTexture &operator=(const SparseTexture &) noexcept = delete;

    [[nodiscard]] auto tile_size_bytes() const noexcept {
        _check_is_valid();
        return _tile_size_bytes;
    }
};

}// namespace luisa::compute
