#pragma once

#include <runtime/rhi/resource.h>
#include <runtime/stream_event.h>
#include <runtime/rhi/command.h>
#include <runtime/rhi/tile_modification.h>

namespace luisa::compute {

namespace detail {
LC_RUNTIME_API void check_sparse_tex2d_map(uint2 size, uint2 tile_size, uint2 start_tile, uint2 tile_count);
LC_RUNTIME_API void check_sparse_tex2d_unmap(uint2 size, uint2 tile_size, uint2 start_tile);
LC_RUNTIME_API void check_sparse_tex3d_map(uint3 size, uint3 tile_size, uint3 start_tile, uint3 tile_count);
LC_RUNTIME_API void check_sparse_tex3d_unmap(uint3 size, uint3 tile_size, uint3 start_tile);
}// namespace detail

template<typename T>
class Buffer;

template<typename T>
class BufferView;

class LC_RUNTIME_API SparseTexture : public Resource {
public:
    struct LC_RUNTIME_API UpdateTiles {
        uint64_t handle;
        luisa::vector<SparseTextureOperation> operations;
        void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
    };
    struct LC_RUNTIME_API ClearTiles {
        uint64_t handle;
        void operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept;
    };

protected:
    size_t _tile_size_bytes;
    uint3 _tile_size;
    luisa::vector<SparseTextureOperation> _operations;
    SparseTexture(DeviceInterface *device, const SparseTextureCreationInfo &info) noexcept;
    SparseTexture(SparseTexture &&) noexcept = default;
    ~SparseTexture() noexcept override;

public:
    // deleted members should be public
    SparseTexture(const SparseTexture &) noexcept = delete;
    SparseTexture &operator=(SparseTexture &&) noexcept = delete;// use _move_from in derived classes
    SparseTexture &operator=(const SparseTexture &) noexcept = delete;

    [[nodiscard]] UpdateTiles update() noexcept;
    [[nodiscard]] ClearTiles clear_tiles() noexcept;
    [[nodiscard]] auto tile_size_bytes() const noexcept { return _tile_size_bytes; }
};

LUISA_MARK_STREAM_EVENT_TYPE(SparseTexture::UpdateTiles)

}// namespace luisa::compute
