#include <runtime/sparse_buffer.h>
#include <core/logging.h>
namespace luisa::compute {
namespace detail {
LC_RUNTIME_API void check_sparse_buffer_map(size_t size_bytes, size_t tile_size, uint start_tile, uint tile_count) {
    if ((start_tile + tile_count) > (size_bytes / tile_size)) [[unlikely]] {
        LUISA_ERROR("Map tile [{}, {}] out of tile range {}.", start_tile, start_tile + tile_count, size_bytes / tile_size);
    }
    if (tile_count == 0) [[unlikely]] {
        LUISA_ERROR("Tile count can not be zero.");
    }
}
LC_RUNTIME_API void check_sparse_buffer_unmap(size_t size_bytes, size_t tile_size, uint start_tile) {
    if ((start_tile) >= (size_bytes / tile_size)) [[unlikely]] {
        LUISA_ERROR("Unmap Tile {} out of tile range {}.", start_tile, size_bytes / tile_size);
    }
}
}// namespace detail
void SparseBufferClearTiles::operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept {
    device->clear_sparse_buffer(stream_handle, handle);
}
void SparseBufferUpdateTiles::operator()(DeviceInterface *device, uint64_t stream_handle) && noexcept {
    device->update_sparse_buffer(stream_handle, handle, std::move(operations));
}
}// namespace luisa::compute
