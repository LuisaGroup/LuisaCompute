#include <core/logging.h>
namespace luisa::compute ::detail {
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
}// namespace luisa::compute::detail
