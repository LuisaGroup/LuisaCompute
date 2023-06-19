#include <luisa/runtime/sparse_buffer.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

LC_RUNTIME_API void check_sparse_buffer_map(size_t size_bytes, size_t tile_size, uint start_tile, uint tile_count) {
    auto tile_range = (size_bytes + tile_size -1) / tile_size;
    if ((start_tile + tile_count) > tile_range) [[unlikely]] {
        LUISA_ERROR("Map tile [{}, {}] out of tile range {}.", start_tile, start_tile + tile_count, tile_range);
    }
    if (tile_count == 0) [[unlikely]] {
        LUISA_ERROR("Tile count can not be zero.");
    }
}

LC_RUNTIME_API void check_sparse_buffer_unmap(size_t size_bytes, size_t tile_size, uint start_tile) {
    auto tile_range = (size_bytes + tile_size -1) / tile_size;
    if ((start_tile) >= tile_range) [[unlikely]] {
        LUISA_ERROR("Unmap Tile {} out of tile range {}.", start_tile, tile_range);
    }
}

}// namespace detail
}// namespace luisa::compute

