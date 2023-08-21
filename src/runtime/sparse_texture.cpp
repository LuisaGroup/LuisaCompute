#include <luisa/runtime/sparse_texture.h>
#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

LC_RUNTIME_API void check_sparse_tex2d_map(uint2 size, uint2 tile_size, uint2 start_tile, uint2 tile_count) {
    auto end = start_tile + tile_count;
    auto total_tile = size / tile_size;
    LUISA_ASSERT(
        !(any((end) > total_tile)),
        "Map Tile [({}, {}), ({}, {})] out of tile range({}, {})", start_tile.x, start_tile.y, end.x, end.y, total_tile.x, total_tile.y);
    LUISA_ASSERT(!(any(tile_count == uint2(0))), "Tile count can not be zero.");
}

LC_RUNTIME_API void check_sparse_tex2d_unmap(uint2 size, uint2 tile_size, uint2 start_tile) {
    auto total_tile = size / tile_size;
    LUISA_ASSERT(
        !(any(start_tile >= total_tile)),
        "Map Tile ({}, {}) out of tile range({}, {})", start_tile.x, start_tile.y, total_tile.x, total_tile.y);
}

LC_RUNTIME_API void check_sparse_tex3d_map(uint3 size, uint3 tile_size, uint3 start_tile, uint3 tile_count) {
    auto end = start_tile + tile_count;
    auto total_tile = size / tile_size;
    LUISA_ASSERT(
        !(any((end) > total_tile)),
        "Map Tile [({}, {}, {}), ({}, {}, {})] out of tile range({}, {}, {})", start_tile.x, start_tile.y, start_tile.z, end.x, end.y, end.z,
        total_tile.x, total_tile.y, total_tile.z);
    LUISA_ASSERT(!any(tile_count == uint3(0)), "Tile count can not be zero.");
}

LC_RUNTIME_API void check_sparse_tex3d_unmap(uint3 size, uint3 tile_size, uint3 start_tile) {
    auto total_tile = size / tile_size;
    LUISA_ASSERT(!any(start_tile >= total_tile), "Map Tile ({}, {}, {}) out of tile range({}, {}, {})", start_tile.x, start_tile.y, start_tile.z, total_tile.x, total_tile.y, total_tile.z);
}
LC_RUNTIME_API void check_tex_heap_match(PixelStorage storage, SparseTextureHeap const &heap) {
    LUISA_ASSERT(is_block_compressed(storage) == heap.is_compressed_type(), "Pixel type and texture-heap type not match.");
}

}// namespace detail

SparseTexture::SparseTexture(DeviceInterface *device, const SparseTextureCreationInfo &info) noexcept
    : Resource{device, Tag::SPARSE_TEXTURE, info},
      _tile_size_bytes{info.tile_size_bytes},
      _tile_size{info.tile_size} {
}

SparseTexture::~SparseTexture() noexcept {
    if (*this) { device()->destroy_sparse_texture(handle()); }
}

}// namespace luisa::compute
