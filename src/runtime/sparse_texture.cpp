#include <luisa/runtime/sparse_texture.h>
#include <luisa/runtime/sparse_heap.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/logging.h>

#include "sparse_texture_tile_plan.h"

namespace luisa::compute {

namespace detail {

LUISA_RUNTIME_API SparseTextureTileRegion check_sparse_texture_tile_region(
    uint32_t dimension,
    uint3 size,
    uint3 tile_size,
    uint32_t mip_levels,
    uint32_t mip_level,
    uint3 start_tile,
    uint3 tile_count) noexcept {
    auto plan = plan_sparse_texture_tile_region({.dimension = dimension,
                                                 .base_extent = size,
                                                 .tile_size = tile_size,
                                                 .mip_levels = mip_levels,
                                                 .mip_level = mip_level,
                                                 .start_tile = start_tile,
                                                 .tile_count = tile_count});
    LUISA_ASSERT(
        static_cast<bool>(plan),
        "Invalid sparse texture tile region at mip {}: start ({}, {}, {}), "
        "count ({}, {}, {}), mip extent ({}, {}, {}), tile grid ({}, {}, {}): {}.",
        mip_level,
        start_tile.x, start_tile.y, start_tile.z,
        tile_count.x, tile_count.y, tile_count.z,
        plan.mip_extent.x, plan.mip_extent.y, plan.mip_extent.z,
        plan.tile_grid.x, plan.tile_grid.y, plan.tile_grid.z,
        sparse_texture_tile_region_status_name(plan.status));
    return {
        .offset = plan.texel_offset,
        .extent = plan.texel_extent};
}

LUISA_RUNTIME_API void check_sparse_texture_copy_buffer_size(
    PixelStorage storage,
    uint3 extent,
    size_t buffer_size) noexcept {
    auto required_size = checked_pixel_storage_size(storage, extent);
    LUISA_ASSERT(
        static_cast<bool>(required_size),
        "Sparse texture copy size overflow for extent ({}, {}, {}) "
        "and pixel storage {}.",
        extent.x, extent.y, extent.z,
        luisa::to_underlying(storage));
    LUISA_ASSERT(
        buffer_size >= required_size.size,
        "Sparse texture copy requires {} bytes but the buffer view "
        "only contains {} bytes.",
        required_size.size, buffer_size);
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
