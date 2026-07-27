// Test for runtime sparse texture tile-region validation.
// This test covers:
// - ceil-divided tile grids and partial edge tiles
// - selected-mip geometry
// - zero counts, invalid mip levels, bounds, and integer overflow

#include "ut/ut.hpp"

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/sparse_image.h>
#include <luisa/runtime/sparse_volume.h>

#include "sparse_texture_tile_plan.h"
#include "sparse_tile_allocation_plan.h"

#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;
namespace runtime_detail = luisa::compute::detail;

template<typename SparseTexture, typename Coord>
constexpr auto sparse_buffer_copy_overloads_compile =
    requires(const SparseTexture &texture,
             const Buffer<float> &buffer,
             BufferView<float> buffer_view) {
        texture.copy_from(
            Coord{0u}, Coord{1u}, 0u, buffer_view);
        texture.copy_from(
            Coord{0u}, Coord{1u}, 0u, buffer);
        texture.copy_to(
            Coord{0u}, Coord{1u}, 0u, buffer_view);
        texture.copy_to(
            Coord{0u}, Coord{1u}, 0u, buffer);
    };

static_assert(sparse_buffer_copy_overloads_compile<
              SparseImage<float>, uint2>);
static_assert(sparse_buffer_copy_overloads_compile<
              SparseVolume<float>, uint3>);

template<typename SparseTexture, typename Coord>
void instantiate_sparse_buffer_copy_overloads(
    const SparseTexture &texture,
    const Buffer<float> &buffer,
    BufferView<float> buffer_view) {
    [[maybe_unused]] auto upload_view = texture.copy_from(
        Coord{0u}, Coord{1u}, 0u, buffer_view);
    [[maybe_unused]] auto upload_buffer = texture.copy_from(
        Coord{0u}, Coord{1u}, 0u, buffer);
    [[maybe_unused]] auto download_view = texture.copy_to(
        Coord{0u}, Coord{1u}, 0u, buffer_view);
    [[maybe_unused]] auto download_buffer = texture.copy_to(
        Coord{0u}, Coord{1u}, 0u, buffer);
}

template void instantiate_sparse_buffer_copy_overloads<
    SparseImage<float>, uint2>(
    const SparseImage<float> &,
    const Buffer<float> &,
    BufferView<float>);
template void instantiate_sparse_buffer_copy_overloads<
    SparseVolume<float>, uint3>(
    const SparseVolume<float> &,
    const Buffer<float> &,
    BufferView<float>);

void instantiate_sparse_texture_tile_updates(
    SparseImage<float> &image,
    SparseVolume<float> &volume,
    const SparseTextureHeap &heap) {
    [[maybe_unused]] auto image_map =
        image.map_tile(uint2{0u}, uint2{1u}, 0u, heap);
    [[maybe_unused]] auto image_unmap =
        image.unmap_tile(uint2{0u}, uint2{1u}, 0u);
    [[maybe_unused]] auto volume_map =
        volume.map_tile(uint3{0u}, uint3{1u}, 0u, heap);
    [[maybe_unused]] auto volume_unmap =
        volume.unmap_tile(uint3{0u}, uint3{1u}, 0u);
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "sparse_texture_tile_plan_accepts_partial_edge_tiles"_test = [] {
        constexpr auto plan = runtime_detail::plan_sparse_texture_tile_region({.dimension = 2u,
                                                                               .base_extent = uint3{257u, 129u, 1u},
                                                                               .tile_size = uint3{128u, 64u, 1u},
                                                                               .mip_levels = 9u,
                                                                               .mip_level = 0u,
                                                                               .start_tile = uint3{2u, 2u, 0u},
                                                                               .tile_count = uint3{1u, 1u, 1u}});
        static_assert(plan);
        static_assert(plan.mip_extent.x == 257u &&
                      plan.mip_extent.y == 129u);
        static_assert(plan.tile_grid.x == 3u &&
                      plan.tile_grid.y == 3u);
        static_assert(plan.texel_offset.x == 256u &&
                      plan.texel_offset.y == 128u);
        static_assert(plan.texel_extent.x == 1u &&
                      plan.texel_extent.y == 1u);
        expect(static_cast<bool>(plan));
        expect(plan.end_tile.x == 3u && plan.end_tile.y == 3u)
            << "the final partial tile is part of the logical tile grid";
    };

    "sparse_texture_tile_plan_uses_selected_mip_extent"_test = [] {
        constexpr auto plan = runtime_detail::plan_sparse_texture_tile_region({.dimension = 3u,
                                                                               .base_extent = uint3{515u, 259u, 67u},
                                                                               .tile_size = uint3{128u, 64u, 16u},
                                                                               .mip_levels = 10u,
                                                                               .mip_level = 1u,
                                                                               .start_tile = uint3{2u, 2u, 2u},
                                                                               .tile_count = uint3{1u, 1u, 1u}});
        static_assert(plan);
        static_assert(plan.mip_extent.x == 257u &&
                      plan.mip_extent.y == 129u &&
                      plan.mip_extent.z == 33u);
        static_assert(plan.tile_grid.x == 3u &&
                      plan.tile_grid.y == 3u &&
                      plan.tile_grid.z == 3u);
        static_assert(plan.texel_offset.x == 256u &&
                      plan.texel_offset.y == 128u &&
                      plan.texel_offset.z == 32u);
        static_assert(plan.texel_extent.x == 1u &&
                      plan.texel_extent.y == 1u &&
                      plan.texel_extent.z == 1u);
        expect(static_cast<bool>(plan));

        constexpr auto outside_selected_mip =
            runtime_detail::plan_sparse_texture_tile_region({.dimension = 3u,
                                                             .base_extent = uint3{515u, 259u, 67u},
                                                             .tile_size = uint3{128u, 64u, 16u},
                                                             .mip_levels = 10u,
                                                             .mip_level = 1u,
                                                             .start_tile = uint3{3u, 0u, 0u},
                                                             .tile_count = uint3{1u}});
        expect(outside_selected_mip.status ==
               runtime_detail::SparseTextureTileRegionStatus::OUT_OF_RANGE)
            << "bounds come from the selected mip rather than the base extent";
    };

    "sparse_texture_tile_plan_rejects_zero_count_and_invalid_mip"_test = [] {
        using Status = runtime_detail::SparseTextureTileRegionStatus;
        constexpr auto zero_count = runtime_detail::plan_sparse_texture_tile_region({.dimension = 2u,
                                                                                     .base_extent = uint3{64u, 64u, 1u},
                                                                                     .tile_size = uint3{32u, 32u, 1u},
                                                                                     .mip_levels = 7u,
                                                                                     .mip_level = 0u,
                                                                                     .start_tile = uint3{0u},
                                                                                     .tile_count = uint3{1u, 0u, 1u}});
        expect(zero_count.status == Status::ZERO_TILE_COUNT)
            << "map and unmap regions must both contain tiles";

        constexpr auto invalid_mip =
            runtime_detail::plan_sparse_texture_tile_region({.dimension = 2u,
                                                             .base_extent = uint3{64u, 64u, 1u},
                                                             .tile_size = uint3{32u, 32u, 1u},
                                                             .mip_levels = 3u,
                                                             .mip_level = 3u,
                                                             .start_tile = uint3{0u},
                                                             .tile_count = uint3{1u}});
        expect(invalid_mip.status == Status::MIP_OUT_OF_RANGE);
    };

    "sparse_texture_tile_plan_rejects_bounds_and_overflow"_test = [] {
        using Status = runtime_detail::SparseTextureTileRegionStatus;
        constexpr auto out_of_range =
            runtime_detail::plan_sparse_texture_tile_region({.dimension = 2u,
                                                             .base_extent = uint3{65u, 64u, 1u},
                                                             .tile_size = uint3{32u, 32u, 1u},
                                                             .mip_levels = 7u,
                                                             .mip_level = 0u,
                                                             .start_tile = uint3{2u, 0u, 0u},
                                                             .tile_count = uint3{2u, 1u, 1u}});
        expect(out_of_range.status == Status::OUT_OF_RANGE);

        constexpr auto max_u32 =
            std::numeric_limits<uint32_t>::max();
        constexpr auto overflow =
            runtime_detail::plan_sparse_texture_tile_region({.dimension = 3u,
                                                             .base_extent = uint3{64u},
                                                             .tile_size = uint3{1u},
                                                             .mip_levels = 7u,
                                                             .mip_level = 0u,
                                                             .start_tile = uint3{max_u32, 0u, 0u},
                                                             .tile_count = uint3{2u, 1u, 1u}});
        expect(overflow.status == Status::ARITHMETIC_OVERFLOW);

        constexpr auto wide_copy =
            runtime_detail::plan_sparse_texture_tile_region({.dimension = 2u,
                                                             .base_extent = uint3{max_u32, 1u, 1u},
                                                             .tile_size = uint3{2u, 1u, 1u},
                                                             .mip_levels = 32u,
                                                             .mip_level = 0u,
                                                             .start_tile = uint3{0u},
                                                             .tile_count = uint3{uint32_t{1u} << 31u, 1u, 1u}});
        static_assert(wide_copy);
        expect(wide_copy.texel_extent.x == max_u32)
            << "tile-to-texel multiplication must not wrap before edge clipping";
    };

    "sparse_texture_buffer_copy_overloads_match_command_abi"_test = [] {
        expect(sparse_buffer_copy_overloads_compile<
               SparseImage<float>, uint2>);
        expect(sparse_buffer_copy_overloads_compile<
               SparseVolume<float>, uint3>);
    };

    "sparse_tile_allocation_uses_physical_tile_bytes"_test = [] {
        using Status = runtime_detail::SparseTileAllocationStatus;
        constexpr auto image = runtime_detail::plan_sparse_tile_allocation(
            2u, uint3{3u, 2u, 1u}, 64u * 1024u);
        static_assert(image);
        expect(image.byte_size == 6u * 64u * 1024u)
            << "partial edge tiles still occupy complete sparse blocks";

        constexpr auto buffer = runtime_detail::plan_sparse_tile_allocation(
            1u, uint3{7u, 1u, 1u}, 128u);
        static_assert(buffer);
        expect(buffer.byte_size == 896u);

        constexpr auto overflow = runtime_detail::plan_sparse_tile_allocation(
            3u, uint3{2u, 2u, 2u},
            std::numeric_limits<size_t>::max() / 4u + 1u);
        expect(overflow.status == Status::ARITHMETIC_OVERFLOW);
    };
}
