#pragma once

#include <cstdint>
#include <limits>

#include <luisa/core/basic_types.h>

namespace luisa::compute::detail {

enum class SparseTextureTileRegionStatus : uint8_t {
    SUCCESS,
    INVALID_DIMENSION,
    ZERO_EXTENT,
    ZERO_TILE_SIZE,
    MIP_OUT_OF_RANGE,
    ZERO_TILE_COUNT,
    ARITHMETIC_OVERFLOW,
    OUT_OF_RANGE
};

[[nodiscard]] constexpr const char *
sparse_texture_tile_region_status_name(
    SparseTextureTileRegionStatus status) noexcept {
    switch (status) {
        case SparseTextureTileRegionStatus::SUCCESS: return "success";
        case SparseTextureTileRegionStatus::INVALID_DIMENSION: return "invalid dimension";
        case SparseTextureTileRegionStatus::ZERO_EXTENT: return "zero texture extent";
        case SparseTextureTileRegionStatus::ZERO_TILE_SIZE: return "zero tile size";
        case SparseTextureTileRegionStatus::MIP_OUT_OF_RANGE: return "mip level out of range";
        case SparseTextureTileRegionStatus::ZERO_TILE_COUNT: return "zero tile count";
        case SparseTextureTileRegionStatus::ARITHMETIC_OVERFLOW: return "tile range arithmetic overflow";
        case SparseTextureTileRegionStatus::OUT_OF_RANGE: return "tile range out of bounds";
    }
    return "unknown";
}

struct SparseTextureTileRegionPlan {
    SparseTextureTileRegionStatus status;
    uint3 mip_extent;
    uint3 tile_grid;
    uint3 end_tile;
    uint3 texel_offset;
    uint3 texel_extent;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseTextureTileRegionStatus::SUCCESS;
    }
};

struct SparseTextureTileRegionRequest {
    uint32_t dimension;
    uint3 base_extent;
    uint3 tile_size;
    uint32_t mip_levels;
    uint32_t mip_level;
    uint3 start_tile;
    uint3 tile_count;
};

struct SparseTextureTileAxisPlan {
    SparseTextureTileRegionStatus status;
    uint32_t mip_extent;
    uint32_t tile_grid;
    uint32_t end_tile;
    uint32_t texel_offset;
    uint32_t texel_extent;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseTextureTileRegionStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr SparseTextureTileAxisPlan
plan_sparse_texture_tile_axis(
    uint32_t base_extent,
    uint32_t tile_size,
    uint32_t mip_level,
    uint32_t start_tile,
    uint32_t tile_count) noexcept {
    if (base_extent == 0u) {
        return {
            .status = SparseTextureTileRegionStatus::ZERO_EXTENT};
    }
    if (tile_size == 0u) {
        return {
            .status = SparseTextureTileRegionStatus::ZERO_TILE_SIZE};
    }
    if (tile_count == 0u) {
        return {
            .status = SparseTextureTileRegionStatus::ZERO_TILE_COUNT};
    }
    auto shifted_extent =
        mip_level < std::numeric_limits<uint32_t>::digits ?
            base_extent >> mip_level :
            0u;
    auto mip_extent = shifted_extent == 0u ? 1u : shifted_extent;
    auto tile_grid =
        mip_extent / tile_size +
        static_cast<uint32_t>(mip_extent % tile_size != 0u);
    if (tile_count >
        std::numeric_limits<uint32_t>::max() - start_tile) {
        return {
            .status =
                SparseTextureTileRegionStatus::ARITHMETIC_OVERFLOW};
    }
    auto end_tile = start_tile + tile_count;
    if (end_tile > tile_grid) {
        return {
            .status = SparseTextureTileRegionStatus::OUT_OF_RANGE};
    }
    auto texel_offset =
        static_cast<uint64_t>(start_tile) * tile_size;
    auto nominal_extent =
        static_cast<uint64_t>(tile_count) * tile_size;
    if (texel_offset >= mip_extent) {
        return {
            .status = SparseTextureTileRegionStatus::OUT_OF_RANGE};
    }
    auto remaining_extent =
        static_cast<uint64_t>(mip_extent) - texel_offset;
    return {
        .status = SparseTextureTileRegionStatus::SUCCESS,
        .mip_extent = mip_extent,
        .tile_grid = tile_grid,
        .end_tile = end_tile,
        .texel_offset = static_cast<uint32_t>(texel_offset),
        .texel_extent = static_cast<uint32_t>(
            nominal_extent < remaining_extent ?
                nominal_extent :
                remaining_extent)};
}

[[nodiscard]] constexpr SparseTextureTileRegionPlan
plan_sparse_texture_tile_region(
    SparseTextureTileRegionRequest request) noexcept {
    SparseTextureTileRegionPlan plan{
        .status = SparseTextureTileRegionStatus::SUCCESS,
        .mip_extent = uint3{1u},
        .tile_grid = uint3{1u},
        .end_tile = uint3{1u},
        .texel_offset = uint3{0u},
        .texel_extent = uint3{1u}};
    if (request.dimension != 2u && request.dimension != 3u) {
        plan.status = SparseTextureTileRegionStatus::INVALID_DIMENSION;
        return plan;
    }
    if (request.mip_level >= request.mip_levels) {
        plan.status = SparseTextureTileRegionStatus::MIP_OUT_OF_RANGE;
        return plan;
    }
    auto x = plan_sparse_texture_tile_axis(
        request.base_extent.x, request.tile_size.x,
        request.mip_level, request.start_tile.x,
        request.tile_count.x);
    if (!x) {
        plan.status = x.status;
        return plan;
    }
    plan.mip_extent.x = x.mip_extent;
    plan.tile_grid.x = x.tile_grid;
    plan.end_tile.x = x.end_tile;
    plan.texel_offset.x = x.texel_offset;
    plan.texel_extent.x = x.texel_extent;

    auto y = plan_sparse_texture_tile_axis(
        request.base_extent.y, request.tile_size.y,
        request.mip_level, request.start_tile.y,
        request.tile_count.y);
    if (!y) {
        plan.status = y.status;
        return plan;
    }
    plan.mip_extent.y = y.mip_extent;
    plan.tile_grid.y = y.tile_grid;
    plan.end_tile.y = y.end_tile;
    plan.texel_offset.y = y.texel_offset;
    plan.texel_extent.y = y.texel_extent;

    if (request.dimension == 3u) {
        auto z = plan_sparse_texture_tile_axis(
            request.base_extent.z, request.tile_size.z,
            request.mip_level, request.start_tile.z,
            request.tile_count.z);
        if (!z) {
            plan.status = z.status;
            return plan;
        }
        plan.mip_extent.z = z.mip_extent;
        plan.tile_grid.z = z.tile_grid;
        plan.end_tile.z = z.end_tile;
        plan.texel_offset.z = z.texel_offset;
        plan.texel_extent.z = z.texel_extent;
    }
    return plan;
}

}// namespace luisa::compute::detail
