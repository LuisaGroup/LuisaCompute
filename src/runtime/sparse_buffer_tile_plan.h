#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace luisa::compute::detail {

enum class SparseBufferTileRegionStatus : uint8_t {
    SUCCESS,
    ZERO_BUFFER_SIZE,
    ZERO_TILE_SIZE,
    ZERO_TILE_COUNT,
    ARITHMETIC_OVERFLOW,
    OUT_OF_RANGE
};

[[nodiscard]] constexpr const char *
sparse_buffer_tile_region_status_name(
    SparseBufferTileRegionStatus status) noexcept {
    switch (status) {
        case SparseBufferTileRegionStatus::SUCCESS: return "success";
        case SparseBufferTileRegionStatus::ZERO_BUFFER_SIZE: return "zero buffer size";
        case SparseBufferTileRegionStatus::ZERO_TILE_SIZE: return "zero tile size";
        case SparseBufferTileRegionStatus::ZERO_TILE_COUNT: return "zero tile count";
        case SparseBufferTileRegionStatus::ARITHMETIC_OVERFLOW: return "tile range arithmetic overflow";
        case SparseBufferTileRegionStatus::OUT_OF_RANGE: return "tile range out of bounds";
    }
    return "unknown";
}

struct SparseBufferTileRegionRequest {
    size_t size_bytes;
    size_t tile_size_bytes;
    uint32_t start_tile;
    uint32_t tile_count;
};

struct SparseBufferTileRegionPlan {
    SparseBufferTileRegionStatus status;
    size_t tile_grid_size;
    uint32_t end_tile;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseBufferTileRegionStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr SparseBufferTileRegionPlan
plan_sparse_buffer_tile_region(
    SparseBufferTileRegionRequest request) noexcept {
    SparseBufferTileRegionPlan plan{
        .status = SparseBufferTileRegionStatus::SUCCESS,
        .tile_grid_size = 0u,
        .end_tile = 0u};
    if (request.size_bytes == 0u) {
        plan.status = SparseBufferTileRegionStatus::ZERO_BUFFER_SIZE;
        return plan;
    }
    if (request.tile_size_bytes == 0u) {
        plan.status = SparseBufferTileRegionStatus::ZERO_TILE_SIZE;
        return plan;
    }
    if (request.tile_count == 0u) {
        plan.status = SparseBufferTileRegionStatus::ZERO_TILE_COUNT;
        return plan;
    }
    plan.tile_grid_size =
        request.size_bytes / request.tile_size_bytes +
        static_cast<size_t>(
            request.size_bytes % request.tile_size_bytes != 0u);
    if (request.tile_count >
        std::numeric_limits<uint32_t>::max() - request.start_tile) {
        plan.status = SparseBufferTileRegionStatus::ARITHMETIC_OVERFLOW;
        return plan;
    }
    plan.end_tile = request.start_tile + request.tile_count;
    if (static_cast<size_t>(plan.end_tile) > plan.tile_grid_size) {
        plan.status = SparseBufferTileRegionStatus::OUT_OF_RANGE;
        return plan;
    }
    return plan;
}

}// namespace luisa::compute::detail
