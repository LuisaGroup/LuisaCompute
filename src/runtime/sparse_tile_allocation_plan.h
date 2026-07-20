#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/basic_types.h>

namespace luisa::compute::detail {

enum class SparseTileAllocationStatus : uint8_t {
    SUCCESS,
    INVALID_DIMENSION,
    ZERO_TILE_SIZE,
    ZERO_TILE_COUNT,
    ARITHMETIC_OVERFLOW
};

[[nodiscard]] constexpr const char *
sparse_tile_allocation_status_name(
    SparseTileAllocationStatus status) noexcept {
    switch (status) {
        case SparseTileAllocationStatus::SUCCESS: return "success";
        case SparseTileAllocationStatus::INVALID_DIMENSION: return "invalid dimension";
        case SparseTileAllocationStatus::ZERO_TILE_SIZE: return "zero tile size";
        case SparseTileAllocationStatus::ZERO_TILE_COUNT: return "zero tile count";
        case SparseTileAllocationStatus::ARITHMETIC_OVERFLOW: return "tile allocation arithmetic overflow";
    }
    return "unknown";
}

struct SparseTileAllocationPlan {
    SparseTileAllocationStatus status;
    size_t byte_size;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseTileAllocationStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr SparseTileAllocationPlan
append_sparse_tile_allocation_axis(
    SparseTileAllocationPlan plan,
    uint32_t tile_count) noexcept {
    if (tile_count == 0u) {
        return {SparseTileAllocationStatus::ZERO_TILE_COUNT, 0u};
    }
    auto count = static_cast<size_t>(tile_count);
    if (plan.byte_size >
        std::numeric_limits<size_t>::max() / count) {
        return {
            SparseTileAllocationStatus::ARITHMETIC_OVERFLOW, 0u};
    }
    plan.byte_size *= count;
    return plan;
}

// Computes the physical heap bytes occupied by a sparse tile region. Sparse
// edge tiles still consume one complete backend-reported tile allocation, so
// this deliberately uses tile_count rather than the clipped texel extent.
[[nodiscard]] constexpr SparseTileAllocationPlan
plan_sparse_tile_allocation(
    uint32_t dimension,
    uint3 tile_count,
    size_t tile_size_bytes) noexcept {
    if (dimension == 0u || dimension > 3u) {
        return {SparseTileAllocationStatus::INVALID_DIMENSION, 0u};
    }
    if (tile_size_bytes == 0u) {
        return {SparseTileAllocationStatus::ZERO_TILE_SIZE, 0u};
    }
    auto plan = SparseTileAllocationPlan{
        SparseTileAllocationStatus::SUCCESS, tile_size_bytes};
    plan = append_sparse_tile_allocation_axis(plan, tile_count.x);
    if (!plan || dimension == 1u) { return plan; }
    plan = append_sparse_tile_allocation_axis(plan, tile_count.y);
    if (!plan || dimension == 2u) { return plan; }
    return append_sparse_tile_allocation_axis(plan, tile_count.z);
}

}// namespace luisa::compute::detail
