#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <vulkan/vulkan_core.h>

#include "timeline_semaphore_plan.h"

namespace lc::vk::detail {

enum class SparseSubmissionTimelineStatus : uint8_t {
    SUCCESS,
    SIGNAL_VALUE_OVERFLOW,
    GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE,
    GPU_COUNTER_AHEAD_OF_SIGNAL,
    MAX_VALUE_DIFFERENCE_EXCEEDED
};

[[nodiscard]] constexpr const char *
sparse_submission_timeline_status_name(
    SparseSubmissionTimelineStatus status) noexcept {
    switch (status) {
        case SparseSubmissionTimelineStatus::SUCCESS: return "success";
        case SparseSubmissionTimelineStatus::SIGNAL_VALUE_OVERFLOW: return "signal value overflow";
        case SparseSubmissionTimelineStatus::GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE: return "tracked GPU signal is ahead of the logical fence";
        case SparseSubmissionTimelineStatus::GPU_COUNTER_AHEAD_OF_SIGNAL: return "GPU counter is ahead of the tracked signal";
        case SparseSubmissionTimelineStatus::MAX_VALUE_DIFFERENCE_EXCEEDED: return "maxTimelineSemaphoreValueDifference exceeded";
    }
    return "unknown";
}

struct SparseSubmissionTimelinePlan {
    SparseSubmissionTimelineStatus status;
    uint64_t bridge_signal_value;
    uint64_t sparse_signal_value;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseSubmissionTimelineStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr SparseSubmissionTimelinePlan
plan_sparse_submission_timeline(
    uint64_t previous_logical_fence,
    uint64_t previous_gpu_signal,
    uint64_t current_gpu_value,
    uint64_t max_value_difference) noexcept {
    auto sparse_signal_increment = plan_timeline_value_increment(
        previous_logical_fence, 2u);
    if (!sparse_signal_increment) {
        return {
            .status =
                SparseSubmissionTimelineStatus::SIGNAL_VALUE_OVERFLOW};
    }
    if (previous_gpu_signal > previous_logical_fence) {
        return {
            .status = SparseSubmissionTimelineStatus::
                GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE};
    }
    if (current_gpu_value > previous_gpu_signal) {
        return {
            .status = SparseSubmissionTimelineStatus::
                GPU_COUNTER_AHEAD_OF_SIGNAL};
    }
    auto bridge_signal_value = previous_logical_fence + 1u;
    auto sparse_signal_value = sparse_signal_increment.value;
    // The sparse signal is the far endpoint of the two new monotonically
    // ordered values. If it is valid against the current/pending interval,
    // the intermediate bridge value is valid as well.
    auto signal_plan = plan_timeline_semaphore_signal(
        current_gpu_value, previous_gpu_signal,
        sparse_signal_value, max_value_difference);
    if (!signal_plan) {
        switch (signal_plan.status) {
            case TimelineSemaphoreValueStatus::
                TRACKED_SIGNAL_BEHIND_CURRENT:
                return {
                    .status = SparseSubmissionTimelineStatus::
                        GPU_COUNTER_AHEAD_OF_SIGNAL};
            case TimelineSemaphoreValueStatus::
                ZERO_MAX_VALUE_DIFFERENCE:
            case TimelineSemaphoreValueStatus::
                TRACKED_SIGNAL_RANGE_EXCEEDED:
            case TimelineSemaphoreValueStatus::
                MAX_VALUE_DIFFERENCE_EXCEEDED:
                return {
                    .status = SparseSubmissionTimelineStatus::
                        MAX_VALUE_DIFFERENCE_EXCEEDED};
            case TimelineSemaphoreValueStatus::
                SIGNAL_VALUE_NOT_INCREASING:
            case TimelineSemaphoreValueStatus::
                WAIT_VALUE_AHEAD_OF_TRACKED_SIGNAL:
            case TimelineSemaphoreValueStatus::SUCCESS:
                break;
        }
        return {
            .status = SparseSubmissionTimelineStatus::
                GPU_SIGNAL_AHEAD_OF_LOGICAL_FENCE};
    }
    return {
        .status = SparseSubmissionTimelineStatus::SUCCESS,
        .bridge_signal_value = bridge_signal_value,
        .sparse_signal_value = sparse_signal_value};
}

enum class SparseImageRequirementsStatus : uint8_t {
    SUCCESS,
    EMPTY,
    METADATA_BINDING_UNSUPPORTED,
    MULTIPLE_COLOR_REQUIREMENTS,
    MISSING_COLOR_REQUIREMENTS
};

struct SparseImageRequirementsSelection {
    SparseImageRequirementsStatus status;
    size_t color_requirement_index;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseImageRequirementsStatus::SUCCESS;
    }
};

enum class SparseImageMipTailStatus : uint8_t {
    SUCCESS,
    ZERO_MIP_LEVELS,
    REQUESTED_MIP_TAIL
};

struct SparseImageMipTailResult {
    SparseImageMipTailStatus status;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseImageMipTailStatus::SUCCESS;
    }
};

// The public sparse-texture API exposes only non-tail image tile binds. An
// image is usable only when every created mip precedes imageMipTailFirstLod.
[[nodiscard]] constexpr SparseImageMipTailResult
validate_sparse_image_mip_tail(
    uint32_t mip_levels, uint32_t mip_tail_first_lod) noexcept {
    if (mip_levels == 0u) {
        return {.status = SparseImageMipTailStatus::ZERO_MIP_LEVELS};
    }
    if (mip_tail_first_lod < mip_levels) {
        return {.status =
                    SparseImageMipTailStatus::REQUESTED_MIP_TAIL};
    }
    return {.status = SparseImageMipTailStatus::SUCCESS};
}

[[nodiscard]] constexpr SparseImageRequirementsSelection
select_sparse_image_requirements(
    std::span<const VkSparseImageMemoryRequirements> requirements) noexcept {
    if (requirements.empty()) {
        return {.status = SparseImageRequirementsStatus::EMPTY};
    }
    auto color_index = requirements.size();
    for (auto i = 0u; i < requirements.size(); i++) {
        auto aspect = requirements[i].formatProperties.aspectMask;
        if ((aspect & VK_IMAGE_ASPECT_METADATA_BIT) != 0u) {
            return {.status =
                        SparseImageRequirementsStatus::METADATA_BINDING_UNSUPPORTED};
        }
        if ((aspect & VK_IMAGE_ASPECT_COLOR_BIT) != 0u) {
            if (color_index != requirements.size()) {
                return {.status =
                            SparseImageRequirementsStatus::MULTIPLE_COLOR_REQUIREMENTS};
            }
            color_index = i;
        }
    }
    if (color_index == requirements.size()) {
        return {.status =
                    SparseImageRequirementsStatus::MISSING_COLOR_REQUIREMENTS};
    }
    return {
        .status = SparseImageRequirementsStatus::SUCCESS,
        .color_requirement_index = color_index};
}

namespace sparse_binding_plan_detail {

[[nodiscard]] constexpr bool checked_add(
    uint64_t lhs, uint64_t rhs, uint64_t &result) noexcept {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

[[nodiscard]] constexpr bool checked_multiply(
    uint64_t lhs, uint64_t rhs, uint64_t &result) noexcept {
    if (lhs != 0u &&
        rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

[[nodiscard]] constexpr uint64_t ceil_divide_nonzero(
    uint64_t value, uint64_t divisor) noexcept {
    return value / divisor + static_cast<uint64_t>(value % divisor != 0u);
}

}// namespace sparse_binding_plan_detail

enum class SparseBufferBindingStatus : uint8_t {
    SUCCESS,
    ZERO_PHYSICAL_RESOURCE_SIZE,
    ZERO_ALIGNMENT,
    MISALIGNED_PHYSICAL_RESOURCE_SIZE,
    ZERO_TILE_COUNT,
    OUT_OF_GRID,
    ARITHMETIC_OVERFLOW
};

struct SparseBufferBindingRequest {
    // This is VkMemoryRequirements::size: the sparse virtual-address range,
    // not the API-visible VkBufferCreateInfo::size.
    VkDeviceSize physical_resource_size;
    VkDeviceSize alignment;
    uint64_t start_tile;
    uint64_t tile_count;
};

struct SparseBufferBindingPlan {
    SparseBufferBindingStatus status;
    uint64_t grid_tile_count;
    VkDeviceSize resource_offset;
    VkDeviceSize binding_size;
    VkDeviceSize required_heap_size;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseBufferBindingStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr bool sparse_buffer_bindings_overlap(
    SparseBufferBindingPlan lhs,
    SparseBufferBindingPlan rhs) noexcept {
    if (!lhs || !rhs) { return false; }
    if (lhs.resource_offset <= rhs.resource_offset) {
        return rhs.resource_offset - lhs.resource_offset <
               lhs.binding_size;
    }
    return lhs.resource_offset - rhs.resource_offset <
           rhs.binding_size;
}

[[nodiscard]] constexpr SparseBufferBindingPlan
plan_sparse_buffer_binding(SparseBufferBindingRequest request) noexcept {
    if (request.physical_resource_size == 0u) {
        return {.status =
                    SparseBufferBindingStatus::ZERO_PHYSICAL_RESOURCE_SIZE};
    }
    if (request.alignment == 0u) {
        return {.status = SparseBufferBindingStatus::ZERO_ALIGNMENT};
    }
    // VkSparseMemoryBind resourceOffset and size must both be integer
    // multiples of VkMemoryRequirements::alignment (VUID 09491). Unlike sparse
    // images, a buffer's final native binding may therefore not be clipped.
    if (request.physical_resource_size % request.alignment != 0u) {
        return {.status =
                    SparseBufferBindingStatus::MISALIGNED_PHYSICAL_RESOURCE_SIZE};
    }
    if (request.tile_count == 0u) {
        return {.status = SparseBufferBindingStatus::ZERO_TILE_COUNT};
    }
    auto grid_tile_count =
        request.physical_resource_size / request.alignment;
    if (request.start_tile >= grid_tile_count ||
        request.tile_count > grid_tile_count - request.start_tile) {
        return {
            .status = SparseBufferBindingStatus::OUT_OF_GRID,
            .grid_tile_count = grid_tile_count};
    }
    uint64_t resource_offset{};
    uint64_t binding_size{};
    uint64_t binding_end{};
    if (!sparse_binding_plan_detail::checked_multiply(
            request.start_tile, request.alignment,
            resource_offset) ||
        !sparse_binding_plan_detail::checked_multiply(
            request.tile_count, request.alignment,
            binding_size) ||
        !sparse_binding_plan_detail::checked_add(
            resource_offset, binding_size,
            binding_end)) {
        return {
            .status = SparseBufferBindingStatus::ARITHMETIC_OVERFLOW,
            .grid_tile_count = grid_tile_count};
    }
    if (binding_end > request.physical_resource_size) {
        return {
            .status = SparseBufferBindingStatus::OUT_OF_GRID,
            .grid_tile_count = grid_tile_count};
    }
    return {
        .status = SparseBufferBindingStatus::SUCCESS,
        .grid_tile_count = grid_tile_count,
        .resource_offset = resource_offset,
        .binding_size = binding_size,
        .required_heap_size = binding_size};
}

enum class SparseImageBindingStatus : uint8_t {
    SUCCESS,
    ZERO_MIP_EXTENT,
    ZERO_GRANULARITY,
    ZERO_TILE_BYTES,
    ZERO_TILE_COUNT,
    MIP_TAIL,
    OUT_OF_GRID,
    OFFSET_OUT_OF_RANGE,
    ARITHMETIC_OVERFLOW
};

struct SparseImageBindingRequest {
    VkExtent3D mip_extent;
    VkExtent3D granularity;
    VkDeviceSize tile_byte_size;
    uint32_t mip_level;
    // Set to UINT32_MAX when the image has no mip tail.
    uint32_t mip_tail_first_lod;
    std::array<uint32_t, 3u> start_tile;
    std::array<uint32_t, 3u> tile_count;
};

struct SparseImageBindingPlan {
    SparseImageBindingStatus status;
    std::array<uint32_t, 3u> grid_extent;
    VkOffset3D offset;
    VkExtent3D extent;
    uint64_t tile_count;
    VkDeviceSize required_heap_size;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseImageBindingStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr bool sparse_image_bindings_overlap(
    SparseImageBindingPlan lhs, uint32_t lhs_mip_level,
    SparseImageBindingPlan rhs, uint32_t rhs_mip_level) noexcept {
    if (!lhs || !rhs || lhs_mip_level != rhs_mip_level) {
        return false;
    }
    auto overlaps_axis = [](int32_t lhs_offset, uint32_t lhs_extent,
                            int32_t rhs_offset,
                            uint32_t rhs_extent) noexcept {
        auto lhs_begin = static_cast<uint64_t>(lhs_offset);
        auto rhs_begin = static_cast<uint64_t>(rhs_offset);
        if (lhs_begin <= rhs_begin) {
            return rhs_begin - lhs_begin < lhs_extent;
        }
        return lhs_begin - rhs_begin < rhs_extent;
    };
    return overlaps_axis(
               lhs.offset.x, lhs.extent.width,
               rhs.offset.x, rhs.extent.width) &&
           overlaps_axis(
               lhs.offset.y, lhs.extent.height,
               rhs.offset.y, rhs.extent.height) &&
           overlaps_axis(
               lhs.offset.z, lhs.extent.depth,
               rhs.offset.z, rhs.extent.depth);
}

[[nodiscard]] constexpr SparseImageBindingPlan
plan_sparse_image_binding(SparseImageBindingRequest request) noexcept {
    auto mip_extent = std::array{
        request.mip_extent.width,
        request.mip_extent.height,
        request.mip_extent.depth};
    auto granularity = std::array{
        request.granularity.width,
        request.granularity.height,
        request.granularity.depth};
    for (auto extent : mip_extent) {
        if (extent == 0u) {
            return {.status =
                        SparseImageBindingStatus::ZERO_MIP_EXTENT};
        }
    }
    for (auto block : granularity) {
        if (block == 0u) {
            return {.status =
                        SparseImageBindingStatus::ZERO_GRANULARITY};
        }
    }
    if (request.tile_byte_size == 0u) {
        return {.status = SparseImageBindingStatus::ZERO_TILE_BYTES};
    }
    for (auto count : request.tile_count) {
        if (count == 0u) {
            return {.status =
                        SparseImageBindingStatus::ZERO_TILE_COUNT};
        }
    }
    if (request.mip_level >= request.mip_tail_first_lod) {
        return {.status = SparseImageBindingStatus::MIP_TAIL};
    }

    SparseImageBindingPlan plan{
        .status = SparseImageBindingStatus::SUCCESS};
    std::array<uint64_t, 3u> offsets{};
    std::array<uint64_t, 3u> clipped_extents{};
    for (auto axis = 0u; axis < 3u; ++axis) {
        auto grid = sparse_binding_plan_detail::ceil_divide_nonzero(
            mip_extent[axis], granularity[axis]);
        plan.grid_extent[axis] = static_cast<uint32_t>(grid);
        auto start = static_cast<uint64_t>(request.start_tile[axis]);
        auto count = static_cast<uint64_t>(request.tile_count[axis]);
        if (start >= grid || count > grid - start) {
            plan.status = SparseImageBindingStatus::OUT_OF_GRID;
            return plan;
        }
        uint64_t nominal_extent{};
        if (!sparse_binding_plan_detail::checked_multiply(
                start, granularity[axis], offsets[axis]) ||
            !sparse_binding_plan_detail::checked_multiply(
                count, granularity[axis], nominal_extent)) {
            plan.status =
                SparseImageBindingStatus::ARITHMETIC_OVERFLOW;
            return plan;
        }
        if (offsets[axis] >
            static_cast<uint64_t>(
                std::numeric_limits<int32_t>::max())) {
            plan.status =
                SparseImageBindingStatus::OFFSET_OUT_OF_RANGE;
            return plan;
        }
        auto remaining_extent =
            static_cast<uint64_t>(mip_extent[axis]) - offsets[axis];
        clipped_extents[axis] =
            std::min(nominal_extent, remaining_extent);
    }

    uint64_t tile_count_xy{};
    if (!sparse_binding_plan_detail::checked_multiply(
            request.tile_count[0u], request.tile_count[1u],
            tile_count_xy) ||
        !sparse_binding_plan_detail::checked_multiply(
            tile_count_xy, request.tile_count[2u],
            plan.tile_count) ||
        !sparse_binding_plan_detail::checked_multiply(
            plan.tile_count, request.tile_byte_size,
            plan.required_heap_size)) {
        plan.status = SparseImageBindingStatus::ARITHMETIC_OVERFLOW;
        return plan;
    }
    plan.offset = {
        static_cast<int32_t>(offsets[0u]),
        static_cast<int32_t>(offsets[1u]),
        static_cast<int32_t>(offsets[2u])};
    plan.extent = {
        static_cast<uint32_t>(clipped_extents[0u]),
        static_cast<uint32_t>(clipped_extents[1u]),
        static_cast<uint32_t>(clipped_extents[2u])};
    return plan;
}

enum class SparseHeapCompatibilityStatus : uint8_t {
    SUCCESS,
    ZERO_BINDING_SIZE,
    ZERO_HEAP_SIZE,
    ZERO_ALIGNMENT,
    HEAP_TOO_SMALL,
    MISALIGNED_ALLOCATION_OFFSET,
    ALLOCATION_RANGE_OVERFLOW,
    MEMORY_TYPE_INDEX_OUT_OF_RANGE,
    INCOMPATIBLE_MEMORY_TYPE
};

struct SparseHeapCompatibilityRequest {
    VkDeviceSize required_binding_size;
    VkDeviceSize logical_heap_size;
    VkDeviceSize allocation_offset;
    VkDeviceSize required_alignment;
    uint32_t memory_type_index;
    uint32_t memory_type_bits;
};

struct SparseHeapCompatibilityResult {
    SparseHeapCompatibilityStatus status;
    VkDeviceSize allocation_end;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseHeapCompatibilityStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr SparseHeapCompatibilityResult
validate_sparse_heap_compatibility(
    SparseHeapCompatibilityRequest request) noexcept {
    if (request.required_binding_size == 0u) {
        return {.status =
                    SparseHeapCompatibilityStatus::ZERO_BINDING_SIZE};
    }
    if (request.logical_heap_size == 0u) {
        return {.status = SparseHeapCompatibilityStatus::ZERO_HEAP_SIZE};
    }
    if (request.required_alignment == 0u) {
        return {.status = SparseHeapCompatibilityStatus::ZERO_ALIGNMENT};
    }
    if (request.required_binding_size > request.logical_heap_size) {
        return {.status = SparseHeapCompatibilityStatus::HEAP_TOO_SMALL};
    }
    if (request.allocation_offset % request.required_alignment != 0u) {
        return {.status =
                    SparseHeapCompatibilityStatus::MISALIGNED_ALLOCATION_OFFSET};
    }
    uint64_t allocation_end{};
    if (!sparse_binding_plan_detail::checked_add(
            request.allocation_offset, request.logical_heap_size,
            allocation_end)) {
        return {.status =
                    SparseHeapCompatibilityStatus::ALLOCATION_RANGE_OVERFLOW};
    }
    if (request.memory_type_index >= 32u) {
        return {.status =
                    SparseHeapCompatibilityStatus::MEMORY_TYPE_INDEX_OUT_OF_RANGE};
    }
    auto selected_memory_type = uint32_t{1u}
                                << request.memory_type_index;
    if ((request.memory_type_bits & selected_memory_type) == 0u) {
        return {.status =
                    SparseHeapCompatibilityStatus::INCOMPATIBLE_MEMORY_TYPE};
    }
    return {
        .status = SparseHeapCompatibilityStatus::SUCCESS,
        .allocation_end = allocation_end};
}

}// namespace lc::vk::detail
