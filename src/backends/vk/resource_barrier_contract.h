#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace lc::vk::detail {

using NativeBufferIdentity = uint64_t;
using NativeImageIdentity = uint64_t;

enum class QueueType : uint8_t {
    GRAPHICS,
    COMPUTE,
    COPY
};

struct QueueScope {
    VkPipelineStageFlags2 stages;
    VkAccessFlags2 access;
};

struct TextureAccessLayout {
    VkAccessFlags2 access;
    VkImageLayout layout;
};

// A VkDescriptorImageInfo declares one layout for every subresource in its
// image view. Use UNDEFINED only as the fold identity; differing tracked mip
// layouts must converge to GENERAL before the descriptor is written.
[[nodiscard]] constexpr VkImageLayout
combine_texture_descriptor_view_layout(
    VkImageLayout first,
    VkImageLayout second) noexcept {
    if (first == VK_IMAGE_LAYOUT_UNDEFINED) { return second; }
    if (second == VK_IMAGE_LAYOUT_UNDEFINED) { return first; }
    return first == second ? first : VK_IMAGE_LAYOUT_GENERAL;
}

[[nodiscard]] constexpr TextureAccessLayout
combine_texture_access_layout(
    TextureAccessLayout first,
    TextureAccessLayout second) noexcept {
    if (first.access == 0u) { return second; }
    if (second.access == 0u) { return first; }
    return TextureAccessLayout{
        first.access | second.access,
        first.layout == second.layout ?
            first.layout :
            VK_IMAGE_LAYOUT_GENERAL};
}

enum class TextureLayoutContract : uint8_t {
    GENERIC_USAGE,
    EXPLICIT_NATIVE
};

// Backend-owned commands query the layout selected by the tracker, so images
// created for simultaneous access may stay in GENERAL to avoid unnecessary
// transitions. Native/custom commands instead publish the exact layout they
// will pass to Vulkan; changing that value would make the barrier disagree
// with the command's VkImageLayout argument.
[[nodiscard]] constexpr VkImageLayout resolve_texture_barrier_layout(
    bool simultaneous_access,
    VkAccessFlags2 access,
    VkImageLayout requested_layout,
    TextureLayoutContract contract) noexcept {
    if (contract == TextureLayoutContract::EXPLICIT_NATIVE ||
        !simultaneous_access ||
        (access & (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0u) {
        return requested_layout;
    }
    return VK_IMAGE_LAYOUT_GENERAL;
}

// Vulkan non-dispatchable handles are either opaque pointers or uint64_t
// values. Preserve every handle bit in an integer key so distinct Luisa
// wrappers of the same VkBuffer share one hazard state without retaining a
// representative wrapper pointer.
[[nodiscard]] inline NativeBufferIdentity
native_buffer_identity(VkBuffer buffer) noexcept {
    static_assert(sizeof(VkBuffer) == sizeof(NativeBufferIdentity));
    return std::bit_cast<NativeBufferIdentity>(buffer);
}

[[nodiscard]] inline NativeImageIdentity
native_image_identity(VkImage image) noexcept {
    static_assert(sizeof(VkImage) == sizeof(NativeImageIdentity));
    return std::bit_cast<NativeImageIdentity>(image);
}

// vkCmdDispatchIndirect is valid on compute-capable queues. Its synchronization
// scope is DRAW_INDIRECT even though no graphics draw is involved, so that bit
// must survive compute-queue stage filtering.
inline constexpr VkPipelineStageFlags2 compute_queue_stage_mask =
    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_COPY_BIT |
    VK_PIPELINE_STAGE_2_TRANSFER_BIT |
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR |
    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |
    VK_PIPELINE_STAGE_2_HOST_BIT;

inline constexpr VkPipelineStageFlags2 copy_queue_stage_mask =
    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
    VK_PIPELINE_STAGE_2_COPY_BIT |
    VK_PIPELINE_STAGE_2_TRANSFER_BIT |
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR |
    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT |
    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |
    VK_PIPELINE_STAGE_2_HOST_BIT;

inline constexpr VkAccessFlags2 compute_queue_access_mask =
    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
    VK_ACCESS_2_UNIFORM_READ_BIT |
    VK_ACCESS_2_SHADER_READ_BIT |
    VK_ACCESS_2_SHADER_WRITE_BIT |
    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
    VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
    VK_ACCESS_2_TRANSFER_READ_BIT |
    VK_ACCESS_2_TRANSFER_WRITE_BIT |
    VK_ACCESS_2_HOST_READ_BIT |
    VK_ACCESS_2_HOST_WRITE_BIT |
    VK_ACCESS_2_MEMORY_READ_BIT |
    VK_ACCESS_2_MEMORY_WRITE_BIT |
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

inline constexpr VkAccessFlags2 copy_queue_access_mask =
    VK_ACCESS_2_TRANSFER_READ_BIT |
    VK_ACCESS_2_TRANSFER_WRITE_BIT |
    VK_ACCESS_2_HOST_READ_BIT |
    VK_ACCESS_2_HOST_WRITE_BIT |
    VK_ACCESS_2_MEMORY_READ_BIT |
    VK_ACCESS_2_MEMORY_WRITE_BIT |
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

[[nodiscard]] constexpr VkPipelineStageFlags2
filter_compute_queue_stages(VkPipelineStageFlags2 stages) noexcept {
    return stages & compute_queue_stage_mask;
}

// A before/after contract may originate on a queue with a wider capability
// set than the queue recording the current command buffer. If any part of the
// declared scope cannot be represented on this queue, retain a conservative
// dependency instead of independently masking stages and accesses into an
// invalid pair. Image layouts are deliberately outside this helper: queue
// normalization must never rewrite oldLayout/newLayout.
[[nodiscard]] constexpr QueueScope normalize_queue_scope(
    QueueType type, VkPipelineStageFlags2 stages,
    VkAccessFlags2 access) noexcept {
    if (type == QueueType::GRAPHICS) {
        if (access != 0u && stages == 0u) {
            return QueueScope{
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT |
                    VK_ACCESS_2_MEMORY_WRITE_BIT};
        }
        return QueueScope{stages, access};
    }
    auto allowed_stages = type == QueueType::COMPUTE ?
                              compute_queue_stage_mask :
                              copy_queue_stage_mask;
    auto allowed_access = type == QueueType::COMPUTE ?
                              compute_queue_access_mask :
                              copy_queue_access_mask;
    auto filtered_stages = stages & allowed_stages;
    auto has_unsupported_stages = (stages & ~allowed_stages) != 0u;
    auto has_unsupported_access = (access & ~allowed_access) != 0u;
    auto access_without_stage = access != 0u && filtered_stages == 0u;
    if (has_unsupported_stages || has_unsupported_access ||
        access_without_stage) {
        return QueueScope{
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            access == 0u ? 0u :
                           VK_ACCESS_2_MEMORY_READ_BIT |
                               VK_ACCESS_2_MEMORY_WRITE_BIT};
    }
    return QueueScope{filtered_stages, access};
}

enum class DirectStorageBufferDescriptorStatus : uint8_t {
    SUCCESS,
    INVALID_DESCRIPTOR_ALIGNMENT,
    EMPTY_RANGE,
    VIEW_OUT_OF_BOUNDS,
    MISALIGNED_DESCRIPTOR_OFFSET,
    INCOMPLETE_TYPED_ELEMENT,
    RANGE_LIMIT_EXCEEDED
};

[[nodiscard]] constexpr const char *
direct_storage_buffer_descriptor_status_name(
    DirectStorageBufferDescriptorStatus status) noexcept {
    switch (status) {
        case DirectStorageBufferDescriptorStatus::SUCCESS: return "success";
        case DirectStorageBufferDescriptorStatus::INVALID_DESCRIPTOR_ALIGNMENT: return "invalid device descriptor alignment";
        case DirectStorageBufferDescriptorStatus::EMPTY_RANGE: return "empty storage-buffer range";
        case DirectStorageBufferDescriptorStatus::VIEW_OUT_OF_BOUNDS: return "buffer view is out of bounds";
        case DirectStorageBufferDescriptorStatus::MISALIGNED_DESCRIPTOR_OFFSET: return "buffer-view offset is not descriptor-aligned";
        case DirectStorageBufferDescriptorStatus::INCOMPLETE_TYPED_ELEMENT: return "typed buffer view is not made of whole elements";
        case DirectStorageBufferDescriptorStatus::RANGE_LIMIT_EXCEEDED: return "storage-buffer range exceeds the device limit";
    }
    return "unknown";
}

// The legacy HLSL path has no descriptor-bias companion in its shader ABI, so
// it cannot move a subview's descriptor base down and repair the index in the
// shader. Validate the exact VkDescriptorBufferInfo it is about to expose and
// fail closed instead of recording a Vulkan-invalid descriptor. The native
// XIR/SPIR-V path uses StorageBufferMetadata and is not subject to this
// restriction.
[[nodiscard]] constexpr DirectStorageBufferDescriptorStatus
validate_direct_storage_buffer_descriptor(
    size_t view_offset, size_t view_size, size_t backing_size,
    size_t logical_element_stride, uint64_t descriptor_alignment,
    uint64_t max_descriptor_range) noexcept {
    if (descriptor_alignment == 0u ||
        (descriptor_alignment & (descriptor_alignment - 1u)) != 0u) {
        return DirectStorageBufferDescriptorStatus::INVALID_DESCRIPTOR_ALIGNMENT;
    }
    if (view_size == 0u) {
        return DirectStorageBufferDescriptorStatus::EMPTY_RANGE;
    }
    if (view_offset > backing_size ||
        view_size > backing_size - view_offset) {
        return DirectStorageBufferDescriptorStatus::VIEW_OUT_OF_BOUNDS;
    }
    if (view_offset % descriptor_alignment != 0u) {
        return DirectStorageBufferDescriptorStatus::MISALIGNED_DESCRIPTOR_OFFSET;
    }
    if (logical_element_stride != 0u &&
        (view_offset % logical_element_stride != 0u ||
         view_size % logical_element_stride != 0u)) {
        return DirectStorageBufferDescriptorStatus::INCOMPLETE_TYPED_ELEMENT;
    }
    if (view_size > max_descriptor_range) {
        return DirectStorageBufferDescriptorStatus::RANGE_LIMIT_EXCEEDED;
    }
    return DirectStorageBufferDescriptorStatus::SUCCESS;
}

}// namespace lc::vk::detail
