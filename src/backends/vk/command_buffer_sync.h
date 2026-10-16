#pragma once

// Capability-adaptive synchronization/copy helpers for the Vulkan backend.
//
// The backend prefers the Vulkan 1.3 core entry points
// (vkCmdPipelineBarrier2/vkCmdCopyBuffer2/...) but must also run on Vulkan
// 1.2 devices (the realistic Android floor). These helpers dispatch on
// Device::sync2_capable()/copy2_capable() and translate the 1.3 command
// structures into the classic Vulkan 1.0 forms when the device does not
// support the 2 entry points.

#include <volk.h>
#include <luisa/core/stl/vector.h>

#include "device.h"

namespace lc::vk::detail {

[[nodiscard]] constexpr VkAccessFlags legacy_access_mask(
    VkAccessFlags2 access) noexcept {
    VkAccessFlags result = 0u;
    if (access & VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT) { result |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT; }
    if (access & VK_ACCESS_2_INDEX_READ_BIT) { result |= VK_ACCESS_INDEX_READ_BIT; }
    if (access & VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT) { result |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT; }
    if (access & VK_ACCESS_2_UNIFORM_READ_BIT) { result |= VK_ACCESS_UNIFORM_READ_BIT; }
    if (access & VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT) { result |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT; }
    if (access & VK_ACCESS_2_SHADER_READ_BIT) { result |= VK_ACCESS_SHADER_READ_BIT; }
    if (access & VK_ACCESS_2_SHADER_WRITE_BIT) { result |= VK_ACCESS_SHADER_WRITE_BIT; }
    if (access & VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT) { result |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT; }
    if (access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) { result |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; }
    if (access & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT) { result |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT; }
    if (access & VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) { result |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; }
    if (access & VK_ACCESS_2_TRANSFER_READ_BIT) { result |= VK_ACCESS_TRANSFER_READ_BIT; }
    if (access & VK_ACCESS_2_TRANSFER_WRITE_BIT) { result |= VK_ACCESS_TRANSFER_WRITE_BIT; }
    if (access & VK_ACCESS_2_HOST_READ_BIT) { result |= VK_ACCESS_HOST_READ_BIT; }
    if (access & VK_ACCESS_2_HOST_WRITE_BIT) { result |= VK_ACCESS_HOST_WRITE_BIT; }
    if (access & VK_ACCESS_2_MEMORY_READ_BIT) { result |= VK_ACCESS_MEMORY_READ_BIT; }
    if (access & VK_ACCESS_2_MEMORY_WRITE_BIT) { result |= VK_ACCESS_MEMORY_WRITE_BIT; }
    if (access & VK_ACCESS_2_SHADER_SAMPLED_READ_BIT) { result |= VK_ACCESS_SHADER_READ_BIT; }
#ifdef VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
    if (access & VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR) { result |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR; }
    if (access & VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR) { result |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR; }
#endif
#ifdef VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR
    if (access & VK_ACCESS_2_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR) { result |= VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR; }
#endif
#ifdef VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT
    if (access & VK_ACCESS_2_CONDITIONAL_RENDERING_READ_BIT_EXT) { result |= VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT; }
#endif
#ifdef VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV
    if (access & VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_NV) { result |= VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV; }
    if (access & VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_NV) { result |= VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV; }
#endif
#ifdef VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT
    if (access & VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT) { result |= VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT; }
    if (access & VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT) { result |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT; }
    if (access & VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT) { result |= VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT; }
#endif
#ifdef VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT
    if (access & VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT) { result |= VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT; }
#endif
#ifdef VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    // VK_ACCESS_2_SHADER_STORAGE_READ/WRITE were merged into SHADER_READ/WRITE
    // in Vulkan 1.3; the legacy bits are already covered above.
    if (access & VK_ACCESS_2_SHADER_STORAGE_READ_BIT) { result |= VK_ACCESS_SHADER_READ_BIT; }
    if (access & VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) { result |= VK_ACCESS_SHADER_WRITE_BIT; }
#endif
    return result;
}

[[nodiscard]] constexpr VkPipelineStageFlags legacy_stage_mask(
    VkPipelineStageFlags2 stages) noexcept {
    VkPipelineStageFlags result = 0u;
    if (stages & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) { result |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT) { result |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT) { result |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT) { result |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT) { result |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT) { result |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT) { result |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT) { result |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT) { result |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT) { result |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT) { result |= VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT) { result |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) { result |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_TRANSFER_BIT) { result |= VK_PIPELINE_STAGE_TRANSFER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_COPY_BIT) { result |= VK_PIPELINE_STAGE_TRANSFER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_CLEAR_BIT) { result |= VK_PIPELINE_STAGE_TRANSFER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_RESOLVE_BIT) { result |= VK_PIPELINE_STAGE_TRANSFER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) { result |= VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT) { result |= VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT) { result |= VK_PIPELINE_STAGE_TRANSFER_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT) { result |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT) { result |= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; }
    if (stages & VK_PIPELINE_STAGE_2_HOST_BIT) { result |= VK_PIPELINE_STAGE_HOST_BIT; }
#ifdef VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
    if (stages & VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR) { result |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR; }
    if (stages & VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR) { result |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR; }
    if (stages & VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR) { result |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR; }
#endif
#ifdef VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_NV
    if (stages & VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_NV) { result |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV; }
    if (stages & VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_NV) { result |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV; }
#endif
#ifdef VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
    if (stages & VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT) { result |= VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT; }
    if (stages & VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT) { result |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT; }
#endif
#ifdef VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR
    if (stages & VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) { result |= VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR; }
#endif
#ifdef VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT
    if (stages & VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT) { result |= VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT; }
#endif
#ifdef VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV
    if (stages & VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_NV) { result |= VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV; }
#endif
#ifdef VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT
    if (stages & VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT) { result |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT; }
#endif
    return result;
}

// vkCmdCopyBuffer with a capability-adaptive dispatch.
inline void cmd_copy_buffer(
    VkCommandBuffer cmd, Device const *device,
    VkCopyBufferInfo2 const *info) noexcept {
    if (device->copy2_capable()) {
#ifdef VK_KHR_copy_commands2
        // Vulkan 1.2 devices expose the extension aliases (vkCmd*2KHR)
        // rather than the core-1.3 names; prefer whichever the loader resolved.
        if (vkCmdCopyBuffer2 != nullptr) {
            vkCmdCopyBuffer2(cmd, info);
        } else {
            vkCmdCopyBuffer2KHR(cmd, info);
        }
#else
        vkCmdCopyBuffer2(cmd, info);
#endif
        return;
    }
    // Inline capacity is sized from the element type so the stack footprint
    // stays comparable across helpers: VkBufferCopy is 24 B, so 8 elements
    // cost 192 B and cover the common multi-region case without a heap
    // allocation; larger counts overflow to the heap.
    luisa::fixed_vector<VkBufferCopy, 8> regions;
    regions.reserve(info->regionCount);
    for (uint32_t i = 0u; i < info->regionCount; ++i) {
        auto const &r = info->pRegions[i];
        regions.emplace_back(VkBufferCopy{r.srcOffset, r.dstOffset, r.size});
    }
    vkCmdCopyBuffer(
        cmd, info->srcBuffer, info->dstBuffer,
        info->regionCount, regions.data());
}

inline void cmd_copy_buffer_to_image(
    VkCommandBuffer cmd, Device const *device,
    VkCopyBufferToImageInfo2 const *info) noexcept {
    if (device->copy2_capable()) {
#ifdef VK_KHR_copy_commands2
        if (vkCmdCopyBufferToImage2 != nullptr) {
            vkCmdCopyBufferToImage2(cmd, info);
        } else {
            vkCmdCopyBufferToImage2KHR(cmd, info);
        }
#else
        vkCmdCopyBufferToImage2(cmd, info);
#endif
        return;
    }
    // VkBufferImageCopy is 56 B: 4 elements cost 224 B of stack (the usual
    // per-mip upload set); larger counts overflow to the heap.
    luisa::fixed_vector<VkBufferImageCopy, 4> regions;
    regions.reserve(info->regionCount);
    for (uint32_t i = 0u; i < info->regionCount; ++i) {
        auto const &r = info->pRegions[i];
        regions.emplace_back(VkBufferImageCopy{
            r.bufferOffset, r.bufferRowLength, r.bufferImageHeight,
            r.imageSubresource, r.imageOffset, r.imageExtent});
    }
    vkCmdCopyBufferToImage(
        cmd, info->srcBuffer, info->dstImage, info->dstImageLayout,
        info->regionCount, regions.data());
}

inline void cmd_copy_image_to_buffer(
    VkCommandBuffer cmd, Device const *device,
    VkCopyImageToBufferInfo2 const *info) noexcept {
    if (device->copy2_capable()) {
#ifdef VK_KHR_copy_commands2
        if (vkCmdCopyImageToBuffer2 != nullptr) {
            vkCmdCopyImageToBuffer2(cmd, info);
        } else {
            vkCmdCopyImageToBuffer2KHR(cmd, info);
        }
#else
        vkCmdCopyImageToBuffer2(cmd, info);
#endif
        return;
    }
    // VkBufferImageCopy is 56 B: 4 elements cost 224 B of stack; larger
    // counts overflow to the heap.
    luisa::fixed_vector<VkBufferImageCopy, 4> regions;
    regions.reserve(info->regionCount);
    for (uint32_t i = 0u; i < info->regionCount; ++i) {
        auto const &r = info->pRegions[i];
        regions.emplace_back(VkBufferImageCopy{
            r.bufferOffset, r.bufferRowLength, r.bufferImageHeight,
            r.imageSubresource, r.imageOffset, r.imageExtent});
    }
    vkCmdCopyImageToBuffer(
        cmd, info->srcImage, info->srcImageLayout, info->dstBuffer,
        info->regionCount, regions.data());
}

inline void cmd_copy_image(
    VkCommandBuffer cmd, Device const *device,
    VkCopyImageInfo2 const *info) noexcept {
    if (device->copy2_capable()) {
#ifdef VK_KHR_copy_commands2
        if (vkCmdCopyImage2 != nullptr) {
            vkCmdCopyImage2(cmd, info);
        } else {
            vkCmdCopyImage2KHR(cmd, info);
        }
#else
        vkCmdCopyImage2(cmd, info);
#endif
        return;
    }
    // VkImageCopy is 68 B: 4 elements cost 272 B of stack; larger counts
    // overflow to the heap.
    luisa::fixed_vector<VkImageCopy, 4> regions;
    regions.reserve(info->regionCount);
    for (uint32_t i = 0u; i < info->regionCount; ++i) {
        auto const &r = info->pRegions[i];
        regions.emplace_back(VkImageCopy{
            r.srcSubresource, r.srcOffset, r.dstSubresource, r.dstOffset,
            r.extent});
    }
    vkCmdCopyImage(
        cmd, info->srcImage, info->srcImageLayout,
        info->dstImage, info->dstImageLayout,
        info->regionCount, regions.data());
}

// vkCmdPipelineBarrier with a capability-adaptive dispatch. The legacy path
// uses a single src/dst stage pair (the union across all barriers), which is
// conservative but functionally correct.
inline void cmd_pipeline_barrier(
    VkCommandBuffer cmd, Device const *device,
    VkDependencyInfo const *info) noexcept {
    if (device->sync2_capable()) {
#ifdef VK_KHR_synchronization2
        // Vulkan 1.2 devices expose the extension alias
        // (vkCmdPipelineBarrier2KHR) rather than the core-1.3 name; prefer
        // whichever the loader resolved.
        if (vkCmdPipelineBarrier2 != nullptr) {
            vkCmdPipelineBarrier2(cmd, info);
        } else {
            vkCmdPipelineBarrier2KHR(cmd, info);
        }
#else
        vkCmdPipelineBarrier2(cmd, info);
#endif
        return;
    }
    // Inline capacities follow the element type sizes so the barrier path
    // stays within a comparable stack budget (~700 B total): VkMemoryBarrier
    // is 24 B (8 x 192 B), VkBufferMemoryBarrier is 56 B (4 x 224 B), and
    // VkImageMemoryBarrier is 72 B (4 x 288 B). Larger counts overflow to
    // the heap.
    luisa::fixed_vector<VkMemoryBarrier, 8> memory_barriers;
    luisa::fixed_vector<VkBufferMemoryBarrier, 4> buffer_barriers;
    luisa::fixed_vector<VkImageMemoryBarrier, 4> image_barriers;
    VkPipelineStageFlags2 combined_src_stages = 0u;
    VkPipelineStageFlags2 combined_dst_stages = 0u;
    if (info->memoryBarrierCount != 0u) {
        memory_barriers.reserve(info->memoryBarrierCount);
        for (uint32_t i = 0u; i < info->memoryBarrierCount; ++i) {
            auto const &b = info->pMemoryBarriers[i];
            combined_src_stages |= b.srcStageMask;
            combined_dst_stages |= b.dstStageMask;
            memory_barriers.emplace_back(VkMemoryBarrier{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                nullptr,
                legacy_access_mask(b.srcAccessMask),
                legacy_access_mask(b.dstAccessMask)});
        }
    }
    if (info->bufferMemoryBarrierCount != 0u) {
        buffer_barriers.reserve(info->bufferMemoryBarrierCount);
        for (uint32_t i = 0u; i < info->bufferMemoryBarrierCount; ++i) {
            auto const &b = info->pBufferMemoryBarriers[i];
            combined_src_stages |= b.srcStageMask;
            combined_dst_stages |= b.dstStageMask;
            buffer_barriers.emplace_back(VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                legacy_access_mask(b.srcAccessMask),
                legacy_access_mask(b.dstAccessMask),
                b.srcQueueFamilyIndex,
                b.dstQueueFamilyIndex,
                b.buffer,
                b.offset,
                b.size});
        }
    }
    if (info->imageMemoryBarrierCount != 0u) {
        image_barriers.reserve(info->imageMemoryBarrierCount);
        for (uint32_t i = 0u; i < info->imageMemoryBarrierCount; ++i) {
            auto const &b = info->pImageMemoryBarriers[i];
            combined_src_stages |= b.srcStageMask;
            combined_dst_stages |= b.dstStageMask;
            image_barriers.emplace_back(VkImageMemoryBarrier{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                legacy_access_mask(b.srcAccessMask),
                legacy_access_mask(b.dstAccessMask),
                b.oldLayout,
                b.newLayout,
                b.srcQueueFamilyIndex,
                b.dstQueueFamilyIndex,
                b.image,
                b.subresourceRange});
        }
    }
    auto src_stage = legacy_stage_mask(combined_src_stages);
    auto dst_stage = legacy_stage_mask(combined_dst_stages);
    // VK_PIPELINE_STAGE_2_NONE (0) has no legacy equivalent; use top/bottom of
    // pipe, the standard "no-op dependency" idiom.
    if (src_stage == 0u) { src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; }
    if (dst_stage == 0u) { dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT; }
    vkCmdPipelineBarrier(
        cmd, src_stage, dst_stage, info->dependencyFlags,
        static_cast<uint32_t>(memory_barriers.size()),
        memory_barriers.data(),
        static_cast<uint32_t>(buffer_barriers.size()),
        buffer_barriers.data(),
        static_cast<uint32_t>(image_barriers.size()),
        image_barriers.data());
}

}// namespace lc::vk::detail
