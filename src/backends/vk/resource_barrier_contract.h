#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan_core.h>
#include <luisa/core/stl/memory.h>

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

// ---------------------------------------------------------------------------
// VK_KHR_unified_image_layouts
// ---------------------------------------------------------------------------
//
// `VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR::unifiedImageLayouts`
// promises that ename:VK_IMAGE_LAYOUT_GENERAL incurs no loss in efficiency
// where it is valid, so a backend-owned command no longer has to pay for a
// per-subresource layout transition just to reach an "optimal" layout. The
// tracker may therefore keep every tracked mip in GENERAL instead of switching
// it between SHADER_READ_ONLY_OPTIMAL / TRANSFER_SRC_OPTIMAL /
// TRANSFER_DST_OPTIMAL / DEPTH_STENCIL_READ_ONLY_OPTIMAL.
//
// This is a *layout-only* policy, never a synchronization policy:
//   * every stage mask and every access mask the tracker emits is identical
//     with and without the extension, and a barrier is still recorded for
//     every hazard - including the "oldLayout == newLayout" case, which Vulkan
//     explicitly allows so that availability/visibility can be expressed
//     without a transition;
//   * `TextureLayoutContract::EXPLICIT_NATIVE` contracts (custom commands,
//     before/after states, bindless descriptor snapshots) keep the exact
//     VkImageLayout they publish, because a descriptor or a render-pass state
//     that disagrees with the barrier is a correctness bug, not a lost
//     optimization;
//   * attachment *writes* keep their attachment layout, so the render passes
//     built by the raster path are unaffected.
//
// The policy is deliberately a pure function so both the enabled and the
// disabled path are unit-testable without a Vulkan device.
struct ImageLayoutPolicy {
    // Backend-owned (generic) usages resolve to VK_IMAGE_LAYOUT_GENERAL.
    bool general_for_generic_usage{false};
    // True only when the reason is VK_KHR_unified_image_layouts itself. A
    // simultaneous-access image already took the GENERAL fast path before the
    // extension existed, so that optimization must never be reported as "this
    // extension is in use".
    bool uses_unified_image_layouts{false};
};

[[nodiscard]] constexpr ImageLayoutPolicy
plan_unified_image_layout_policy(
    bool unified_image_layouts_enabled,
    bool simultaneous_access) noexcept {
    return ImageLayoutPolicy{
        // The extension may only *widen* the existing optimization, never
        // narrow it: an unsupported/disabled extension still leaves the
        // simultaneous-access fast path intact.
        unified_image_layouts_enabled || simultaneous_access,
        unified_image_layouts_enabled};
}

// Backend-owned commands query the layout selected by the tracker, so images
// created for simultaneous access may stay in GENERAL to avoid unnecessary
// transitions. Native/custom commands instead publish the exact layout they
// will pass to Vulkan; changing that value would make the barrier disagree
// with the command's VkImageLayout argument.
[[nodiscard]] constexpr VkImageLayout resolve_texture_barrier_layout(
    ImageLayoutPolicy policy,
    VkAccessFlags2 access,
    VkImageLayout requested_layout,
    TextureLayoutContract contract) noexcept {
    if (contract == TextureLayoutContract::EXPLICIT_NATIVE ||
        !policy.general_for_generic_usage ||
        (access & (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) != 0u) {
        return requested_layout;
    }
    return VK_IMAGE_LAYOUT_GENERAL;
}

// Pre-existing (extension-free) spelling: an image created for simultaneous
// access already avoids the transitions, and VK_KHR_unified_image_layouts is
// off.
[[nodiscard]] constexpr VkImageLayout resolve_texture_barrier_layout(
    bool simultaneous_access,
    VkAccessFlags2 access,
    VkImageLayout requested_layout,
    TextureLayoutContract contract) noexcept {
    return resolve_texture_barrier_layout(
        plan_unified_image_layout_policy(false, simultaneous_access),
        access, requested_layout, contract);
}

// A tracked mip keeps a usage-specific layout only while the recorded access
// still fits that layout's single compatible access bit; anything else
// collapses to GENERAL, which is valid for every access. This is the
// pre-VK_KHR_unified_image_layouts rule and stays in force unchanged whenever
// the policy is inactive.
[[nodiscard]] constexpr VkImageLayout limit_texture_layout_to_access(
    VkImageLayout layout, VkAccessFlags2 access) noexcept {
    switch (layout) {
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return access == VK_ACCESS_2_SHADER_READ_BIT ?
                       layout :
                       VK_IMAGE_LAYOUT_GENERAL;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return access == VK_ACCESS_2_TRANSFER_READ_BIT ?
                       layout :
                       VK_IMAGE_LAYOUT_GENERAL;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return access == VK_ACCESS_2_TRANSFER_WRITE_BIT ?
                       layout :
                       VK_IMAGE_LAYOUT_GENERAL;
        default: return layout;
    }
}

// One tracked mip's VkImageMemoryBarrier2 decision. Grouping the hazard and the
// layout pair in one value keeps the two independent: the layout policy is
// applied when the layout is *selected* (`resolve_texture_barrier_layout`), so
// no stage or access mask can depend on it.
struct ImageBarrierPlan {
    VkPipelineStageFlags2 src_stage_mask{0u};
    VkPipelineStageFlags2 dst_stage_mask{0u};
    VkAccessFlags2 src_access_mask{0u};
    VkAccessFlags2 dst_access_mask{0u};
    VkImageLayout old_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    VkImageLayout new_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    // Vulkan 1.3/1.4 allow oldLayout == newLayout; such a barrier only makes
    // earlier accesses available and visible. It is still required, so the
    // tracker never drops a barrier because the layout happens to match.
    [[nodiscard]] constexpr bool transitions_layout() const noexcept {
        return old_layout != new_layout;
    }
    [[nodiscard]] constexpr bool has_hazard() const noexcept {
        return src_stage_mask != 0u || dst_stage_mask != 0u ||
               src_access_mask != 0u || dst_access_mask != 0u;
    }
};

[[nodiscard]] constexpr ImageBarrierPlan plan_tracked_image_barrier(
    VkPipelineStageFlags2 before_stage,
    VkAccessFlags2 before_access,
    VkImageLayout before_layout,
    VkPipelineStageFlags2 after_stage,
    VkAccessFlags2 after_access,
    VkImageLayout after_layout) noexcept {
    return ImageBarrierPlan{
        before_stage,
        after_stage,
        before_access,
        after_access,
        before_layout,
        limit_texture_layout_to_access(after_layout, after_access)};
}

// Vulkan non-dispatchable handles are either opaque pointers or uint64_t
// values. Preserve every handle bit in an integer key so distinct Luisa
// wrappers of the same VkBuffer share one hazard state without retaining a
// representative wrapper pointer.
[[nodiscard]] inline NativeBufferIdentity
native_buffer_identity(VkBuffer buffer) noexcept {
    static_assert(sizeof(VkBuffer) == sizeof(NativeBufferIdentity));
    return luisa::bit_cast<NativeBufferIdentity>(buffer);
}

[[nodiscard]] inline NativeImageIdentity
native_image_identity(VkImage image) noexcept {
    static_assert(sizeof(VkImage) == sizeof(NativeImageIdentity));
    return luisa::bit_cast<NativeImageIdentity>(image);
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
// VK_KHR_maintenance10's depth/stencil copy format feature bits
// (VK_FORMAT_FEATURE_2_{DEPTH,STENCIL}_COPY_ON_{COMPUTE,TRANSFER}_BIT_KHR) only
// decide whether such a copy may run on a non-graphics queue at all; the copy
// queue scope above already carries TRANSFER_READ/TRANSFER_WRITE, so the
// relaxation has to be applied where a depth/stencil copy chooses its stream
// (stream.cpp / texture.cpp), not in this barrier contract.

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

// ---------------------------------------------------------------------------
// VK_KHR_maintenance8 / VK_KHR_maintenance9 / VK_KHR_maintenance10
// ---------------------------------------------------------------------------
//
// The barrier tracker negotiates all three extensions (see
// vk_extension_registry.h) because they are free capability, but it takes no
// fast path for any of them: every relaxation they add removes API friction
// that lives outside this tracker. They are therefore never reported through
// `Device::mark_optional_extension_active`, and the enabled/disabled paths of
// the tracker are byte-identical. The relaxations, and the owner that would
// have to adopt each one:
//
// maintenance8, "allow copies between depth/stencil and matching color
// attachments": the backend copies a depth/stencil aspect straight to/from a
// VkBuffer (VkBufferImageCopy2 with Texture::get_aspect(), see stream.cpp), so
// it never goes through a size-compatible color image and has no precondition
// to relax.
//
// maintenance8, "src/dst sync scopes work when doing queue family ownership
// transfers", and maintenance9, "ownership transfers are optional for buffers
// and linear images, and for optimal images the queue pair advertises in
// VkQueueFamilyOwnershipTransferPropertiesKHR::optimalImageTransferToQueueFamilies":
// every VkImageMemoryBarrier2/VkBufferMemoryBarrier2 this tracker emits uses
// VK_QUEUE_FAMILY_IGNORED (see ResourceBarrier::barrier_filter) because
// backend-owned resources are created in VK_SHARING_MODE_CONCURRENT over the
// participating families (see queue_family_contract.h). No ownership transfer
// is ever recorded, so there is nothing to relax here. Reaping the benefit
// means switching resource creation to VK_SHARING_MODE_EXCLUSIVE, which lives
// in texture.cpp / buffer.cpp / device.cpp.
//
// maintenance9, "image memory barriers and layout transitions on 3D images
// created with VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT are scoped to the
// slices named by VkImageSubresourceRange": this tracker never addresses a
// slice subrange - its VkImageSubresourceRange uses baseArrayLayer 0 and
// layerCount 1, i.e. exactly the whole mip/subresource that pre-maintenance9
// required. Slice-scoped barriers only help render-pass attachment views, i.e.
// the raster path.
//
// maintenance9, "VkDevice with no queues", "vkCmdSetEvent2 without a
// dependency", "VK_QUERY_POOL_CREATE_RESET_BIT_KHR",
// "defaultVertexAttributeValue", "image2DViewOf3DSparse" and "any integer bit
// width for bitwise ops": queue/event/query-pool/vertex-input/shader-codegen
// features. The tracker creates no query pool, records no vkCmdSetEvent2, and
// emits no shader code.
//
// maintenance10, "image format feature bits for copying depth or stencil
// aspects on non-graphics queue families"
// (VK_FORMAT_FEATURE_2_{DEPTH,STENCIL}_COPY_ON_{COMPUTE,TRANSFER}_BIT_KHR): a
// queue-selection question. The tracker's copy queue scope already permits
// TRANSFER_READ/TRANSFER_WRITE (see copy_queue_access_mask), so the relaxation
// belongs where a depth/stencil copy picks its stream, not here.
//
// maintenance10, "sRGB resolve transfer function" and "resolve mode plus
// depth/stencil resolve in vkCmdResolveImage2": the backend never calls
// vkCmdResolveImage/vkCmdResolveImage2 at all (the raster path builds render
// passes with pResolveAttachments = nullptr, see raster_pipeline.cpp), so no
// resolve contract exists to relax.
//
// maintenance8, "an additional 64 access flags"
// (VkMemoryBarrierAccessFlags3KHR in the barrier pNext chain): the tracker
// terminates every barrier with pNext = nullptr and only ever sets
// VkAccessFlagBits2 bits, all of which are already defined; VkAccessFlagBits3KHR
// currently has no bit beyond VK_ACCESS_3_NONE_KHR.

}// namespace lc::vk::detail
