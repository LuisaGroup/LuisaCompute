#pragma once

// Capability-adaptive queue submission for the Vulkan backend (plan P2.1).
//
// The backend submits through exactly two native entry points:
//
//   * `vkQueueSubmit2` (+ its VK_KHR_synchronization2 alias) on a
//     `Device::sync2_capable()` device, with `VkSubmitInfo2`,
//     `VkCommandBufferSubmitInfo` and `VkSemaphoreSubmitInfo`. This is what makes
//     a per-semaphore 64-bit stage mask expressible at all; the legacy
//     `VkSubmitInfo` only has one 32-bit `pWaitDstStageMask` per submission and no
//     per-signal stage array.
//   * `vkQueueSubmit` otherwise, i.e. on a Vulkan 1.2 floor
//     (`VulkanDeviceConfigExt::min_api_version() <= VK_API_VERSION_1_2`) or a
//     device that does not attest synchronization2. That path is unchanged: the
//     legacy `VkSubmitInfo` built here is field-for-field the one the backend
//     issued before P2.1, including the caller's `pNext` chain.
//
// Everything that could go wrong in this translation is a *host* decision, so the
// translation lives in this header as a pure function (`convert_submit2`) and is
// covered by `src/tests/unit/ext/test_vk_queue_submit_plan.cpp` without a device
// and without calling a Vulkan entry point. `queue_submit` (declared here,
// defined in `queue_submit_plan.cpp`) is the single funnel every backend
// submission site uses, so the semaphore conversion cannot be duplicated or
// drift between the dispatch, event, present and CUDA-interop paths.
//
// What the conversion carries:
//   * waits: the semaphore, its timeline value (0 for a binary semaphore, which
//     is what the legacy `VkTimelineSemaphoreSubmitInfo` arrays hold for binary
//     semaphores), and the *exact* legacy `pWaitDstStageMask` widened to 64 bits.
//     Every bit defined in `VkPipelineStageFlagBits` has the identical numeric
//     value in `VkPipelineStageFlagBits2` (asserted below), so widening is not an
//     approximation.
//   * signals: the semaphore, its timeline value, and a conservative
//     `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT`. The legacy structure carries no
//     per-signal stage, so there is nothing to preserve; ALL_COMMANDS is the
//     spec-valid choice that never under-orders a later wait.
//   * command buffers: same order, same count.
//   * `deviceIndex = 0`, the legacy structure's implicit single-device meaning.
//     `VkDeviceGroupSubmitInfo` has no `VkSubmitInfo2` equivalent and therefore
//     fails closed (see the pNext rule below).
//   * `VkTimelineSemaphoreSubmitInfo` is consumed: its values move into the
//     individual `VkSemaphoreSubmitInfo` entries and the structure is *not* left
//     in the `VkSubmitInfo2` pNext chain.
//
// pNext rule: only `VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO` is
// understood. Any other node (`VkProtectedSubmitInfo`,
// `VkDeviceGroupSubmitInfo`, `VkWin32KeyedMutexAcquireReleaseInfoKHR`,
// `VkD3D12FenceSubmitInfoKHR`, `VkPerformanceQuerySubmitInfoKHR`, ...) has no
// equivalent in `VkSubmitInfo2`, so the conversion reports
// `kUnsupportedPNext` and the caller submits the legacy way. Nothing is ever
// silently dropped.
#include <volk.h>

#include <luisa/core/stl/vector.h>

namespace lc::vk {
class Device;
}// namespace lc::vk

namespace lc::vk::detail {

// Inline capacity of the conversion scratch. The backend's shapes are one
// command buffer and at most two waits (the swapchain image-acquire binary
// semaphore plus the stream's own timeline), so nothing here is expected to grow;
// `luisa::fixed_vector` spills to the heap if a caller ever exceeds it.
constexpr auto kInlineSubmitEntries = 4u;

// One queue submission expressed in the legacy (Vulkan 1.0) vocabulary the backend
// uses everywhere. This is exactly the `VkSubmitInfo` the backend would issue
// today, plus the `VkFence` and the queue it goes to.
struct QueueSubmitRequest {
    VkCommandBuffer const *command_buffers{nullptr};
    uint32_t command_buffer_count{0u};
    VkSemaphore const *wait_semaphores{nullptr};
    // Legacy 32-bit dst stage masks, one per wait.
    VkPipelineStageFlags const *wait_stage_masks{nullptr};
    uint32_t wait_count{0u};
    VkSemaphore const *signal_semaphores{nullptr};
    uint32_t signal_count{0u};
    VkFence fence{VK_NULL_HANDLE};
    // The exact pNext chain the legacy `VkSubmitInfo` carries. Only a
    // `VkTimelineSemaphoreSubmitInfo` can be converted (P2.1); anything else makes
    // the submission take the legacy path.
    void const *pnext{nullptr};
};

// Which native entry point actually performed a submission.
enum class QueueSubmitVariant : uint32_t {
    kLegacy,
    kSubmit2,
};

enum class Submit2ConversionStatus : uint32_t {
    kOk,
    // The caller's pNext chain carries state a `VkSubmitInfo2` cannot express.
    kUnsupportedPNext,
    // The request contradicts itself (a count without its array, or a timeline
    // structure that describes a different number of semaphores than the
    // submission has). Never converted, never silently repaired.
    kInvalidRequest,
};

[[nodiscard]] constexpr const char *submit2_conversion_status_name(
    Submit2ConversionStatus status) noexcept {
    switch (status) {
        case Submit2ConversionStatus::kOk: return "ok";
        case Submit2ConversionStatus::kUnsupportedPNext: return "unsupported-pnext";
        case Submit2ConversionStatus::kInvalidRequest: return "invalid-request";
    }
    return "unknown";
}

// A legacy VkPipelineStageFlags bit and its 64-bit counterpart must agree
// numerically, otherwise "widen the legacy mask" would silently reorder the wait.
// These cover the bits the backend actually passes (top-of-pipe for a producer
// handoff, all-commands for an event wait, color-attachment-output for a
// swapchain acquire) plus the long-standing generic bits.
static_assert(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
static_assert(VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT == VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
static_assert(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT == VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
static_assert(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT == VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
static_assert(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
static_assert(VK_PIPELINE_STAGE_2_TRANSFER_BIT == VK_PIPELINE_STAGE_TRANSFER_BIT);
static_assert(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
static_assert(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
static_assert(VK_PIPELINE_STAGE_2_HOST_BIT == VK_PIPELINE_STAGE_HOST_BIT);
static_assert(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

// The exact legacy stage mask, widened. See the static_asserts above: for every
// bit Vulkan defines in `VkPipelineStageFlagBits` the 64-bit value is identical,
// so this preserves the legacy ordering instead of approximating it.
[[nodiscard]] constexpr VkPipelineStageFlags2 stage_mask2_from_legacy(
    VkPipelineStageFlags stage) noexcept {
    return static_cast<VkPipelineStageFlags2>(stage);
}

// Conversion scratch owned by the caller. The pointers inside `submit_info2` refer
// to this object's storage, so the object must not be copied or moved after
// `convert_submit2` filled it - which is why copies and moves are deleted and why
// `queue_submit` builds it on the stack right before the call.
struct Submit2Plan {
    Submit2Plan() noexcept = default;
    Submit2Plan(Submit2Plan const &) = delete;
    Submit2Plan &operator=(Submit2Plan const &) = delete;
    Submit2Plan(Submit2Plan &&) = delete;
    Submit2Plan &operator=(Submit2Plan &&) = delete;
    ~Submit2Plan() noexcept = default;

    luisa::fixed_vector<VkCommandBufferSubmitInfo, kInlineSubmitEntries> command_buffers;
    luisa::fixed_vector<VkSemaphoreSubmitInfo, kInlineSubmitEntries> waits;
    luisa::fixed_vector<VkSemaphoreSubmitInfo, kInlineSubmitEntries> signals;
    // Filled by `convert_submit2`; its pointers refer to the three vectors above.
    VkSubmitInfo2 submit_info2{};

    // `vkQueueSubmit2` takes an array of `VkSubmitInfo2`; the backend always
    // submits exactly one when there is anything to submit, and zero for a
    // fence-only submission - the same shape the legacy path uses.
    [[nodiscard]] uint32_t submit_count() const noexcept {
        return submit_info2.commandBufferInfoCount == 0u &&
                       submit_info2.waitSemaphoreInfoCount == 0u &&
                       submit_info2.signalSemaphoreInfoCount == 0u ?
                   0u :
                   1u;
    }
};

namespace queue_submit_detail {

// Walks the legacy pNext chain looking for the one node that has an equivalent in
// `VkSubmitInfo2`. `VkTimelineSemaphoreSubmitInfo` and its `_KHR` alias share one
// sType value, so a single case covers both spellings.
[[nodiscard]] inline VkTimelineSemaphoreSubmitInfo const *find_timeline_info(
    void const *pnext, bool &unsupported) noexcept {
    unsupported = false;
    auto found = static_cast<VkTimelineSemaphoreSubmitInfo const *>(nullptr);
    for (auto node = static_cast<VkBaseInStructure const *>(pnext);
         node != nullptr; node = node->pNext) {
        switch (node->sType) {
            case VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO:
                if (found != nullptr) {
                    // Two timeline structures in one chain is not something any
                    // caller means; refuse instead of picking one.
                    unsupported = true;
                    return nullptr;
                }
                found = reinterpret_cast<VkTimelineSemaphoreSubmitInfo const *>(node);
                break;
            default:
                unsupported = true;
                return nullptr;
        }
    }
    return found;
}

}// namespace queue_submit_detail

// Translates a legacy submission into `VkSubmitInfo2`. Pure: no Vulkan call, no
// allocation beyond the fixed vectors, no device. Returns why it refused rather
// than producing a partial plan.
[[nodiscard]] inline Submit2ConversionStatus convert_submit2(
    QueueSubmitRequest const &request, Submit2Plan &plan) noexcept {
    plan.command_buffers.clear();
    plan.waits.clear();
    plan.signals.clear();
    plan.submit_info2 = VkSubmitInfo2{};
    // Count/array consistency. Every backend call site passes a null pointer with a
    // zero count; a count without its array (or the reverse) means the caller
    // built the request wrong, and submitting it would submit garbage.
    auto consistent =
        (request.command_buffer_count == 0u) == (request.command_buffers == nullptr) &&
        (request.wait_count == 0u) == (request.wait_semaphores == nullptr) &&
        (request.wait_count == 0u) == (request.wait_stage_masks == nullptr) &&
        (request.signal_count == 0u) == (request.signal_semaphores == nullptr);
    if (!consistent) { return Submit2ConversionStatus::kInvalidRequest; }
    auto unsupported = false;
    auto timeline = queue_submit_detail::find_timeline_info(request.pnext, unsupported);
    if (unsupported) { return Submit2ConversionStatus::kUnsupportedPNext; }
    if (timeline != nullptr) {
        // Vulkan requires the two counts of the chained structure to match the
        // submission's semaphore counts; a mismatch means the values describe a
        // different submission than the one being converted.
        if (timeline->waitSemaphoreValueCount != request.wait_count ||
            timeline->signalSemaphoreValueCount != request.signal_count) {
            return Submit2ConversionStatus::kInvalidRequest;
        }
        if ((request.wait_count != 0u && timeline->pWaitSemaphoreValues == nullptr) ||
            (request.signal_count != 0u && timeline->pSignalSemaphoreValues == nullptr)) {
            return Submit2ConversionStatus::kInvalidRequest;
        }
    }
    plan.command_buffers.reserve(request.command_buffer_count);
    for (auto i = uint32_t{0u}; i < request.command_buffer_count; ++i) {
        plan.command_buffers.emplace_back(VkCommandBufferSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = request.command_buffers[i],
            .deviceMask = 0u});
    }
    // The timeline values, if any, are read once here: they are the only state of
    // the consumed `VkTimelineSemaphoreSubmitInfo` that has to survive.
    auto wait_value = [timeline](uint32_t i) noexcept -> uint64_t {
        return timeline == nullptr ? 0u : timeline->pWaitSemaphoreValues[i];
    };
    auto signal_value = [timeline](uint32_t i) noexcept -> uint64_t {
        return timeline == nullptr ? 0u : timeline->pSignalSemaphoreValues[i];
    };
    plan.waits.reserve(request.wait_count);
    for (auto i = uint32_t{0u}; i < request.wait_count; ++i) {
        plan.waits.emplace_back(VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = request.wait_semaphores[i],
            .value = wait_value(i),
            .stageMask = stage_mask2_from_legacy(request.wait_stage_masks[i]),
            .deviceIndex = 0u});
    }
    plan.signals.reserve(request.signal_count);
    for (auto i = uint32_t{0u}; i < request.signal_count; ++i) {
        plan.signals.emplace_back(VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = request.signal_semaphores[i],
            // Binary semaphores carry value 0, exactly like the legacy timeline
            // array.
            .value = signal_value(i),
            // The legacy structure has no per-signal stage mask, so there is
            // nothing to preserve: ALL_COMMANDS is the conservative,
            // spec-valid choice and never under-orders a later wait.
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0u});
    }
    plan.submit_info2 = VkSubmitInfo2{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        // Deliberately empty: the timeline structure was consumed above and must
        // not survive here (its values now live in the semaphore infos), and
        // nothing else may be chained on a converted submission.
        .pNext = nullptr,
        .flags = 0u,
        .waitSemaphoreInfoCount = static_cast<uint32_t>(plan.waits.size()),
        .pWaitSemaphoreInfos = plan.waits.empty() ? nullptr : plan.waits.data(),
        .commandBufferInfoCount = static_cast<uint32_t>(plan.command_buffers.size()),
        .pCommandBufferInfos = plan.command_buffers.empty() ? nullptr : plan.command_buffers.data(),
        .signalSemaphoreInfoCount = static_cast<uint32_t>(plan.signals.size()),
        .pSignalSemaphoreInfos = plan.signals.empty() ? nullptr : plan.signals.data()};
    return Submit2ConversionStatus::kOk;
}

// The exact legacy `VkSubmitInfo` for `request`, including the caller's pNext
// chain. On a non-sync2 device (Vulkan 1.2 floor) this reproduces the pre-P2.1
// submission field for field, so the legacy branch cannot drift.
[[nodiscard]] inline VkSubmitInfo make_legacy_submit_info(
    QueueSubmitRequest const &request) noexcept {
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = request.pnext;
    info.waitSemaphoreCount = request.wait_count;
    info.pWaitSemaphores = request.wait_semaphores;
    info.pWaitDstStageMask = request.wait_stage_masks;
    info.signalSemaphoreCount = request.signal_count;
    info.pSignalSemaphores = request.signal_semaphores;
    info.commandBufferCount = request.command_buffer_count;
    info.pCommandBuffers = request.command_buffers;
    return info;
}

// `vkQueueSubmit`'s submitCount for `request`: 0 for a fence-only submission (the
// present path retires extra frame fences that way), 1 otherwise. Chosen so the
// legacy branch issues the same call as before P2.1.
[[nodiscard]] inline uint32_t legacy_submit_count(
    QueueSubmitRequest const &request) noexcept {
    return request.command_buffer_count == 0u &&
                   request.wait_count == 0u &&
                   request.signal_count == 0u ?
               0u :
               1u;
}

// Performs one native submission of `request` on `queue`:
//   * `vkQueueSubmit2` when `device->sync2_capable()` and the request converts,
//     preferring the core entry point over the VK_KHR_synchronization2 alias;
//   * `vkQueueSubmit` otherwise (non-sync2 device, no resolvable entry point, or a
//     pNext chain with no `VkSubmitInfo2` equivalent).
// The caller keeps ownership of the queue mutex and of the
// `VulkanDeviceConfigExt` callbacks: this helper neither locks nor consults the
// config extension, so queue host-synchronization and callback behaviour stay
// exactly where they were.
[[nodiscard]] QueueSubmitVariant queue_submit(
    VkQueue queue, Device const *device,
    QueueSubmitRequest const &request) noexcept;

}// namespace lc::vk::detail