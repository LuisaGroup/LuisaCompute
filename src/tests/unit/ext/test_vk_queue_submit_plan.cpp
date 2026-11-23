// Test for the Vulkan backend's legacy-to-VkSubmitInfo2 conversion and its
// capability-adaptive submission helper (src/backends/vk/queue_submit_plan.h,
// plan P2.1).
// This test covers:
// - a binary wait/signal submission: waits keep their exact legacy stage mask
//   widened to 64 bits, binary semaphores carry timeline value 0, signals get a
//   spec-valid conservative ALL_COMMANDS mask and deviceIndex 0
// - a timeline wait/signal submission: the values of the chained
//   VkTimelineSemaphoreSubmitInfo move into VkSemaphoreSubmitInfo and that
//   structure is *not* reachable from the converted VkSubmitInfo2 pNext chain
// - a mixed submission (one binary wait with a swapchain-style stage mask, one
//   timeline wait, two signals with different values)
// - several command buffers: order and count preserved, deviceMask 0
// - a fence-only submission: submitCount 0 on both paths, the fence preserved
// - a pNext chain the conversion cannot express (protected submit, device group,
//   two timeline structures) fails closed to the legacy path instead of dropping
//   the state
// - count/array inconsistencies are rejected as invalid rather than converted
// - the legacy VkSubmitInfo is a field-for-field copy of the request (counts,
//   arrays, pNext chain), i.e. the Vulkan 1.2 path cannot drift
// Everything runs on the host: no Vulkan device, no shader, no entry point. The
// submit helper itself is not called here (it lives in the backend library); what
// is asserted is the plan the helper submits.
#include "ut/ut.hpp"

#include "queue_submit_plan.h"

#include <array>
#include <cstdint>

using namespace boost::ut;
using namespace boost::ut::literals;
using lc::vk::detail::convert_submit2;
using lc::vk::detail::legacy_submit_count;
using lc::vk::detail::make_legacy_submit_info;
using lc::vk::detail::QueueSubmitRequest;
using lc::vk::detail::stage_mask2_from_legacy;
using lc::vk::detail::Submit2ConversionStatus;
using lc::vk::detail::Submit2Plan;

namespace {

// Fake handles: the conversion only copies and compares them, so distinct
// addresses are distinct objects as far as it can tell. Distinct objects rather
// than integer casts keep the initializers portable across the platforms where a
// Vulkan handle is a pointer.
struct FakeHandles {
    int slots[8]{0, 1, 2, 3, 4, 5, 6, 7};
};
FakeHandles g_fake_handles;

constexpr auto kWaitBinarySlot = 0u;
constexpr auto kWaitTimelineSlot = 1u;
constexpr auto kSignalASlot = 2u;
constexpr auto kSignalBSlot = 3u;
constexpr auto kCommandBufferASlot = 4u;
constexpr auto kCommandBufferBSlot = 5u;
constexpr auto kFenceSlot = 6u;

[[nodiscard]] VkSemaphore fake_semaphore(uint32_t slot) noexcept {
    return reinterpret_cast<VkSemaphore>(&g_fake_handles.slots[slot]);
}
[[nodiscard]] VkCommandBuffer fake_command_buffer(uint32_t slot) noexcept {
    return reinterpret_cast<VkCommandBuffer>(&g_fake_handles.slots[slot]);
}
[[nodiscard]] VkFence fake_fence(uint32_t slot) noexcept {
    return reinterpret_cast<VkFence>(&g_fake_handles.slots[slot]);
}

// A pNext node the conversion must refuse: "protected submit" has no
// VkSubmitInfo2 equivalent.
struct FakeProtectedSubmitInfo {
    VkStructureType sType{VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO};
    void *pNext{nullptr};
    VkBool32 protectedSubmit{VK_TRUE};
};

// A second *understood* node type, used to prove that a supported chain node
// alone is not enough: the chain is only convertible when every node is the
// timeline structure.
struct FakeDeviceGroupSubmitInfo {
    VkStructureType sType{VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO};
    void *pNext{nullptr};
    uint32_t waitSemaphoreCount{0u};
    const uint32_t *pWaitSemaphoreDeviceIndices{nullptr};
    uint32_t commandBufferCount{0u};
    const uint32_t *pCommandBufferDeviceMasks{nullptr};
    uint32_t signalSemaphoreCount{0u};
    const uint32_t *pSignalSemaphoreDeviceIndices{nullptr};
};

void test_binary_waits_and_signals() {
    std::array<VkSemaphore, 1> wait_semaphores{fake_semaphore(kWaitBinarySlot)};
    std::array<VkPipelineStageFlags, 1> wait_stages{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    std::array<VkSemaphore, 1> signal_semaphores{fake_semaphore(kSignalASlot)};
    std::array<VkCommandBuffer, 1> command_buffers{fake_command_buffer(kCommandBufferASlot)};
    QueueSubmitRequest request{};
    request.command_buffers = command_buffers.data();
    request.command_buffer_count = 1u;
    request.wait_semaphores = wait_semaphores.data();
    request.wait_stage_masks = wait_stages.data();
    request.wait_count = 1u;
    request.signal_semaphores = signal_semaphores.data();
    request.signal_count = 1u;
    request.fence = fake_fence(kFenceSlot);
    // No timeline structure: every semaphore here is binary, which is exactly the
    // shape the swapchain present path has.
    Submit2Plan plan;
    expect(convert_submit2(request, plan) == Submit2ConversionStatus::kOk)
        << "a binary wait/signal submission must convert";
    expect(plan.waits.size() == 1u && plan.signals.size() == 1u &&
           plan.command_buffers.size() == 1u)
        << "one converted entry per legacy semaphore / command buffer";
    expect(plan.waits[0].semaphore == fake_semaphore(kWaitBinarySlot));
    expect(plan.waits[0].value == 0u)
        << "a binary semaphore carries timeline value 0, like the legacy array";
    expect(plan.waits[0].stageMask == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
        << "the exact legacy dst stage must survive as the 64-bit stage mask";
    expect(plan.waits[0].deviceIndex == 0u)
        << "the legacy structure has no device group, so the index stays 0";
    expect(plan.waits[0].sType == VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);
    expect(plan.signals[0].semaphore == fake_semaphore(kSignalASlot));
    expect(plan.signals[0].value == 0u)
        << "a binary signal carries timeline value 0";
    expect(plan.signals[0].stageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        << "the legacy structure has no per-signal stage array, so the conversion "
           "must use the conservative ALL_COMMANDS mask";
    expect(plan.signals[0].deviceIndex == 0u);
    expect(plan.command_buffers[0].commandBuffer == fake_command_buffer(kCommandBufferASlot));
    expect(plan.command_buffers[0].deviceMask == 0u);
    expect(plan.command_buffers[0].sType == VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO);
    expect(plan.submit_info2.sType == VK_STRUCTURE_TYPE_SUBMIT_INFO_2);
    expect(plan.submit_info2.pNext == nullptr)
        << "the converted submission must not chain anything";
    expect(plan.submit_info2.flags == 0u);
    expect(plan.submit_info2.waitSemaphoreInfoCount == 1u);
    expect(plan.submit_info2.commandBufferInfoCount == 1u);
    expect(plan.submit_info2.signalSemaphoreInfoCount == 1u);
    expect(plan.submit_info2.pWaitSemaphoreInfos == plan.waits.data())
        << "the converted structure must point at the plan's own storage";
    expect(plan.submit_info2.pCommandBufferInfos == plan.command_buffers.data());
    expect(plan.submit_info2.pSignalSemaphoreInfos == plan.signals.data());
    expect(plan.submit_count() == 1u);
}

void test_timeline_values_move_out_of_the_legacy_structure() {
    std::array<uint64_t, 1> wait_values{7u};
    std::array<uint64_t, 1> signal_values{9u};
    std::array<VkSemaphore, 1> wait_semaphores{fake_semaphore(kWaitTimelineSlot)};
    std::array<VkPipelineStageFlags, 1> wait_stages{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    std::array<VkSemaphore, 1> signal_semaphores{fake_semaphore(kSignalASlot)};
    std::array<VkCommandBuffer, 1> command_buffers{fake_command_buffer(kCommandBufferASlot)};
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 1u;
    timeline_info.pWaitSemaphoreValues = wait_values.data();
    timeline_info.signalSemaphoreValueCount = 1u;
    timeline_info.pSignalSemaphoreValues = signal_values.data();
    QueueSubmitRequest request{};
    request.command_buffers = command_buffers.data();
    request.command_buffer_count = 1u;
    request.wait_semaphores = wait_semaphores.data();
    request.wait_stage_masks = wait_stages.data();
    request.wait_count = 1u;
    request.signal_semaphores = signal_semaphores.data();
    request.signal_count = 1u;
    request.pnext = &timeline_info;
    Submit2Plan plan;
    expect(convert_submit2(request, plan) == Submit2ConversionStatus::kOk)
        << "a timeline submission must convert";
    expect(plan.waits[0].value == 7u)
        << "the timeline wait value must move into VkSemaphoreSubmitInfo";
    expect(plan.signals[0].value == 9u)
        << "the timeline signal value must move into VkSemaphoreSubmitInfo";
    expect(plan.waits[0].stageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    // VkTimelineSemaphoreSubmitInfo does not exist in a synchronization2
    // submission; leaving it chained would be invalid.
    expect(plan.submit_info2.pNext == nullptr)
        << "the consumed timeline structure must not remain in the pNext chain";
    auto chain_is_clean = true;
    for (auto node = static_cast<VkBaseInStructure const *>(plan.submit_info2.pNext);
         node != nullptr; node = node->pNext) {
        chain_is_clean = false;
    }
    expect(chain_is_clean) << "the SubmitInfo2 chain must be empty";
}

void test_mixed_waits_and_signals() {
    // A swapchain-style binary acquire plus the stream's own timeline, and two
    // signals with different values: the per-semaphore mapping must not collapse
    // into one submission-wide mask.
    std::array<uint64_t, 2> wait_values{0u, 11u};
    std::array<uint64_t, 2> signal_values{13u, 0u};
    std::array<VkSemaphore, 2> wait_semaphores{fake_semaphore(kWaitBinarySlot), fake_semaphore(kWaitTimelineSlot)};
    std::array<VkPipelineStageFlags, 2> wait_stages{
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
    std::array<VkSemaphore, 2> signal_semaphores{fake_semaphore(kSignalASlot), fake_semaphore(kSignalBSlot)};
    std::array<VkCommandBuffer, 2> command_buffers{fake_command_buffer(kCommandBufferASlot), fake_command_buffer(kCommandBufferBSlot)};
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 2u;
    timeline_info.pWaitSemaphoreValues = wait_values.data();
    timeline_info.signalSemaphoreValueCount = 2u;
    timeline_info.pSignalSemaphoreValues = signal_values.data();
    QueueSubmitRequest request{};
    request.command_buffers = command_buffers.data();
    request.command_buffer_count = 2u;
    request.wait_semaphores = wait_semaphores.data();
    request.wait_stage_masks = wait_stages.data();
    request.wait_count = 2u;
    request.signal_semaphores = signal_semaphores.data();
    request.signal_count = 2u;
    request.pnext = &timeline_info;
    Submit2Plan plan;
    expect(convert_submit2(request, plan) == Submit2ConversionStatus::kOk);
    expect(plan.waits.size() == 2u && plan.signals.size() == 2u &&
           plan.command_buffers.size() == 2u);
    expect(plan.waits[0].semaphore == fake_semaphore(kWaitBinarySlot) && plan.waits[0].value == 0u)
        << "the binary wait keeps value 0";
    expect(plan.waits[1].semaphore == fake_semaphore(kWaitTimelineSlot) && plan.waits[1].value == 11u)
        << "the timeline wait keeps its own value";
    expect(plan.waits[0].stageMask == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT &&
           plan.waits[1].stageMask == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
        << "each wait keeps its own legacy stage mask, not a merged one";
    expect(plan.signals[0].value == 13u && plan.signals[1].value == 0u)
        << "each signal keeps its own timeline value";
    expect(plan.signals[0].stageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
           plan.signals[1].stageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    expect(plan.command_buffers[0].commandBuffer == fake_command_buffer(kCommandBufferASlot) &&
           plan.command_buffers[1].commandBuffer == fake_command_buffer(kCommandBufferBSlot))
        << "command-buffer order must be preserved";
    expect(plan.submit_info2.commandBufferInfoCount == 2u)
        << "command-buffer count must be preserved";
}

void test_fence_only_submission() {
    // The present path retires the extra frame fences of a batch with
    // submitCount == 0 submissions that only carry a fence.
    QueueSubmitRequest request{};
    request.fence = fake_fence(kFenceSlot);
    Submit2Plan plan;
    expect(convert_submit2(request, plan) == Submit2ConversionStatus::kOk);
    expect(plan.waits.empty() && plan.signals.empty() && plan.command_buffers.empty());
    expect(plan.submit_count() == 0u)
        << "an empty submission must stay submitCount 0 on the sync2 path";
    expect(legacy_submit_count(request) == 0u)
        << "an empty submission must stay submitCount 0 on the legacy path";
    expect(plan.submit_info2.waitSemaphoreInfoCount == 0u &&
           plan.submit_info2.commandBufferInfoCount == 0u &&
           plan.submit_info2.signalSemaphoreInfoCount == 0u);
    expect(plan.submit_info2.pWaitSemaphoreInfos == nullptr &&
           plan.submit_info2.pCommandBufferInfos == nullptr &&
           plan.submit_info2.pSignalSemaphoreInfos == nullptr)
        << "a zero count must come with a null array";
    auto legacy = make_legacy_submit_info(request);
    expect(legacy.sType == VK_STRUCTURE_TYPE_SUBMIT_INFO);
    expect(legacy.commandBufferCount == 0u && legacy.pCommandBuffers == nullptr);
    expect(legacy.waitSemaphoreCount == 0u && legacy.signalSemaphoreCount == 0u);
    expect(legacy.pNext == nullptr);
}

void test_unsupported_pnext_falls_back() {
    std::array<VkSemaphore, 1> signal_semaphores{fake_semaphore(kSignalASlot)};
    std::array<VkCommandBuffer, 1> command_buffers{fake_command_buffer(kCommandBufferASlot)};
    QueueSubmitRequest request{};
    request.command_buffers = command_buffers.data();
    request.command_buffer_count = 1u;
    request.signal_semaphores = signal_semaphores.data();
    request.signal_count = 1u;
    {
        FakeProtectedSubmitInfo protected_submit{};
        request.pnext = &protected_submit;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kUnsupportedPNext)
            << "protected-submit state has no VkSubmitInfo2 equivalent, so the "
               "conversion must refuse instead of dropping it";
    }
    {
        FakeDeviceGroupSubmitInfo device_group{};
        request.pnext = &device_group;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kUnsupportedPNext)
            << "device-group state has no VkSubmitInfo2 equivalent";
    }
    {
        // Supported node followed by an unsupported one: the whole chain is
        // inconvertible, never partially translated.
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.waitSemaphoreValueCount = 0u;
        timeline_info.signalSemaphoreValueCount = 0u;
        FakeProtectedSubmitInfo protected_submit{};
        timeline_info.pNext = &protected_submit;
        request.pnext = &timeline_info;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kUnsupportedPNext)
            << "a convertible node must not hide an inconvertible one behind it";
    }
    {
        // Two timeline structures in one chain: not a shape any caller means, and
        // picking one of them would silently drop the other's values.
        VkTimelineSemaphoreSubmitInfo first{};
        first.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        VkTimelineSemaphoreSubmitInfo second{};
        second.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        first.pNext = &second;
        request.pnext = &first;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kUnsupportedPNext)
            << "a duplicated timeline structure must fall back, not be resolved "
               "arbitrarily";
    }
    // The legacy build of the same request keeps the chain verbatim, which is how
    // the fallback preserves the state.
    FakeProtectedSubmitInfo protected_submit{};
    request.pnext = &protected_submit;
    auto legacy = make_legacy_submit_info(request);
    expect(legacy.pNext == &protected_submit)
        << "the legacy path must carry the unhandled chain unchanged";
}

void test_invalid_requests() {
    std::array<VkSemaphore, 1> signal_semaphores{fake_semaphore(kSignalASlot)};
    std::array<uint64_t, 1> signal_values{5u};
    {
        // A count without its array.
        QueueSubmitRequest request{};
        request.wait_count = 1u;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kInvalidRequest)
            << "a wait count without its semaphore array must be rejected";
    }
    {
        // An array without its count.
        QueueSubmitRequest request{};
        request.signal_semaphores = signal_semaphores.data();
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kInvalidRequest)
            << "a semaphore array without its count must be rejected";
    }
    {
        // Wait semaphores but no stage masks: the legacy structure would read a
        // null array.
        QueueSubmitRequest request{};
        request.wait_semaphores = signal_semaphores.data();
        request.wait_count = 1u;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kInvalidRequest)
            << "wait semaphores without dst stage masks must be rejected";
    }
    {
        // A timeline structure that describes a different submission.
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.waitSemaphoreValueCount = 1u;
        timeline_info.pWaitSemaphoreValues = signal_values.data();
        timeline_info.signalSemaphoreValueCount = 1u;
        timeline_info.pSignalSemaphoreValues = signal_values.data();
        QueueSubmitRequest request{};
        request.signal_semaphores = signal_semaphores.data();
        request.signal_count = 1u;
        request.pnext = &timeline_info;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kInvalidRequest)
            << "a timeline wait count that does not match the submission must be "
               "rejected";
    }
    {
        // Counts agree but the value array is missing.
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.waitSemaphoreValueCount = 0u;
        timeline_info.signalSemaphoreValueCount = 1u;
        timeline_info.pSignalSemaphoreValues = nullptr;
        QueueSubmitRequest request{};
        request.signal_semaphores = signal_semaphores.data();
        request.signal_count = 1u;
        request.pnext = &timeline_info;
        Submit2Plan plan;
        expect(convert_submit2(request, plan) == Submit2ConversionStatus::kInvalidRequest)
            << "a timeline count without its value array must be rejected";
    }
}

void test_legacy_reproduction() {
    // The Vulkan 1.2 path must stay bit-identical: the legacy VkSubmitInfo is a
    // field-for-field copy of the request, including the timeline chain.
    std::array<uint64_t, 1> wait_values{3u};
    std::array<uint64_t, 1> signal_values{4u};
    std::array<VkSemaphore, 1> wait_semaphores{fake_semaphore(kWaitTimelineSlot)};
    std::array<VkPipelineStageFlags, 1> wait_stages{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    std::array<VkSemaphore, 1> signal_semaphores{fake_semaphore(kSignalASlot)};
    std::array<VkCommandBuffer, 1> command_buffers{fake_command_buffer(kCommandBufferASlot)};
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = 1u;
    timeline_info.pWaitSemaphoreValues = wait_values.data();
    timeline_info.signalSemaphoreValueCount = 1u;
    timeline_info.pSignalSemaphoreValues = signal_values.data();
    QueueSubmitRequest request{};
    request.command_buffers = command_buffers.data();
    request.command_buffer_count = 1u;
    request.wait_semaphores = wait_semaphores.data();
    request.wait_stage_masks = wait_stages.data();
    request.wait_count = 1u;
    request.signal_semaphores = signal_semaphores.data();
    request.signal_count = 1u;
    request.pnext = &timeline_info;
    auto legacy = make_legacy_submit_info(request);
    expect(legacy.sType == VK_STRUCTURE_TYPE_SUBMIT_INFO);
    expect(legacy.pNext == &timeline_info)
        << "the legacy path keeps the timeline structure chained";
    expect(legacy.waitSemaphoreCount == 1u && legacy.pWaitSemaphores == wait_semaphores.data())
        << "wait semaphores are passed through untouched";
    expect(legacy.pWaitDstStageMask == wait_stages.data())
        << "the legacy 32-bit dst stage array is passed through untouched";
    expect(legacy.signalSemaphoreCount == 1u && legacy.pSignalSemaphores == signal_semaphores.data());
    expect(legacy.commandBufferCount == 1u && legacy.pCommandBuffers == command_buffers.data());
    expect(legacy_submit_count(request) == 1u);
    // The fence is not part of VkSubmitInfo; both paths pass it to the entry point.
    expect(request.fence == VK_NULL_HANDLE);
}

void test_stage_mask_widening_is_exact() {
    // The widening used for wait stage masks is only sound because the legacy and
    // 64-bit values agree bit for bit; assert the function itself, not just the
    // static_asserts in the header.
    expect(stage_mask2_from_legacy(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) ==
           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    expect(stage_mask2_from_legacy(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) ==
           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
    expect(stage_mask2_from_legacy(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) ==
           (VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT))
        << "combined masks must widen bit for bit";
    expect(stage_mask2_from_legacy(0u) == 0u);
}

}// namespace

namespace {

void register_tests() {
    "vk_submit2_binary_waits_and_signals"_test = [] { test_binary_waits_and_signals(); };
    "vk_submit2_timeline_values_are_consumed"_test = [] { test_timeline_values_move_out_of_the_legacy_structure(); };
    "vk_submit2_mixed_waits_and_signals"_test = [] { test_mixed_waits_and_signals(); };
    "vk_submit2_fence_only_submission"_test = [] { test_fence_only_submission(); };
    "vk_submit2_unsupported_pnext_falls_back"_test = [] { test_unsupported_pnext_falls_back(); };
    "vk_submit2_invalid_requests_are_rejected"_test = [] { test_invalid_requests(); };
    "vk_submit2_legacy_reproduction"_test = [] { test_legacy_reproduction(); };
    "vk_submit2_stage_mask_widening_is_exact"_test = [] { test_stage_mask_widening_is_exact(); };
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    register_tests();
    return 0;
}