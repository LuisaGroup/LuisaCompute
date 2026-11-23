// Implementation of the capability-adaptive submission funnel (plan P2.1).
//
// The conversion itself is inline in `queue_submit_plan.h` so the host test can
// cover it without this translation unit (and therefore without the backend
// library). What lives here is the single native call: the sync2/legacy decision,
// the alias selection, and the entry-point invocation.
#include "queue_submit_plan.h"

#include "device.h"
#include "log.h"

namespace lc::vk::detail {

namespace {

// True when a `vkQueueSubmit2` entry point was actually resolved. On a Vulkan 1.3
// device volk fills the core name; on a Vulkan 1.2 device with
// VK_KHR_synchronization2 enabled only the `_KHR` alias is guaranteed, which is
// the same rule `cmd_pipeline_barrier` follows for vkCmdPipelineBarrier2.
//
// `Device::sync2_capable()` already implies "the feature is enabled on the logical
// device" (core 1.3, or the extension plus its feature), so this only decides
// which spelling of an available entry point to call.
struct Submit2EntryPoints {
    PFN_vkQueueSubmit2 core{nullptr};
    PFN_vkQueueSubmit2KHR alias{nullptr};

    [[nodiscard]] bool available() const noexcept {
        return core != nullptr || alias != nullptr;
    }
};

[[nodiscard]] Submit2EntryPoints submit2_entry_points() noexcept {
#ifdef VK_KHR_synchronization2
    return Submit2EntryPoints{
        .core = ::vkQueueSubmit2,
        .alias = ::vkQueueSubmit2KHR};
#else
    return Submit2EntryPoints{.core = ::vkQueueSubmit2, .alias = nullptr};
#endif
}

void call_submit2(Submit2EntryPoints const &entry_points, VkQueue queue,
                  uint32_t submit_count, VkSubmitInfo2 const *submits,
                  VkFence fence) noexcept {
    if (entry_points.core != nullptr) {
        VK_CHECK_RESULT(entry_points.core(queue, submit_count, submits, fence));
    } else {
        VK_CHECK_RESULT(entry_points.alias(queue, submit_count, submits, fence));
    }
}

}// namespace

QueueSubmitVariant queue_submit(VkQueue queue, Device const *device,
                                QueueSubmitRequest const &request) noexcept {
    if (device != nullptr && device->sync2_capable()) {
        auto entry_points = submit2_entry_points();
        if (entry_points.available()) {
            // Convert first: the legacy path must stay available as the fallback
            // for a request the conversion cannot express.
            Submit2Plan plan;
            auto status = convert_submit2(request, plan);
            LUISA_ASSERT(
                status != Submit2ConversionStatus::kInvalidRequest,
                "Invalid Vulkan queue submission: the counts and arrays of the "
                "request or of its timeline-semaphore values disagree ({}).",
                submit2_conversion_status_name(status));
            if (status == Submit2ConversionStatus::kOk) {
                auto submit_count = plan.submit_count();
                call_submit2(entry_points, queue, submit_count,
                             submit_count == 0u ? nullptr : &plan.submit_info2,
                             request.fence);
                return QueueSubmitVariant::kSubmit2;
            }
            // kUnsupportedPNext: the caller chained state that VkSubmitInfo2
            // cannot express (protected submit, device group, keyed mutex, ...).
            // Preserving it is only possible through the legacy call.
        }
    }
    auto legacy_info = make_legacy_submit_info(request);
    auto legacy_count = legacy_submit_count(request);
    VK_CHECK_RESULT(vkQueueSubmit(
        queue, legacy_count, legacy_count == 0u ? nullptr : &legacy_info,
        request.fence));
    return QueueSubmitVariant::kLegacy;
}

}// namespace lc::vk::detail