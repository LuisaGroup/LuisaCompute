#include "device.h"
#include "event.h"
#include "stream.h"
#include "timeline_semaphore_plan.h"
#include "log.h"
namespace lc::vk {
Event::Event(Device *device)
    : Resource(device) {
    VkSemaphoreTypeCreateInfo timelineCreateInfo;
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.pNext = NULL;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo createInfo;
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &timelineCreateInfo;
    createInfo.flags = 0;

    VK_CHECK_RESULT(vkCreateSemaphore(device->logic_device(), &createInfo, Device::alloc_callbacks(), &_semaphore));
}
void Event::_update_fence(uint64_t value) {
    auto old_value = _last_fence.load(std::memory_order_relaxed);
    while (value > old_value &&
           !_last_fence.compare_exchange_weak(
               old_value, value,
               std::memory_order_release,
               std::memory_order_relaxed)) {
        LUISA_INTRIN_PAUSE();
    }
}
VkTimelineSemaphoreSubmitInfo Event::get_timeline_submit(uint64_t const *value_ptr) {
    VkTimelineSemaphoreSubmitInfo timelineInfo1{};
    timelineInfo1.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo1.pNext = nullptr;
    timelineInfo1.waitSemaphoreValueCount = 0;
    timelineInfo1.pWaitSemaphoreValues = nullptr;
    timelineInfo1.signalSemaphoreValueCount = 1;
    timelineInfo1.pSignalSemaphoreValues = value_ptr;

    return timelineInfo1;
}
void Event::mark_signal_fence(uint64_t fence) {
    uint64_t old_val = _signaled_event.load(std::memory_order_relaxed);
    while (fence > old_val &&
           !_signaled_event.compare_exchange_weak(
               old_val, fence,
               std::memory_order_release,
               std::memory_order_relaxed)) {
        LUISA_INTRIN_PAUSE();
    }
}
void Event::_mark_gpu_completion(uint64_t value) const noexcept {
    auto old_value =
        _completed_gpu_event.load(std::memory_order_relaxed);
    while (value > old_value &&
           !_completed_gpu_event.compare_exchange_weak(
               old_value, value,
               std::memory_order_release,
               std::memory_order_relaxed)) {
        LUISA_INTRIN_PAUSE();
    }
}
uint64_t Event::current_gpu_value() const {
    uint64_t value{};
    VK_CHECK_RESULT(vkGetSemaphoreCounterValue(
        device()->logic_device(), _semaphore, &value));
    _mark_gpu_completion(value);
    return value;
}
void Event::_signal_sparse(
    uint64_t const *wait_value_ptr,
    uint64_t const *signal_value_ptr,
    VkBindSparseInfo *sparse_info,
    VkTimelineSemaphoreSubmitInfo *timeline_ptr) {
    _update_fence(*signal_value_ptr);
    *timeline_ptr = get_timeline_submit(signal_value_ptr);
    timeline_ptr->waitSemaphoreValueCount =
        wait_value_ptr == nullptr ? 0u : 1u;
    timeline_ptr->pWaitSemaphoreValues = wait_value_ptr;
    timeline_ptr->pNext = sparse_info->pNext;
    sparse_info->pNext = timeline_ptr;
    sparse_info->waitSemaphoreCount =
        wait_value_ptr == nullptr ? 0u : 1u;
    sparse_info->pWaitSemaphores =
        wait_value_ptr == nullptr ? nullptr : &_semaphore;
    sparse_info->signalSemaphoreCount = 1;
    sparse_info->pSignalSemaphores = &_semaphore;
}
void Event::_signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer) {
    std::lock_guard submission_lock{_submission_mtx};
    auto tracked_signal = last_signaled_fence();
    auto current_value = known_completed_gpu_fence();
    auto max_value_difference =
        device()->max_timeline_semaphore_value_difference();
    auto value_plan = detail::plan_timeline_semaphore_signal(
        current_value, tracked_signal, value,
        max_value_difference);
    // The completion watermark is deliberately conservative. Query Vulkan
    // only when that lower bound would reject an otherwise valid large jump;
    // the ordinary dispatch path remains free of driver round-trips.
    if (!value_plan &&
        (value_plan.status == detail::TimelineSemaphoreValueStatus::
                                  TRACKED_SIGNAL_RANGE_EXCEEDED ||
         value_plan.status == detail::TimelineSemaphoreValueStatus::
                                  MAX_VALUE_DIFFERENCE_EXCEEDED)) {
        current_value = current_gpu_value();
        value_plan = detail::plan_timeline_semaphore_signal(
            current_value, tracked_signal, value,
            max_value_difference);
    }
    LUISA_ASSERT(
        static_cast<bool>(value_plan),
        "Invalid Vulkan timeline-semaphore signal {} (current {}, tracked "
        "signal {}, max difference {}): {}.",
        value, current_value, tracked_signal, max_value_difference,
        detail::timeline_semaphore_value_status_name(value_plan.status));
    auto timelineInfo1 = get_timeline_submit(&value);
    VkSubmitInfo info1{};
    info1.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info1.pNext = &timelineInfo1;
    info1.waitSemaphoreCount = 0;
    info1.pWaitSemaphores = nullptr;
    info1.signalSemaphoreCount = 1;
    info1.pSignalSemaphores = &_semaphore;
    // ... Enqueue initial device work here.
    info1.commandBufferCount = cmdbuffer ? 1 : 0;
    info1.pCommandBuffers = cmdbuffer;
    {
        std::lock_guard queue_lock{stream.queue_mtx()};
        auto config_ext = device()->config_ext();
        if (!(config_ext && config_ext->signal_semaphore(
                                stream.queue(), _semaphore, value))) {
            VK_CHECK_RESULT(vkQueueSubmit(
                stream.queue(), 1, &info1, VK_NULL_HANDLE));
        }
    }
    _update_fence(value);
    mark_signal_fence(value);
}
void Event::_wait(Stream &stream, uint64_t value) {
    std::lock_guard submission_lock{_submission_mtx};
    auto tracked_signal = last_signaled_fence();
    auto current_value = known_completed_gpu_fence();
    auto max_value_difference =
        device()->max_timeline_semaphore_value_difference();
    auto value_plan = detail::plan_timeline_semaphore_wait(
        current_value, tracked_signal, value,
        max_value_difference);
    if (!value_plan &&
        value_plan.status == detail::TimelineSemaphoreValueStatus::
                                 TRACKED_SIGNAL_RANGE_EXCEEDED) {
        current_value = current_gpu_value();
        value_plan = detail::plan_timeline_semaphore_wait(
            current_value, tracked_signal, value,
            max_value_difference);
    }
    LUISA_ASSERT(
        static_cast<bool>(value_plan),
        "Invalid Vulkan timeline-semaphore wait {} (current {}, tracked "
        "signal {}, max difference {}): {}.",
        value, current_value, tracked_signal, max_value_difference,
        detail::timeline_semaphore_value_status_name(value_plan.status));
    if (value_plan.already_satisfied) { return; }
    VkTimelineSemaphoreSubmitInfo timelineInfo1{};
    timelineInfo1.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo1.pNext = nullptr;
    timelineInfo1.waitSemaphoreValueCount = 1;
    timelineInfo1.pWaitSemaphoreValues = &value;
    timelineInfo1.signalSemaphoreValueCount = 0;
    timelineInfo1.pSignalSemaphoreValues = nullptr;

    VkSubmitInfo info1{};
    info1.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info1.pNext = &timelineInfo1;
    info1.waitSemaphoreCount = 1;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    info1.pWaitDstStageMask = &stage;
    info1.pWaitSemaphores = &_semaphore;
    info1.signalSemaphoreCount = 0;
    info1.pSignalSemaphores = nullptr;
    // ... Enqueue initial device work here.
    info1.commandBufferCount = 0;
    info1.pCommandBuffers = nullptr;
    std::lock_guard queue_lock{stream.queue_mtx()};
    auto config_ext = device()->config_ext();
    if (!(config_ext && config_ext->wait_semaphore(
                            stream.queue(), _semaphore, value))) {
        VK_CHECK_RESULT(vkQueueSubmit(
            stream.queue(), 1, &info1, VK_NULL_HANDLE));
    }
}
void Event::_host_wait(uint64_t value) {
    if (device()->config_ext() && device()->config_ext()->sync_semaphore(_semaphore, value)) {
        _mark_gpu_completion(value);
        return;
    }
    VkSemaphoreWaitInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &_semaphore,
        .pValues = &value};
    VK_CHECK_RESULT(vkWaitSemaphores(device()->logic_device(), &info, std::numeric_limits<uint64_t>::max()));
    _mark_gpu_completion(value);
}
void Event::_notify(uint64_t value) {
    {
        std::lock_guard lck(_event_mtx);
        _finished_event = std::max<uint64_t>(_finished_event, value);
    }
}
void Event::sync(uint64_t value) {
    while (_finished_event < value) {
        std::this_thread::yield();
    }
}

Event::~Event() {
    sync(_last_fence);
    _host_wait(_signaled_event.load(std::memory_order_relaxed));
    vkDestroySemaphore(device()->logic_device(), _semaphore, Device::alloc_callbacks());
}
}// namespace lc::vk
