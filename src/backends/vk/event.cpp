#include "device.h"
#include "event.h"
#include "stream.h"
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
void Event::update_fence(uint64_t value) {
    std::lock_guard lck(eventMtx);
    lastFence = std::max(lastFence, value);
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
#ifndef NDEBUG
    uint64_t old_val = signaledEvent.load(std::memory_order_relaxed);
    while (fence > old_val &&
           !signaledEvent.compare_exchange_weak(
               old_val, fence,
               std::memory_order_release,
               std::memory_order_relaxed)) {
        LUISA_INTRIN_PAUSE();
    }
#endif
}
void Event::signal_sparse(Stream &stream, uint64_t const *value_ptr, VkBindSparseInfo *sparse_info, VkTimelineSemaphoreSubmitInfo *timeline_ptr) {
    {
        std::lock_guard lck(eventMtx);
        lastFence = std::max(lastFence, *value_ptr);
    }
    *timeline_ptr = get_timeline_submit(value_ptr);
    timeline_ptr->pNext = sparse_info->pNext;
    sparse_info->pNext = timeline_ptr;
    sparse_info->waitSemaphoreCount = 0;
    sparse_info->pWaitSemaphores = nullptr;
    sparse_info->signalSemaphoreCount = 1;
    sparse_info->pSignalSemaphores = &_semaphore;
}
void Event::signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer) {
    {
        std::lock_guard lck(eventMtx);
        lastFence = std::max(lastFence, value);
    }
    if (device()->config_ext() && device()->config_ext()->signal_semaphore(stream.queue(), _semaphore, value))
        return;
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
    stream.queue_mtx().lock();
    VK_CHECK_RESULT(vkQueueSubmit(stream.queue(), 1, &info1, VK_NULL_HANDLE));
    stream.queue_mtx().unlock();
    mark_signal_fence(value);
}
void Event::wait(Stream &stream, uint64_t value) {
#ifndef NDEBUG
    auto evt_value = signaledEvent.load();
    if (evt_value < value)
        LUISA_ERROR("Waiting for fence {} greater than last signaled-fence {}", value, evt_value);
#endif
    if (device()->config_ext() && device()->config_ext()->wait_semaphore(stream.queue(), _semaphore, value))
        return;
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
    stream.queue_mtx().lock();
    VK_CHECK_RESULT(vkQueueSubmit(stream.queue(), 1, &info1, VK_NULL_HANDLE));
    stream.queue_mtx().unlock();
}
void Event::host_wait(uint64_t value) {
    if (device()->config_ext() && device()->config_ext()->sync_semaphore(_semaphore, value))
        return;
    VkSemaphoreWaitInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &_semaphore,
        .pValues = &value};
    VK_CHECK_RESULT(vkWaitSemaphores(device()->logic_device(), &info, std::numeric_limits<uint64_t>::max()));
}
void Event::notify(uint64_t value) {
    {
        std::lock_guard lck(eventMtx);
        finishedEvent = std::max<uint64_t>(finishedEvent, value);
    }
}
void Event::sync(uint64_t value) {
    while (finishedEvent < value) {
        std::this_thread::yield();
    }
}

Event::~Event() {
    vkDestroySemaphore(device()->logic_device(), _semaphore, Device::alloc_callbacks());
}
}// namespace lc::vk