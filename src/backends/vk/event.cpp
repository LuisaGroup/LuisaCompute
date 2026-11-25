#include "device.h"
#include "event.h"
#include "stream.h"
#include "submit_profile.h"
#include "queue_submit_plan.h"
#include "timeline_semaphore_plan.h"
#include "log.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace lc::vk {
namespace {
// Phase 3B (opt C): how long `Event::sync` polls before it blocks on the counter.
// A microsecond budget converted into a `LUISA_INTRIN_PAUSE` loop bound (~100
// pauses per microsecond on this class of CPU). The value is a latency knob, never
// a correctness one: the loop exits on the counter in both phases, and a value that
// changed before the blocking wait returns from it immediately.
[[nodiscard]] uint32_t sync_pause_iterations() noexcept {
    static auto const iterations = [] {
        // Phase 3B (opt C) measurement: blocking costs 150-200 us per decode token on
        // this host. `example_qwen --version dsl-q8 --synthetic-prompt-len 128`
        // (interleaved A/B, 2 runs per arm) reported 10.18/10.29 ms/token with a 500 us
        // budget that really slept during the ~9 ms GPU wait, and 10.00/10.01 ms/token
        // with the budget raised past the wait. The wakeup of a blocked thread is a
        // kernel transition (WaitOnAddress), which is far more expensive than the
        // microsecond the poll costs, so the default budget is 20 ms: it still ends in a
        // blocking wait for a pathological multi-second stall, but a decode token -
        // and every per-command `commit()+synchronize()` - is served by polling.
        auto us = uint64_t{20'000u};
        if (auto e = std::getenv("LUISA_VK_SYNC_SPIN_US");
            e != nullptr && e[0] != '\0') {
            auto parsed = std::strtoull(e, nullptr, 10);
            if (parsed != 0ull) { us = std::min<uint64_t>(parsed, 100'000ull); }
        }
        return static_cast<uint32_t>(us * 100ull);
    }();
    return iterations;
}
}// namespace
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
bool Event::gpu_completed(uint64_t value) const {
    if (known_completed_gpu_fence() >= value) { return true; }
    // The watermark is only advanced by a host wait or an earlier query, so it
    // can lag a device that has already finished this value. Ask the driver once
    // instead of blocking the caller on the stream's retirement thread.
    return current_gpu_value() >= value;
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
    // The submission in the backend's legacy vocabulary: one optional command
    // buffer, no wait, this event's timeline signal. `detail::queue_submit`
    // converts it to vkQueueSubmit2 on a sync2-capable device and issues the
    // identical legacy vkQueueSubmit otherwise (plan P2.1).
    detail::QueueSubmitRequest request{};
    request.command_buffers = cmdbuffer;
    request.command_buffer_count = cmdbuffer ? 1u : 0u;
    request.signal_semaphores = &_semaphore;
    request.signal_count = 1u;
    request.fence = VK_NULL_HANDLE;
    request.pnext = &timelineInfo1;
    {
        std::lock_guard queue_lock{stream.queue_mtx()};
        auto config_ext = device()->config_ext();
        if (!(config_ext && config_ext->signal_semaphore(
                                stream.queue(), _semaphore, value))) {
            auto variant = detail::queue_submit(
                stream.queue(), device(), request);
            // Counter-only: one native submission of the dispatch path (F6),
            // split by entry point (plan P2.1). A callback-submitted signal is
            // not counted here because the backend issued no native call.
            submit_profile::record_queue_submit(
                variant == detail::QueueSubmitVariant::kSubmit2);
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

    // A wait-only submission, in the backend's legacy vocabulary: no command
    // buffer, no signal, this event's timeline wait with the ALL_COMMANDS dst
    // stage the backend has always used. `detail::queue_submit` carries that exact
    // mask into the 64-bit `VkSemaphoreSubmitInfo::stageMask` on a sync2 device and
    // issues the identical legacy vkQueueSubmit otherwise (plan P2.1).
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    detail::QueueSubmitRequest request{};
    request.wait_semaphores = &_semaphore;
    request.wait_stage_masks = &stage;
    request.wait_count = 1u;
    request.pnext = &timelineInfo1;
    std::lock_guard queue_lock{stream.queue_mtx()};
    auto config_ext = device()->config_ext();
    if (!(config_ext && config_ext->wait_semaphore(
                            stream.queue(), _semaphore, value))) {
        // This submission only waits on the timeline, so the host cannot observe
        // its completion through a timeline value or a later signal: it needs a
        // fence of its own. Retiring that fence before the semaphore is destroyed
        // is what keeps the batch from still referring to it at destroy time
        // (VUID-vkDestroySemaphore-semaphore-05149).
        _reap_wait_fences(false);
        VkFence fence{};
        if (!_free_wait_fences.empty()) {
            fence = _free_wait_fences.back();
            _free_wait_fences.pop_back();
            VK_CHECK_RESULT(vkResetFences(device()->logic_device(), 1u, &fence));
        } else {
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VK_CHECK_RESULT(vkCreateFence(
                device()->logic_device(), &fence_info, Device::alloc_callbacks(), &fence));
        }
        request.fence = fence;
        // Native submission of the wait-only path. It is deliberately not counted
        // as a `queue_submits` entry: that counter describes the dispatch path
        // (F6), not the timeline-only waits.
        (void)detail::queue_submit(stream.queue(), device(), request);
        _wait_fences.emplace_back(fence);
    }
}
void Event::_reap_wait_fences(bool wait_for_pending) noexcept {
    auto device_handle = device()->logic_device();
    if (wait_for_pending) {
        if (!_wait_fences.empty()) {
            VK_CHECK_RESULT(vkWaitForFences(
                device_handle, static_cast<uint32_t>(_wait_fences.size()),
                _wait_fences.data(), VK_TRUE,
                std::numeric_limits<uint64_t>::max()));
            for (auto fence : _wait_fences) {
                vkDestroyFence(device_handle, fence, Device::alloc_callbacks());
            }
            _wait_fences.clear();
        }
        for (auto fence : _free_wait_fences) {
            vkDestroyFence(device_handle, fence, Device::alloc_callbacks());
        }
        _free_wait_fences.clear();
        return;
    }
    // Poll before recording another pending wait, so a long-lived timeline that is
    // waited on many times keeps this list bounded by the waits still in flight.
    for (size_t i = 0u; i < _wait_fences.size();) {
        auto status = vkGetFenceStatus(device_handle, _wait_fences[i]);
        LUISA_ASSERT(
            status == VK_SUCCESS || status == VK_NOT_READY,
            "vkGetFenceStatus failed on a Vulkan event wait fence: {}.",
            static_cast<int>(status));
        if (status != VK_SUCCESS) {
            i++;
            continue;
        }
        auto fence = _wait_fences[i];
        if (_free_wait_fences.size() < kMaxPooledWaitFences) {
            _free_wait_fences.emplace_back(fence);
        } else {
            vkDestroyFence(device_handle, fence, Device::alloc_callbacks());
        }
        _wait_fences[i] = _wait_fences.back();
        _wait_fences.pop_back();
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
    // Phase 3B (opt C): wake every `sync()` that blocked after its polling budget.
    // The store above is what they observe; the notification is only the wakeup.
    _finished_event.notify_all();
}
void Event::sync(uint64_t value) {
    // Phase 3B (opt C): poll first (this is the micro-latency path: a per-command
    // `commit()+synchronize()` waits tens of microseconds), then block instead of
    // spinning a whole core through a multi-millisecond GPU wait. `yield()` in a loop
    // neither slept nor helped: it kept the caller and the retirement thread both
    // runnable while the host argmax/encode had work to do.
    auto const spin = sync_pause_iterations();
    for (auto i = 0u; i < spin; ++i) {
        if (_finished_event.load(std::memory_order_acquire) >= value) { return; }
        LUISA_INTRIN_PAUSE();
    }
    while (true) {
        auto const observed = _finished_event.load(std::memory_order_acquire);
        if (observed >= value) { return; }
        // Blocks while the counter still holds `observed`; a `_notify` that already
        // moved it makes this return immediately, so no wakeup can be lost.
        _finished_event.wait(observed, std::memory_order_acquire);
    }
}

Event::~Event() {
    sync(_last_fence);
    _host_wait(_signaled_event.load(std::memory_order_relaxed));
    // A batch that only waited on this timeline still refers to the semaphore until
    // it retires, so every queued wait submission has to be drained first: a
    // semaphore may only be destroyed once all submitted batches that refer to it
    // have completed execution (VUID-vkDestroySemaphore-semaphore-05149).
    _reap_wait_fences(true);
    vkDestroySemaphore(device()->logic_device(), _semaphore, Device::alloc_callbacks());
}
}// namespace lc::vk
