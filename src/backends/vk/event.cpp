#include "device.h"
#include "event.h"
#include "stream.h"
#include "submit_profile.h"
#include "queue_submit_plan.h"
#include "timeline_semaphore_plan.h"
#include "in_flight_depth.h"
#include "log.h"
#include <algorithm>
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
        return detail::parse_spin_pause_iterations(
            std::getenv("LUISA_VK_SYNC_SPIN_US"), 20'000u);
    }();
    return iterations;
}
}// namespace
Event::Event(Device *device)
    : Resource(device),
      _wait_token(std::make_shared<EventWaitToken>()) {
    _wait_token->evt = this;
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
VkFence Event::_acquire_wait_fence_locked() {
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
    return fence;
}
void Event::_signal_batch(
    Stream &stream,
    uint64_t value,
    VkCommandBuffer *cmdbuffer,
    vstd::span<const EventWaitEntry> waits,
    vstd::span<const EventSignalEntry> extra_signals) {
    if (waits.empty() && extra_signals.empty()) {
        _signal(stream, value, cmdbuffer);
        return;
    }

    luisa::fixed_vector<EventSignalEntry, 8> uniq_signals;
    uniq_signals.emplace_back(EventSignalEntry{this, value});
    for (auto const &s : extra_signals) {
        if (s.evt == nullptr) { continue; }
        LUISA_ASSERT(s.evt != this, "Duplicate internal stream event signal.");
        bool merged = false;
        for (auto &existing : uniq_signals) {
            if (existing.evt == s.evt) {
                if (s.value > existing.value) {
                    existing.value = s.value;
                }
                merged = true;
                break;
            }
        }
        if (!merged) {
            uniq_signals.emplace_back(s);
        }
    }

    auto is_signal_evt = [&](Event const *e) noexcept {
        for (auto const &s : uniq_signals) {
            if (s.evt == e) { return true; }
        }
        return false;
    };

    // Flush any earlier pending waits so at most the tail wait is attached to
    // the combined VkSubmitInfo (each wait carries a lifetime VkFence on its Event).
    if (waits.size() > 1u) {
        for (size_t i = 0u; i + 1u < waits.size(); ++i) {
            auto const &w = waits[i];
            if (!w.token) { continue; }
            std::lock_guard token_lock{w.token->mtx};
            if (w.token->evt != nullptr) {
                w.token->evt->_wait(stream, w.value);
            }
        }
    }

    std::unique_lock<std::mutex> tail_token_lock;
    Event *tail_wait_evt = nullptr;
    uint64_t tail_wait_value = 0u;
    VkSemaphore tail_wait_semaphore = VK_NULL_HANDLE;
    VkFence wait_fence = VK_NULL_HANDLE;
    if (!waits.empty() && waits.back().token) {
        auto const &tw = waits.back();
        tail_token_lock = std::unique_lock<std::mutex>{tw.token->mtx};
        tail_wait_evt = tw.token->evt;
        tail_wait_value = tw.value;
        if (tail_wait_evt == nullptr) {
            tail_token_lock.unlock();
        } else if (is_signal_evt(tail_wait_evt)) {
            tail_wait_evt->_wait(stream, tail_wait_value);
            tail_wait_evt = nullptr;
            tail_token_lock.unlock();
        } else {
            std::lock_guard wait_sub_lock{tail_wait_evt->_submission_mtx};
            auto tracked_signal = tail_wait_evt->last_signaled_fence();
            auto current_value = tail_wait_evt->known_completed_gpu_fence();
            auto max_value_difference =
                tail_wait_evt->device()->max_timeline_semaphore_value_difference();
            auto value_plan = detail::plan_timeline_semaphore_wait(
                current_value, tracked_signal, tail_wait_value,
                max_value_difference);
            if (!value_plan &&
                value_plan.status == detail::TimelineSemaphoreValueStatus::
                                         TRACKED_SIGNAL_RANGE_EXCEEDED) {
                current_value = tail_wait_evt->current_gpu_value();
                value_plan = detail::plan_timeline_semaphore_wait(
                    current_value, tracked_signal, tail_wait_value,
                    max_value_difference);
            }
            LUISA_ASSERT(
                static_cast<bool>(value_plan),
                "Invalid Vulkan timeline-semaphore wait {} (current {}, tracked "
                "signal {}, max difference {}): {}.",
                tail_wait_value, current_value, tracked_signal, max_value_difference,
                detail::timeline_semaphore_value_status_name(value_plan.status));
            if (value_plan.already_satisfied) {
                tail_wait_evt = nullptr;
                tail_token_lock.unlock();
            } else {
                tail_wait_semaphore = tail_wait_evt->semaphore();
                wait_fence = tail_wait_evt->_acquire_wait_fence_locked();
            }
        }
    }

    // Lock signal _submission_mtxs in ascending pointer order to avoid deadlock.
    luisa::fixed_vector<Event *, 8> lock_order;
    lock_order.reserve(uniq_signals.size());
    for (auto const &s : uniq_signals) {
        lock_order.emplace_back(s.evt);
    }
    std::sort(lock_order.begin(), lock_order.end());
    luisa::fixed_vector<std::unique_lock<std::mutex>, 8> sig_locks;
    sig_locks.reserve(lock_order.size());
    for (auto *e : lock_order) {
        sig_locks.emplace_back(e->_submission_mtx);
    }

    luisa::fixed_vector<VkSemaphore, 8> sig_sems;
    luisa::fixed_vector<uint64_t, 8> sig_vals;
    sig_sems.reserve(uniq_signals.size());
    sig_vals.reserve(uniq_signals.size());
    for (auto const &s : uniq_signals) {
        auto *se = s.evt;
        auto tracked_signal = se->last_signaled_fence();
        auto current_value = se->known_completed_gpu_fence();
        auto max_value_difference =
            se->device()->max_timeline_semaphore_value_difference();
        auto value_plan = detail::plan_timeline_semaphore_signal(
            current_value, tracked_signal, s.value,
            max_value_difference);
        if (!value_plan &&
            (value_plan.status == detail::TimelineSemaphoreValueStatus::
                                      TRACKED_SIGNAL_RANGE_EXCEEDED ||
             value_plan.status == detail::TimelineSemaphoreValueStatus::
                                      MAX_VALUE_DIFFERENCE_EXCEEDED)) {
            current_value = se->current_gpu_value();
            value_plan = detail::plan_timeline_semaphore_signal(
                current_value, tracked_signal, s.value,
                max_value_difference);
        }
        LUISA_ASSERT(
            static_cast<bool>(value_plan),
            "Invalid Vulkan timeline-semaphore signal {} (current {}, tracked "
            "signal {}, max difference {}): {}.",
            s.value, current_value, tracked_signal, max_value_difference,
            detail::timeline_semaphore_value_status_name(value_plan.status));
        sig_sems.emplace_back(se->semaphore());
        sig_vals.emplace_back(s.value);
    }

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline_info.waitSemaphoreValueCount = tail_wait_evt != nullptr ? 1u : 0u;
    timeline_info.pWaitSemaphoreValues = tail_wait_evt != nullptr ? &tail_wait_value : nullptr;
    timeline_info.signalSemaphoreValueCount = static_cast<uint32_t>(sig_vals.size());
    timeline_info.pSignalSemaphoreValues = sig_vals.data();

    detail::QueueSubmitRequest request{};
    request.command_buffers = cmdbuffer;
    request.command_buffer_count = cmdbuffer ? 1u : 0u;
    request.wait_semaphores = tail_wait_evt != nullptr ? &tail_wait_semaphore : nullptr;
    request.wait_stage_masks = tail_wait_evt != nullptr ? &wait_stage : nullptr;
    request.wait_count = tail_wait_evt != nullptr ? 1u : 0u;
    request.signal_semaphores = sig_sems.data();
    request.signal_count = static_cast<uint32_t>(sig_sems.size());
    request.fence = wait_fence;
    request.pnext = &timeline_info;
    {
        std::lock_guard queue_lock{stream.queue_mtx()};
        auto variant = detail::queue_submit(
            stream.queue(), device(), request);
        submit_profile::record_queue_submit(
            variant == detail::QueueSubmitVariant::kSubmit2);
    }
    for (auto const &s : uniq_signals) {
        s.evt->_update_fence(s.value);
        s.evt->mark_signal_fence(s.value);
    }
    sig_locks.clear();

    if (wait_fence != VK_NULL_HANDLE) {
        std::lock_guard wait_sub_lock{tail_wait_evt->_submission_mtx};
        tail_wait_evt->_wait_fences.emplace_back(wait_fence);
    }
    if (tail_token_lock.owns_lock()) {
        tail_token_lock.unlock();
    }
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
        auto fence = _acquire_wait_fence_locked();
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
        // Phase 3B (opt C): wake every `sync()` that blocked after its polling
        // budget. The store above is what they observe; the notification is only
        // the wakeup. It stays inside the lock: a spinner that observes the store
        // may return from `sync()` and destroy this event immediately, so touching
        // the counter after the lock is dropped would be a use-after-free.
        _finished_event.notify_all();
    }
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
    {
        // Barrier against any in-flight `_notify()` that published `_last_fence`
        // and is still executing `_finished_event.notify_all()` under `_event_mtx`.
        std::lock_guard lck(_event_mtx);
    }
    _host_wait(_signaled_event.load(std::memory_order_relaxed));
    if (_wait_token) {
        std::lock_guard token_lck{_wait_token->mtx};
        _wait_token->evt = nullptr;
    }
    // A batch that only waited on this timeline still refers to the semaphore until
    // it retires, so every queued wait submission has to be drained first: a
    // semaphore may only be destroyed once all submitted batches that refer to it
    // have completed execution (VUID-vkDestroySemaphore-semaphore-05149).
    _reap_wait_fences(true);
    vkDestroySemaphore(device()->logic_device(), _semaphore, Device::alloc_callbacks());
}
}// namespace lc::vk
