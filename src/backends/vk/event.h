#pragma once

#include <atomic>
#include <mutex>

#include <volk.h>
#include <luisa/core/spin_mutex.h>
#include "resource.h"

namespace lc::vk {

class Stream;

class Event : public Resource {

    friend class Stream;
    VkSemaphore _semaphore{};
    mutable std::atomic_uint64_t _signaled_event = 0;
    mutable std::atomic_uint64_t _completed_gpu_event = 0;
    mutable std::atomic_uint64_t _finished_event = 0;
    mutable std::mutex _submission_mtx;
    mutable luisa::spin_mutex _event_mtx;
    mutable std::atomic_uint64_t _last_fence = 0;
    void _update_fence(uint64_t value);
    void _signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void _signal_sparse(
        uint64_t const *wait_value_ptr,
        uint64_t const *signal_value_ptr,
        VkBindSparseInfo *sparse_info,
        VkTimelineSemaphoreSubmitInfo *timeline_ptr);
    void _wait(Stream &stream, uint64_t value);
    void _host_wait(uint64_t value);
    void _notify(uint64_t value);
    void _mark_gpu_completion(uint64_t value) const noexcept;

public:
    static VkTimelineSemaphoreSubmitInfo get_timeline_submit(uint64_t const *value_ptr);
    [[nodiscard]] auto semaphore() const { return _semaphore; }
    [[nodiscard]] auto last_fence() const noexcept {
        return _last_fence.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto last_signaled_fence() const noexcept {
        return _signaled_event.load(std::memory_order_acquire);
    }
    [[nodiscard]] auto known_completed_gpu_fence() const noexcept {
        return _completed_gpu_event.load(std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t current_gpu_value() const;
    [[nodiscard]] bool is_complete(uint64_t fence) const {
        std::lock_guard lck{_event_mtx};
        return _finished_event >= fence;
    }
    void mark_signal_fence(uint64_t fence);
    void sync(uint64_t value);
    Event(Device *device);
    ~Event();
};

}// namespace lc::vk
