#pragma once

#include <volk.h>
#include <luisa/core/spin_mutex.h>
#include "resource.h"

namespace lc::vk {

class Stream;

class Event : public Resource {

    friend class Stream;
    VkSemaphore _semaphore{};
#ifndef NDEBUG
    mutable std::atomic_uint64_t signaledEvent = 0;
#endif
    mutable std::atomic_uint64_t finishedEvent = 0;
    mutable luisa::spin_mutex eventMtx;
    mutable uint64_t lastFence = 0;
    void update_fence(uint64_t value);
    void signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void signal_sparse(Stream &stream, uint64_t const *value_ptr, VkBindSparseInfo *sparse_info, VkTimelineSemaphoreSubmitInfo *timeline_ptr);
    void wait(Stream &stream, uint64_t value);
    void host_wait(uint64_t value);
    void notify(uint64_t value);

public:
    static VkTimelineSemaphoreSubmitInfo get_timeline_submit(uint64_t const *value_ptr);
    [[nodiscard]] auto semaphore() const { return _semaphore; }
    [[nodiscard]] auto last_fence() const { return lastFence; }
    [[nodiscard]] bool is_complete(uint64_t fence) const {
        std::lock_guard lck{eventMtx};
        return finishedEvent >= fence;
    }
    void mark_signal_fence(uint64_t fence);
    void sync(uint64_t value);
    Event(Device *device);
    ~Event();
};

}// namespace lc::vk
