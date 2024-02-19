#pragma once
#include "resource.h"
namespace lc::vk {
class Stream;
class Event : public Resource {
    friend class Stream;
    VkSemaphore _semaphore;
    mutable std::mutex eventMtx;
    mutable std::condition_variable cv;
    mutable uint64 finishedEvent = 0;
    mutable uint64 lastFence = 0;
    void update_fence(uint64_t value);
    void signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void wait(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void host_wait(uint64_t value);
    void notify(uint64_t value);
public:
    [[nodiscard]] auto semaphore() const { return _semaphore; }
    [[nodiscard]] auto last_fence() const { return lastFence; }
    [[nodiscard]] bool is_complete(uint64_t fence) const {
        std::lock_guard lck{eventMtx};
        return finishedEvent >= fence;
    }
    void sync(uint64_t value);
    Event(Device *device);
    ~Event();
};
}// namespace lc::vk
