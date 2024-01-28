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
    [[nodiscard]] auto semaphore() const { return _semaphore; }
    [[nodiscard]] auto last_fence() const { return lastFence; }
    Event(Device *device);
    ~Event();
    void signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void wait(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void sync(uint64_t value);
    void notify(uint64_t value);
};
}// namespace lc::vk
