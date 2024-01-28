#pragma once
#include "resource.h"
#include "event.h"
#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include <vulkan/vulkan.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <luisa/vstl/stack_allocator.h>
namespace lc::vk {
class Event;
class Stream;
using namespace luisa::compute;
class CommandBuffer;
class CommandBuffer : public Resource {
    VkCommandPool _pool;
    VkCommandBuffer _cmdbuffer;
public:
    using Resource::operator bool;
    CommandBuffer(Stream &stream);
    CommandBuffer(CommandBuffer &&);
    ~CommandBuffer();
    [[nodiscard]] auto cmdbuffer() const { return _cmdbuffer; }
    void begin();
    void end();
};
namespace temp_buffer {
template<typename Pack>
class Visitor : public vstd::StackAllocatorVisitor {
public:
    Device* device;
    uint64 allocate(uint64 size) override;
    void deallocate(uint64 handle) override;
};
template<typename T>
class BufferAllocator {
    static constexpr size_t kLargeBufferSize = 65536ull;
    vstd::StackAllocator alloc;
    vstd::vector<vstd::unique_ptr<T>> largeBuffers;

public:
    Visitor<T> visitor;
    BufferView allocate(size_t size);
    BufferView allocate(size_t size, size_t align);
    void clear();
    BufferAllocator(size_t initCapacity);
    ~BufferAllocator();
};
}// namespace temp_buffer
class Stream : public Resource {
    struct SyncExt {
        Event *evt;
        uint64_t value;
    };
    struct NotifyEvt {
        Event *evt;
        uint64_t value;
    };
    using Callbacks = luisa::vector<luisa::move_only_function<void()>>;
    using AsyncCmd = vstd::variant<
        Callbacks,
        CommandBuffer,
        SyncExt,
        NotifyEvt>;
    Event _evt;
    VkCommandPool _pool;
    VkQueue _queue;
    std::atomic_bool _enabled{true};
    std::thread _thd;
    std::condition_variable _cv;
    std::mutex _mtx;
    vstd::LockFreeArrayQueue<AsyncCmd> _exec;
    temp_buffer::BufferAllocator<UploadBuffer> upload_alloc;
    temp_buffer::BufferAllocator<DefaultBuffer> default_alloc;
    temp_buffer::BufferAllocator<ReadbackBuffer> readback_alloc;

public:
    [[nodiscard]] auto queue() const { return _queue; }
    [[nodiscard]] auto pool() const { return _pool; }
    Stream(Device *device, StreamTag tag);
    ~Stream();
    void dispatch(
        vstd::span<const luisa::unique_ptr<Command>> cmds,
        Callbacks &&callbacks,
        bool inqueue_limit);
    void sync();
    void signal(Event *event, uint64_t value);
    void wait(Event *event, uint64_t value);
};

}// namespace lc::vk
