#pragma once
#include "resource.h"
#include "event.h"
#include <vulkan/vulkan.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/vstl/lockfree_array_queue.h>
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
class Stream : public Resource {
    struct SignalEvt {
        Event *evt;
        uint64_t value;
    };
    using Callbacks = luisa::vector<luisa::move_only_function<void()>>;
    using AsyncCmd = vstd::variant<
        Callbacks,
        CommandBuffer,
        SignalEvt>;
    Event _evt;
    VkCommandPool _pool;
    VkQueue _queue;
    std::atomic_bool _enabled{true};
    std::thread _thd;
    std::condition_variable _cv;
    std::mutex _mtx;
    vstd::LockFreeArrayQueue<AsyncCmd> _exec;

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
    void signal(Event* event, uint64_t value);
    void wait(Event* event, uint64_t value);
};

}// namespace lc::vk
