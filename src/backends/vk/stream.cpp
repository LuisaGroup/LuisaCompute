#include "stream.h"
#include "device.h"
#include <luisa/core/logging.h>
#include "log.h"
namespace lc::vk {
Stream::Stream(Device *device, StreamTag tag)
    : Resource{device},
      _evt(device),
      _thd([this]() {
          while (_enabled) {
              while (auto p = _exec.pop()) {
                  p->visit(
                      [&]<typename T>(T const &t) {
                          if constexpr (std::is_same_v<T, Callbacks>) {
                              for (auto &i : t) {
                                  i();
                              }
                          } else if constexpr (std::is_same_v<T, SignalEvt>) {
                              t.evt->notify(t.value);
                          }
                      });
              }
              std::unique_lock lck{_mtx};
              while (_enabled && _exec.length() == 0) {
                  _cv.wait(lck);
              }
          }
      }) {
    VkCommandPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    switch (tag) {
        case StreamTag::GRAPHICS:
            pool_ci.queueFamilyIndex = device->graphics_queue_index();
            _queue = device->graphics_queue();
            break;
        case StreamTag::COPY:
            pool_ci.queueFamilyIndex = device->copy_queue_index();
            _queue = device->compute_queue();
            break;
        case StreamTag::COMPUTE:
            pool_ci.queueFamilyIndex = device->compute_queue_index();
            _queue = device->copy_queue();
            break;
        default:
            LUISA_ASSERT(false, "Illegal stream tag.");
    }
    VK_CHECK_RESULT(vkCreateCommandPool(device->logic_device(), &pool_ci, Device::alloc_callbacks(), &_pool));
}
Stream::~Stream() {
    {
        std::lock_guard lck{_mtx};
        _enabled = false;
    }
    _cv.notify_one();
    _thd.join();
    vkDestroyCommandPool(device()->logic_device(), _pool, Device::alloc_callbacks());
}
void Stream::dispatch(
    vstd::span<const luisa::unique_ptr<Command>> cmds,
    luisa::vector<luisa::move_only_function<void()>> &&callbacks,
    bool inqueue_limit) {

    if (cmds.empty() && callbacks.empty()) {
        return;
    }
    if (inqueue_limit) {
        if (_evt.last_fence() > 2) {
            _evt.sync(_evt.last_fence() - 2);
        }
    }
    auto fence = _evt.last_fence() + 1;
    if (!cmds.empty()) {
        CommandBuffer cmdbuffer{*this};
        auto cb = cmdbuffer.cmdbuffer();
        _evt.signal(*this, fence, &cb);
        _exec.push(std::move(cmdbuffer));
        _exec.push(SignalEvt{
            .evt = &_evt,
            .value = fence});
    }
    if (!callbacks.empty()) {
        _exec.push(std::move(callbacks));
    }
    _mtx.lock();
    _mtx.unlock();
    _cv.notify_one();
}
void Stream::sync() {
    _evt.sync(_evt.last_fence());
}
CommandBuffer::CommandBuffer(Stream &stream)
    : Resource(stream.device()),
      _pool(stream.pool()) {
    VkCommandBufferAllocateInfo cb_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = _pool,
        .commandBufferCount = 1};
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device()->logic_device(), &cb_ci, &_cmdbuffer));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK_RESULT(vkCreateFence(device()->logic_device(), &fence_info, Device::alloc_callbacks(), nullptr));
}
CommandBuffer::~CommandBuffer() {
    if (_cmdbuffer)
        vkFreeCommandBuffers(device()->logic_device(), _pool, 1, &_cmdbuffer);
}
void CommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK_CHECK_RESULT(vkBeginCommandBuffer(_cmdbuffer, &bi));
}
void CommandBuffer::end() {
    VK_CHECK_RESULT(vkEndCommandBuffer(_cmdbuffer));
}
CommandBuffer::CommandBuffer(CommandBuffer &&rhs)
    : Resource(std::move(rhs)),
      _pool(rhs._pool),
      _cmdbuffer(rhs._cmdbuffer) {
    rhs._cmdbuffer = nullptr;
}
void Stream::signal(Event *event, uint64_t value) {
    event->signal(*this, value);
    _exec.push(SignalEvt{event, value});
}
void Stream::wait(Event *event, uint64_t value) {
    event->wait(*this, value);
}

}// namespace lc::vk
