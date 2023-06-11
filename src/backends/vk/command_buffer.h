#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
#include <luisa/runtime/rhi/stream_tag.h>
namespace lc::vk {
using namespace luisa::compute;
class CommandBuffer : public Resource {
    VkCommandPool _pool;
    VkCommandBuffer _cmd;

public:
    auto pool() const { return _pool; }
    auto cmd() const { return _cmd; }
    CommandBuffer(Device *device, StreamTag tag);
    void begin();
    void end();
    ~CommandBuffer();
};
}// namespace lc::vk
