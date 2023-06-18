#include "command_buffer.h"
#include "device.h"
#include <luisa/core/logging.h>
#include "log.h"
namespace lc::vk {
CommandBuffer::CommandBuffer(Device *device, StreamTag tag)
    : Resource{device} {
    VkCommandPoolCreateInfo pool_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    switch (tag) {
        case StreamTag::GRAPHICS:
            pool_ci.queueFamilyIndex = device->graphics_queue_index();
            break;
        case StreamTag::COPY:
            pool_ci.queueFamilyIndex = device->copy_queue_index();
            break;
        case StreamTag::COMPUTE:
            pool_ci.queueFamilyIndex = device->compute_queue_index();
            break;
        default:
            LUISA_ASSERT(false, "Illegal stream tag.");
    }
    VK_CHECK_RESULT(vkCreateCommandPool(device->logic_device(), &pool_ci, nullptr, &_pool));
    VkCommandBufferAllocateInfo cb_ci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = _pool,
        .commandBufferCount = 1};
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device->logic_device(), &cb_ci, &_cmd));
}
CommandBuffer::~CommandBuffer() {
    vkFreeCommandBuffers(device()->logic_device(), _pool, 1, &_cmd);
    vkDestroyCommandPool(device()->logic_device(), _pool, nullptr);
}
void CommandBuffer::begin() {
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK_CHECK_RESULT(vkBeginCommandBuffer(_cmd, &bi));
}
void CommandBuffer::end() {
    VK_CHECK_RESULT(vkEndCommandBuffer(_cmd));
}
}// namespace lc::vk
