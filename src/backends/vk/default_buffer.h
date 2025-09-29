#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class DefaultBuffer : public Buffer {
    VkBuffer _buffer{};
    union {
        VmaAllocation _allocation;
        VkDeviceMemory _allocated_memory;
    };
    bool _external_allocation{false};

public:
    DefaultBuffer(Device *device, size_t size_bytes, bool used_as_accel = false, VkBufferUsageFlagBits extra_bit = (VkBufferUsageFlagBits)0);
    DefaultBuffer(Device *device, VkBuffer vk_buffer, VkDeviceMemory memory, size_t size_bytes);
    DefaultBuffer(DefaultBuffer &&rhs);
    ~DefaultBuffer();
    VkBuffer vk_buffer() const override { return _buffer; }
    VkDeviceMemory external_device_memory() const { return _allocated_memory; }
    bool is_external_allocation() const { return _external_allocation; }
};
}// namespace lc::vk
