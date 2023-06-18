#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class DefaultBuffer : public Buffer {
    AllocatedBuffer _res;

public:
    DefaultBuffer(Device *device, size_t size_bytes);
    ~DefaultBuffer();
    VkBuffer vk_buffer() const override { return _res.buffer; }
};
}// namespace lc::vk
