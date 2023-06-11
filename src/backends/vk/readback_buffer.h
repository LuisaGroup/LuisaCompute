#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class ReadbackBuffer : public Buffer {
    AllocatedBuffer _res;

public:
    ReadbackBuffer(Device *device, size_t size_bytes);
    ~ReadbackBuffer();
    void copy_to(void *data, size_t offset, size_t size);
    VkBuffer vk_buffer() const override { return _res.buffer; }
};
}// namespace lc::vk
