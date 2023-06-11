#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class UploadBuffer : public Buffer {
    AllocatedBuffer _res;

public:
    UploadBuffer(Device *device, size_t size_bytes);
    ~UploadBuffer();
    void copy_from(void const *data, size_t offset, size_t size);
    VkBuffer vk_buffer() const override { return _res.buffer; }
};
}// namespace lc::vk
