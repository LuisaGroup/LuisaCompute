#pragma once
#include "buffer.h"
#include "vk_allocator.h"
namespace lc::vk {
class UploadBuffer : public Buffer {
    AllocatedBuffer _res;
    void *_mapped_ptr{};
    mutable BufferFlusher _flusher;

public:
    UploadBuffer(Device *device, size_t size_bytes, AccessType access = AccessType::kUpload);
    ~UploadBuffer();
    void copy_from(void const *data, size_t offset, size_t size) const;
    VkBuffer vk_buffer() const override { return _res.buffer; }
    void *mapped_ptr() const { return _mapped_ptr; }
    bool flush_host() const override;
    void flush_range(size_t begin, size_t end) override;
};
}// namespace lc::vk
