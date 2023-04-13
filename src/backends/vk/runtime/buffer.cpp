#include "upload_buffer.h"
#include "readback_buffer.h"
#include "device.h"
#include "../log.h"
namespace lc::vk {
UploadBuffer::UploadBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  AccessType::Upload)} {
}
UploadBuffer::~UploadBuffer() {
    device()->allocator().destroy_buffer(_res);
}
void UploadBuffer::copy_from(void const *data, size_t size) {
    void *mapped_ptr;
    VK_CHECK_RESULT(vmaMapMemory(
        device()->allocator().allocator(),
        _res.allocation,
        &mapped_ptr));
    memcpy(mapped_ptr, data, size);
    vmaUnmapMemory(
        device()->allocator().allocator(),
        _res.allocation);
}

}// namespace lc::vk