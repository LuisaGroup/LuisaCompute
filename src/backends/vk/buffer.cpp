#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include "device.h"
#include "log.h"
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
ReadbackBuffer::ReadbackBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  AccessType::ReadBack)} {
}
ReadbackBuffer::~ReadbackBuffer() {
    device()->allocator().destroy_buffer(_res);
}
void UploadBuffer::copy_from(void const *data, size_t offset, size_t size) {
    void *mapped_ptr;
    VK_CHECK_RESULT(vmaMapMemory(
        device()->allocator().allocator(),
        _res.allocation,
        &mapped_ptr));
    memcpy(reinterpret_cast<std::byte *>(mapped_ptr) + offset, data, size);
    vmaFlushAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        offset, size);
    vmaUnmapMemory(
        device()->allocator().allocator(),
        _res.allocation);
}
void ReadbackBuffer::copy_to(void *data, size_t offset, size_t size) {
    void *mapped_ptr;
    VK_CHECK_RESULT(vmaMapMemory(
        device()->allocator().allocator(),
        _res.allocation,
        &mapped_ptr));
    memcpy(data, reinterpret_cast<std::byte *>(mapped_ptr) + offset, size);
    vmaFlushAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        offset, size);
    vmaUnmapMemory(
        device()->allocator().allocator(),
        _res.allocation);
}
DefaultBuffer::DefaultBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  static_cast<VkBufferUsageFlagBits>(
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR),
                  AccessType::None)} {
}
DefaultBuffer::~DefaultBuffer() {
    device()->allocator().destroy_buffer(_res);
}
}// namespace lc::vk
