#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include "sparse_buffer.h"
#include "device.h"
#include "log.h"
namespace lc::vk {
UploadBuffer::UploadBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  (VkBufferUsageFlagBits)((uint)VK_BUFFER_USAGE_TRANSFER_SRC_BIT | (uint)VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                  AccessType::Upload)} {
    VK_CHECK_RESULT(vmaMapMemory(
        device->allocator().allocator(),
        _res.allocation,
        &_mapped_ptr));
}
UploadBuffer::~UploadBuffer() {
    if (_mapped_ptr) {
        vmaUnmapMemory(
            device()->allocator().allocator(),
            _res.allocation);
    }
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
    VK_CHECK_RESULT(vmaMapMemory(
        device->allocator().allocator(),
        _res.allocation,
        &_mapped_ptr));
}
ReadbackBuffer::~ReadbackBuffer() {
    if (_mapped_ptr) {
        vmaUnmapMemory(
            device()->allocator().allocator(),
            _res.allocation);
    }
    device()->allocator().destroy_buffer(_res);
}
void UploadBuffer::copy_from(void const *data, size_t offset, size_t size) const {
    memcpy(reinterpret_cast<std::byte *>(_mapped_ptr) + offset, data, size);
    vmaFlushAllocation(
        device()->allocator().allocator(),
        static_cast<VmaAllocation>(_res.allocation),
        offset, size);
}
void ReadbackBuffer::copy_to(void *data, size_t offset, size_t size) const {
    memcpy(data, reinterpret_cast<std::byte *>(_mapped_ptr) + offset, size);
    vmaFlushAllocation(
        device()->allocator().allocator(),
        static_cast<VmaAllocation>(_res.allocation),
        offset, size);
}
bool UploadBuffer::flush_host() const {
    _flusher.flush(device(), _res.allocation);
    return true;
}
bool ReadbackBuffer::flush_host() const {
    _flusher.flush(device(), _res.allocation);
    return true;
}

DefaultBuffer::DefaultBuffer(Device *device, VkBuffer vk_buffer, VkDeviceMemory memory, size_t size_bytes) 
: Buffer{device, size_bytes} 
{
    _buffer = vk_buffer;
    _allocated_memory = memory;
    _external_allocation = true;
}

DefaultBuffer::DefaultBuffer(Device *device, size_t size_bytes, bool used_as_accel, VkBufferUsageFlagBits extra_bit)
    : Buffer{device, size_bytes} {
    auto res = device->allocator()
                   .allocate_buffer(
                       size_bytes,
                       static_cast<VkBufferUsageFlagBits>(
                           extra_bit |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                           VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
                           (used_as_accel ? (VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) :
                                            0)),
                       AccessType::None);
    _buffer = res.buffer;
    _allocation = res.allocation;
}
DefaultBuffer::~DefaultBuffer() {
    if (!_buffer) return;
    if (_external_allocation) {
        vkDestroyBuffer(device()->logic_device(), _buffer, Device::alloc_callbacks());
        vkFreeMemory(device()->logic_device(), _allocated_memory, Device::alloc_callbacks());
    } else if (_allocation) {
        device()->allocator().destroy_buffer({_buffer, _allocation});
    }
}
uint64_t Buffer::get_device_address() const {
    VkBufferDeviceAddressInfoKHR buffer_device_address_info{};
    buffer_device_address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    buffer_device_address_info.buffer = vk_buffer();
    return vkGetBufferDeviceAddress(device()->logic_device(), &buffer_device_address_info);
}
DefaultBuffer::DefaultBuffer(DefaultBuffer &&rhs)
    : Buffer(std::move(rhs)) {
    _buffer = rhs._buffer;
    _external_allocation = rhs._external_allocation;
    if (_external_allocation)
        _allocated_memory = rhs._allocated_memory;
    else
        _allocation = rhs._allocation;
    rhs._buffer = nullptr;
}
SparseBuffer::SparseBuffer(Device *device, size_t size_bytes, bool used_as_accel, VkBufferUsageFlagBits extra_bit)
    : Buffer(device, size_bytes) {
    VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
        .size = size_bytes,
        .usage = static_cast<VkBufferUsageFlags>(
            extra_bit |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT |
            (used_as_accel ? (VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) :
                             0)),
        .queueFamilyIndexCount = 0};
    VK_CHECK_RESULT(vkCreateBuffer(
        device->logic_device(),
        &create_info,
        Device::alloc_callbacks(),
        &_buffer));
}
SparseBuffer::SparseBuffer(SparseBuffer &&rhs)
    : Buffer(std::move(rhs)) {
    _buffer = rhs._buffer;
    rhs._buffer = nullptr;
}
SparseBuffer::~SparseBuffer() {
    if (_buffer) {
        vkDestroyBuffer(device()->logic_device(), _buffer, Device::alloc_callbacks());
    }
}
void BufferFlusher::mark_dirty(size_t begin, size_t end) {
    {
        auto t = _begin.load();
        while (true) {
            auto desired = std::min(begin, t);
            if (_begin.compare_exchange_weak(t, desired))
                break;
        }
    }
    {
        auto t = _end.load();
        while (true) {
            auto desired = std::max(end, t);
            if (_end.compare_exchange_weak(t, desired))
                break;
        }
    }
}
void BufferFlusher::flush(Device *device, void *alloc) {
    size_t begin, end;
    begin = _begin.exchange(std::numeric_limits<size_t>::max());
    end = _end.exchange(0);
    if (begin < end) {
        vmaFlushAllocation(
            device->allocator().allocator(),
            static_cast<VmaAllocation>(alloc),
            begin, end - begin);
    }
}
}// namespace lc::vk
