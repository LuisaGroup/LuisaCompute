#include "indirect_copy_command_buffer.h"

#include "copy_command_plan.h"
#include "device.h"
#include "log.h"

#include <cstring>
#include <limits>

#include <luisa/core/logging.h>

namespace lc::vk {

namespace indirect_copy_command_buffer_detail {

// VkBuffer usage of the command array. VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT is
// required by VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-12210; the
// transfer destination bit keeps the buffer usable as a staging target and the
// storage bit matches the backend's standard buffer mask so the device address
// queries of the allocator stay uniform.
[[nodiscard]] inline VkBufferUsageFlags
indirect_copy_usage(bool device_address) noexcept {
    return static_cast<VkBufferUsageFlags>(
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        (device_address ? VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT : 0u));
}

[[nodiscard]] inline size_t
indirect_copy_array_size(size_t capacity) noexcept {
    LUISA_ASSERT(capacity > 0u,
                 "A Vulkan indirect copy command array needs a positive "
                 "capacity.");
    LUISA_ASSERT(capacity <=
                     std::numeric_limits<size_t>::max() /
                         static_cast<size_t>(
                             detail::indirect_copy_command_stride),
                 "A Vulkan indirect copy command array of {} records "
                 "overflows the buffer size.",
                 capacity);
    return capacity *
           static_cast<size_t>(detail::indirect_copy_command_stride);
}

}// namespace indirect_copy_command_buffer_detail

IndirectCopyCommandBuffer::IndirectCopyCommandBuffer(
    Device *device, size_t capacity)
    : Buffer{device,
             indirect_copy_command_buffer_detail::indirect_copy_array_size(capacity),
             indirect_copy_command_buffer_detail::indirect_copy_array_size(capacity),
             device != nullptr && device->enable_device_address()},
      _capacity{capacity} {
    LUISA_ASSERT(device != nullptr,
                 "A Vulkan indirect copy command array needs a live device.");
    _allocation = device->allocator().allocate_buffer(
        byte_size(),
        static_cast<VkBufferUsageFlagBits>(
            indirect_copy_command_buffer_detail::indirect_copy_usage(
                device_address_capable())),
        AccessType::kUpload);
    VK_CHECK_RESULT(vmaMapMemory(
        device->allocator().allocator(),
        _allocation.allocation, &_mapped));
    LUISA_ASSERT(_mapped != nullptr,
                 "A Vulkan indirect copy command array must stay mapped.");
    if (device_address_capable()) {
        _device_address = get_device_address();
        LUISA_ASSERT(
            _device_address != 0u &&
                _device_address %
                        detail::copy_command_alignment ==
                    0u,
            "Vulkan buffer device addresses must be non-zero and 4-byte "
            "aligned.");
    }
}

IndirectCopyCommandBuffer::~IndirectCopyCommandBuffer() {
    if (_mapped != nullptr) {
        vmaUnmapMemory(
            device()->allocator().allocator(),
            _allocation.allocation);
        _mapped = nullptr;
    }
    device()->allocator().destroy_buffer(_allocation);
}

uint64_t IndirectCopyCommandBuffer::next_record_address() const noexcept {
    if (_device_address == 0u || available_records() == 0u) { return 0u; }
    return _device_address +
           static_cast<uint64_t>(_record_cursor) *
               detail::indirect_copy_command_stride;
}

uint64_t IndirectCopyCommandBuffer::append(
    VkCopyMemoryIndirectCommandKHR const *records,
    size_t count) {
    LUISA_ASSERT(records != nullptr && count > 0u,
                 "A Vulkan indirect copy command upload needs at least one "
                 "record.");
    auto const address = next_record_address();
    if (address == 0u || records == nullptr || count == 0u ||
        count > available_records()) {
        // The array has no room left for this batch (or is not addressable), so
        // the caller keeps the pre-existing per-copy path. A record that is
        // already in flight must never be overwritten.
        return 0u;
    }
    auto const offset = _record_cursor *
                        static_cast<size_t>(
                            detail::indirect_copy_command_stride);
    auto const bytes = count * static_cast<size_t>(
                                   detail::indirect_copy_command_stride);
    LUISA_ASSERT(offset + bytes <= byte_size(),
                 "Vulkan indirect copy command upload of {}+{} bytes exceeds "
                 "the {}-byte command array.",
                 offset, bytes, byte_size());
    std::memcpy(static_cast<std::byte *>(_mapped) + offset, records, bytes);
    VK_CHECK_RESULT(vmaFlushAllocation(
        device()->allocator().allocator(),
        _allocation.allocation,
        static_cast<VkDeviceSize>(offset),
        static_cast<VkDeviceSize>(bytes)));
    _record_cursor += count;
    return address;
}

}// namespace lc::vk
