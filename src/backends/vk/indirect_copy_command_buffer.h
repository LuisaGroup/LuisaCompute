#pragma once

#include "buffer.h"
#include "vk_allocator.h"

namespace lc::vk {

// The command array a vkCmdCopyMemoryIndirectKHR reads its
// VkCopyMemoryIndirectCommandKHR records from.
//
// Requirements this class satisfies:
//   * VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-12210: the range must be
//     allocated from a buffer created with VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT.
//   * VUID-VkCopyMemoryIndirectInfoKHR-copyAddressRange-10942 / -10943: the
//     address is 4-byte aligned and the stride is a multiple of 4 and at least
//     sizeof(VkCopyMemoryIndirectCommandKHR) (the records are packed tightly).
//   * The records must reach the device before the command is submitted. The
//     buffer is host writable, `append()` writes and flushes them before the
//     batch's command is recorded, and Vulkan makes host writes that precede a
//     queue submission visible to that submission, so no barrier is needed -
//     exactly the contract the backend's upload buffers already rely on.
//
// Lifetime of the records: the device reads them when the *command* executes,
// which is after the whole command buffer has been recorded, so a record region
// may not be reused by another batch of the same command buffer. Every
// `append()` therefore hands out a fresh, disjoint record region and the cursor
// only restarts once the command buffer that consumed the records has retired
// (`reset_record_cursor()`, called from `CommandBufferState::reset()` after the
// GPU finished). A command buffer whose batches need more records than
// `capacity()` keeps the pre-existing per-copy path for the remaining copies
// instead of overwriting records that are already in flight.
//
// The buffer is created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT so its
// VkDeviceAddress is stable for the lifetime of the object. When the device does
// not expose buffer device addresses the class reports
// `device_address_capable() == false` and `device_address() == 0`, and the
// planner (`detail::plan_buffer_copy_batch`) then keeps the direct path.
class IndirectCopyCommandBuffer final : public Buffer {
private:
    AllocatedBuffer _allocation{};
    void *_mapped{nullptr};
    size_t _capacity{0u};// in VkCopyMemoryIndirectCommandKHR records
    // Records already handed to commands recorded in the command buffer that is
    // being recorded. Starts at zero for every command buffer and only moves
    // forward until the command buffer retires.
    size_t _record_cursor{0u};
    uint64_t _device_address{0u};

public:
    // `capacity` is a record count; the backing VkBuffer holds
    // `capacity * sizeof(VkCopyMemoryIndirectCommandKHR)` bytes.
    IndirectCopyCommandBuffer(Device *device, size_t capacity);
    ~IndirectCopyCommandBuffer() override;
    IndirectCopyCommandBuffer(IndirectCopyCommandBuffer const &) = delete;
    IndirectCopyCommandBuffer(IndirectCopyCommandBuffer &&) = delete;
    IndirectCopyCommandBuffer &operator=(IndirectCopyCommandBuffer const &) = delete;
    IndirectCopyCommandBuffer &operator=(IndirectCopyCommandBuffer &&) = delete;
    [[nodiscard]] VkBuffer vk_buffer() const override { return _allocation.buffer; }
    // Number of VkCopyMemoryIndirectCommandKHR records the array can hold.
    [[nodiscard]] size_t capacity() const noexcept { return _capacity; }
    // Records that are still free for the command buffer being recorded.
    [[nodiscard]] size_t available_records() const noexcept {
        return _record_cursor < _capacity ? _capacity - _record_cursor : 0u;
    }
    // Device address of the next free record, or 0 when the array is full or
    // not addressable.
    [[nodiscard]] uint64_t next_record_address() const noexcept;
    // Device address of record 0, or 0 when the buffer is not addressable.
    [[nodiscard]] uint64_t device_address() const noexcept {
        return _device_address;
    }
    // Copies `count` packed records into the next free region of the array,
    // flushes them to the device and advances the cursor. Returns the device
    // address of the first written record, or 0 when the array has no room left
    // (the caller then keeps the pre-existing per-copy path).
    [[nodiscard]] uint64_t append(
        VkCopyMemoryIndirectCommandKHR const *records, size_t count);
    // Restarts the record namespace. Only valid once every command buffer that
    // consumed records has completed on the device.
    void reset_record_cursor() noexcept { _record_cursor = 0u; }
};

}// namespace lc::vk
