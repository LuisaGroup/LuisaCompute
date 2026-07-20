#include "upload_buffer.h"
#include "readback_buffer.h"
#include "default_buffer.h"
#include "sparse_buffer.h"
#include "device.h"
#include "device_feature_plan.h"
#include "log.h"
#include <bit>
#include <limits>
#include <numeric>
namespace lc::vk {
namespace {

[[nodiscard]] size_t word_addressable_size(size_t logical_size) noexcept {
    LUISA_ASSERT(logical_size <= std::numeric_limits<size_t>::max() - 3u,
                 "Vulkan buffer size {} cannot be rounded to a storage word.",
                 logical_size);
    return (logical_size + 3u) & ~size_t{3u};
}

}// namespace

StorageBufferDescriptorRange storage_buffer_descriptor_range(
    const Buffer *buffer, size_t view_offset, size_t view_size,
    size_t logical_element_stride) {
    LUISA_ASSERT(buffer != nullptr, "Vulkan buffer argument is null.");
    LUISA_ASSERT(view_size > 0u,
                 "Vulkan does not permit an empty storage-buffer descriptor.");
    LUISA_ASSERT(view_offset <= buffer->byte_size() &&
                     view_size <= buffer->byte_size() - view_offset,
                 "Vulkan buffer view [{}, {}) exceeds the logical buffer size {}.",
                 view_offset, view_offset + view_size, buffer->byte_size());

    const auto descriptor_alignment = std::max<VkDeviceSize>(
        1u, buffer->device()->properties().limits.minStorageBufferOffsetAlignment);
    LUISA_ASSERT(std::has_single_bit(static_cast<uint64_t>(descriptor_alignment)),
                 "Vulkan minStorageBufferOffsetAlignment {} is not a power of two.",
                 descriptor_alignment);
    if (logical_element_stride != 0u) {
        LUISA_ASSERT(view_offset % logical_element_stride == 0u &&
                         view_size % logical_element_stride == 0u,
                     "Typed Vulkan buffer view [{}, {}) is not a multiple of "
                     "its {}-byte element stride.",
                     view_offset, view_offset + view_size,
                     logical_element_stride);
    }

    // Every SPIR-V byte-addressed binding is physically a uint32 array. Typed
    // bindings instead require descriptor-relative indices to remain exact
    // multiples of the logical element stride.
    const auto storage_element_alignment = static_cast<VkDeviceSize>(
        logical_element_stride == 0u ? sizeof(uint32_t) :
                                       logical_element_stride);
    const auto alignment_factor =
        descriptor_alignment /
        std::gcd(descriptor_alignment, storage_element_alignment);
    LUISA_ASSERT(storage_element_alignment <=
                     std::numeric_limits<VkDeviceSize>::max() /
                         alignment_factor,
                 "Vulkan buffer descriptor base alignment overflows for {} and {}.",
                 descriptor_alignment, storage_element_alignment);
    const auto descriptor_base_alignment =
        alignment_factor * storage_element_alignment;
    const auto descriptor_offset =
        static_cast<VkDeviceSize>(view_offset) /
        descriptor_base_alignment * descriptor_base_alignment;

    LUISA_ASSERT(view_offset <=
                     std::numeric_limits<VkDeviceSize>::max() - view_size,
                 "Vulkan buffer view end overflows for offset {} and size {}.",
                 view_offset, view_size);
    const auto logical_end =
        static_cast<VkDeviceSize>(view_offset) + view_size;
    LUISA_ASSERT(logical_end <= std::numeric_limits<VkDeviceSize>::max() - 3u,
                 "Vulkan buffer view end {} cannot be rounded to a storage word.",
                 logical_end);
    const auto addressable_end = (logical_end + 3u) & ~VkDeviceSize{3u};
    LUISA_ASSERT(addressable_end <= buffer->addressable_byte_size(),
                 "Vulkan XIR/SPIR-V word-backed access needs bytes through {}, "
                 "but this buffer proves only {} addressable bytes. External "
                 "buffers with a partial final word must provide physical padding.",
                 addressable_end, buffer->addressable_byte_size());

    const auto descriptor_range = addressable_end - descriptor_offset;
    const auto descriptor_bias =
        static_cast<uint64_t>(view_offset - descriptor_offset);
    LUISA_ASSERT(descriptor_bias <= std::numeric_limits<uint32_t>::max(),
                 "Vulkan descriptor-relative buffer bias {} exceeds the "
                 "32-bit address range guaranteed by maxStorageBufferRange.",
                 descriptor_bias);
    LUISA_ASSERT(descriptor_range <=
                     buffer->device()->properties().limits.maxStorageBufferRange,
                 "Vulkan storage-buffer descriptor range {} exceeds the device "
                 "limit {} (aligned base {}, logical view [{}, {})).",
                 descriptor_range,
                 buffer->device()->properties().limits.maxStorageBufferRange,
                 descriptor_offset, view_offset, logical_end);
    return StorageBufferDescriptorRange{
        descriptor_offset,
        descriptor_range,
        StorageBufferMetadata{
            descriptor_bias,
            static_cast<uint64_t>(view_size)}};
}

void vma_defragment(Device *device) {
    if (!device) return;

    VmaDefragmentationInfo defrag_info = {};
    defrag_info.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;
    defrag_info.pool = VK_NULL_HANDLE;// Defragment all default pools

    VmaDefragmentationContext defrag_ctx;
    VkResult result = vmaBeginDefragmentation(
        device->allocator().allocator(),
        &defrag_info,
        &defrag_ctx);

    if (result != VK_SUCCESS) {
        return;
    }

    // Perform defragmentation passes
    [&] {
        VmaDefragmentationPassMoveInfo pass_info = {};
        result = vmaBeginDefragmentationPass(
            device->allocator().allocator(),
            defrag_ctx,
            &pass_info);

        if (result == VK_SUCCESS) {
            // No more moves needed
            return;
        }

        if (result != VK_INCOMPLETE) {
            // Error occurred
            return;
        }

        // Mark all moves as IGNORE since we don't have access to buffer/image handles
        // to recreate them at the new locations. This will still allow VMA to free
        // empty memory blocks.
        for (uint32_t i = 0; i < pass_info.moveCount; ++i) {
            pass_info.pMoves[i].operation = VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE;
        }

        result = vmaEndDefragmentationPass(
            device->allocator().allocator(),
            defrag_ctx,
            &pass_info);
    }();

    VmaDefragmentationStats stats = {};
    vmaEndDefragmentation(
        device->allocator().allocator(),
        defrag_ctx,
        &stats);

    // Log compaction results
    if (stats.bytesMoved > 0 || stats.bytesFreed > 0) {
        LUISA_INFO("VMA memory compacted: {} bytes moved, {} bytes freed, {} allocations moved, {} blocks freed",
                   stats.bytesMoved,
                   stats.bytesFreed,
                   stats.allocationsMoved,
                   stats.deviceMemoryBlocksFreed);
    }
}
UploadBuffer::UploadBuffer(Device *device, size_t size_bytes)
    : Buffer{device, size_bytes},
      _res{
          device->allocator()
              .allocate_buffer(
                  size_bytes,
                  (VkBufferUsageFlagBits)((uint)VK_BUFFER_USAGE_TRANSFER_SRC_BIT | (uint)VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
                  AccessType::kUpload)} {
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
                  AccessType::kReadBack)} {
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
    _flusher.mark_dirty(offset, offset + size);
}
void ReadbackBuffer::copy_to(void *data, size_t offset, size_t size) const {
    vmaInvalidateAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        offset,
        size);
    memcpy(data, reinterpret_cast<std::byte *>(_mapped_ptr) + offset, size);
}
bool UploadBuffer::flush_host() const {
    _flusher.flush(device(), _res.allocation);
    return true;
}
bool ReadbackBuffer::flush_host() const {
    // ReadbackBuffer uses vmaInvalidateAllocation inline in copy_to();
    // no flush needed here. Return true to signal host-visibility.
    return true;
}
void ReadbackBuffer::flush_range(size_t begin, size_t end) {
    VK_CHECK_RESULT(vmaInvalidateAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        begin,
        end - begin));
}
void UploadBuffer::flush_range(size_t begin, size_t end) {
    VK_CHECK_RESULT(vmaFlushAllocation(
        device()->allocator().allocator(),
        _res.allocation,
        begin, end - begin));
}

DefaultBuffer::DefaultBuffer(Device *device, VkBuffer vk_buffer, VkDeviceMemory memory, size_t size_bytes)
    : Buffer{device, size_bytes} {
    _buffer = vk_buffer;
    _allocated_memory = memory;
    _external_allocation = true;
}

DefaultBuffer::DefaultBuffer(Device *device, size_t size_bytes, bool used_as_accel, VkBufferUsageFlagBits extra_bit)
    : Buffer{device, size_bytes, word_addressable_size(size_bytes)} {
    auto res = device->allocator()
                   .allocate_buffer(
                       addressable_byte_size(),
                       static_cast<VkBufferUsageFlagBits>(
                           extra_bit |
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                           VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT |
                           (device->enable_device_address() ? VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT : 0) |
                           ((device->enable_raytracing() && used_as_accel) ? VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR :
                                                                             0)),
                       AccessType::kNone);
    _buffer = res.buffer;
    _allocation = res.allocation;
}
DefaultBuffer::~DefaultBuffer() {
    if (!_buffer) return;
    if (_external_allocation) {
        if (_allocated_memory) {// owned
            vkDestroyBuffer(device()->logic_device(), _buffer, Device::alloc_callbacks());
            vkFreeMemory(device()->logic_device(), _allocated_memory, Device::alloc_callbacks());
        }
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
DefaultBuffer::DefaultBuffer(DefaultBuffer &&rhs) noexcept
    : Buffer(std::move(rhs)),
      _buffer(rhs._buffer),
      _external_allocation(rhs._external_allocation) {
    if (_external_allocation)
        _allocated_memory = rhs._allocated_memory;
    else
        _allocation = rhs._allocation;
    rhs._buffer = nullptr;
}
SparseBuffer::SparseBuffer(Device *device, size_t size_bytes, bool used_as_accel, VkBufferUsageFlagBits extra_bit)
    : Buffer(device, size_bytes, word_addressable_size(size_bytes)) {
    auto enabled = device->enabled_features();
    auto sparse_features = detail::validate_sparse_buffer_features({.sparse_binding = enabled.sparseBinding == VK_TRUE,
                                                                    .sparse_residency_buffer =
                                                                        enabled.sparseResidencyBuffer == VK_TRUE});
    LUISA_ASSERT(
        static_cast<bool>(sparse_features),
        "Vulkan sparse-buffer creation is unavailable: {}.",
        detail::sparse_residency_feature_status_name(
            sparse_features.status));
    VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
        .size = addressable_byte_size(),
        .usage = static_cast<VkBufferUsageFlags>(
            extra_bit |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            (device->enable_device_address() ? VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT : 0) |
            ((device->enable_raytracing() && used_as_accel) ? (VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) :
                                                              0)),
        .queueFamilyIndexCount = 0};
    device->allocator().apply_queue_sharing(create_info);
    VK_CHECK_RESULT(vkCreateBuffer(
        device->logic_device(), &create_info,
        Device::alloc_callbacks(), &_buffer));
    vkGetBufferMemoryRequirements(
        device->logic_device(), _buffer, &_memory_requirements);
    LUISA_ASSERT(
        _memory_requirements.alignment != 0u,
        "Vulkan reported zero sparse-buffer alignment.");
    // Sparse bindings address [0, VkMemoryRequirements::size), whose size and
    // every bind are aligned to VkMemoryRequirements::alignment. Keep the
    // VkBuffer's API-visible size unchanged; the queried requirements already
    // include any virtual-address padding needed by the final sparse block.
    LUISA_ASSERT(
        _memory_requirements.size != 0u &&
            _memory_requirements.size %
                    _memory_requirements.alignment ==
                0u,
        "Vulkan reported invalid sparse-buffer requirements: size {}, "
        "alignment {}.",
        _memory_requirements.size, _memory_requirements.alignment);
    _sparse_binding_size = _memory_requirements.size;
}
SparseBuffer::SparseBuffer(SparseBuffer &&rhs) noexcept
    : Buffer(std::move(rhs)),
      _buffer(rhs._buffer),
      _memory_requirements(rhs._memory_requirements),
      _sparse_binding_size(rhs._sparse_binding_size) {
    rhs._buffer = nullptr;
}
SparseBuffer::~SparseBuffer() {
    if (_buffer) {
        vkDestroyBuffer(device()->logic_device(), _buffer, Device::alloc_callbacks());
    }
}
void BufferFlusher::mark_dirty(size_t range_begin, size_t range_end) {
    // Track the minimal begin and maximal end across all dirty ranges.
    // Use CAS loops for thread safety (though contention is typically low).
    size_t prev_begin = begin.load(std::memory_order_relaxed);
    while (range_begin < prev_begin) {
        if (begin.compare_exchange_weak(prev_begin, range_begin, std::memory_order_release, std::memory_order_relaxed))
            break;
    }
    size_t prev_end = end.load(std::memory_order_relaxed);
    while (range_end > prev_end) {
        if (end.compare_exchange_weak(prev_end, range_end, std::memory_order_release, std::memory_order_relaxed))
            break;
    }
}
void BufferFlusher::flush(Device *device, void *alloc) {
    size_t flush_begin = begin.exchange(std::numeric_limits<size_t>::max());
    size_t flush_end = end.exchange(0);
    if (flush_begin < flush_end) {
        VK_CHECK_RESULT(vmaFlushAllocation(
            device->allocator().allocator(),
            static_cast<VmaAllocation>(alloc),
            flush_begin, flush_end - flush_begin));
    }
}
}// namespace lc::vk
