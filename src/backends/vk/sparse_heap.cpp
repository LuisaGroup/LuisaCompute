#include "sparse_heap.h"

#include <limits>

#include <luisa/core/logging.h>

#include "device.h"
#include "sparse_binding_plan.h"

namespace lc::vk {

VulkanSparseHeap::VulkanSparseHeap(
    Device *device, size_t byte_size) noexcept
    : _device{device}, _byte_size{byte_size} {
    LUISA_ASSERT(device != nullptr,
                 "Cannot create a Vulkan sparse heap without a device.");
    LUISA_ASSERT(byte_size != 0u,
                 "Cannot create an empty Vulkan sparse heap.");
}

VulkanSparseHeap::~VulkanSparseHeap() noexcept {
    if (_allocation != nullptr) {
        _device->allocator().dealloc_sparse(_allocation);
    }
}

SparseHeapMemory VulkanSparseHeap::acquire(
    VkMemoryRequirements requirements,
    VkDeviceSize required_binding_size) noexcept {
    std::lock_guard lock{_mutex};
    LUISA_ASSERT(requirements.alignment != 0u,
                 "Vulkan reported zero sparse-memory alignment.");
    LUISA_ASSERT(requirements.memoryTypeBits != 0u,
                 "Vulkan reported no compatible memory types for a sparse resource.");
    LUISA_ASSERT(required_binding_size != 0u,
                 "A Vulkan sparse binding must consume at least one byte.");
    LUISA_ASSERT(
        required_binding_size <= static_cast<VkDeviceSize>(_byte_size),
        "Vulkan sparse binding requires {} bytes, but its heap exposes only {} bytes.",
        required_binding_size, _byte_size);

    if (_allocation == nullptr) {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(
            _device->physical_device(), &memory_properties);
        auto allowed_memory_types = requirements.memoryTypeBits;
        for (auto i = 0u; i < memory_properties.memoryTypeCount; i++) {
            if ((memory_properties.memoryTypes[i].propertyFlags &
                 VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0u) {
                allowed_memory_types &= ~(1u << i);
            }
        }
        LUISA_ASSERT(
            allowed_memory_types != 0u,
            "All Vulkan memory types compatible with this sparse resource are "
            "lazily allocated, which sparse bindings forbid.");

        const auto alignment = requirements.alignment;
        const auto logical_size = static_cast<VkDeviceSize>(_byte_size);
        LUISA_ASSERT(
            logical_size <= std::numeric_limits<VkDeviceSize>::max() -
                                (alignment - 1u),
            "Vulkan sparse heap size {} cannot be aligned to {} bytes.",
            logical_size, alignment);
        requirements.size =
            (logical_size + alignment - 1u) / alignment * alignment;
        requirements.memoryTypeBits = allowed_memory_types;
        VmaAllocationCreateInfo allocation_info{
            .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT |
                     VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
            .usage = VMA_MEMORY_USAGE_GPU_ONLY,
            .memoryTypeBits = allowed_memory_types};
        _device->allocator().alloc_sparse(
            requirements, &allocation_info,
            _allocation, &_allocation_info);
    }

    auto compatibility = detail::validate_sparse_heap_compatibility({
        .required_binding_size = required_binding_size,
        .logical_heap_size = static_cast<VkDeviceSize>(_byte_size),
        .allocation_offset = _allocation_info.offset,
        .required_alignment = requirements.alignment,
        .memory_type_index = _allocation_info.memoryType,
        .memory_type_bits = requirements.memoryTypeBits});
    LUISA_ASSERT(
        static_cast<bool>(compatibility),
        "Vulkan sparse heap is incompatible with the resource (status {}, "
        "memory type {}, mask 0x{:08x}, offset {}, alignment {}).",
        static_cast<uint32_t>(compatibility.status),
        _allocation_info.memoryType, requirements.memoryTypeBits,
        _allocation_info.offset, requirements.alignment);
    LUISA_ASSERT(
        required_binding_size <= _allocation_info.size,
        "Vulkan sparse binding requires {} bytes, but the physical heap "
        "allocation contains only {} bytes.",
        required_binding_size, _allocation_info.size);
    return SparseHeapMemory{
        .memory = _allocation_info.deviceMemory,
        .offset = _allocation_info.offset};
}

}// namespace lc::vk
