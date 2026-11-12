#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "vk_mem_alloc.h"
#pragma clang diagnostic pop
#include "vk_allocator.h"
#include "device.h"
#include "log.h"
#include <algorithm>
#include <limits>

namespace lc::vk {

namespace vk_allocator_detail {

// Flattens VkPhysicalDeviceMemoryProperties into the handle-free shape the pure
// memory type selector consumes.
[[nodiscard]] uint32_t flatten_memory_types(
    VkPhysicalDeviceMemoryProperties const &properties,
    detail::MemoryTypeCandidate *candidates) noexcept {
    auto count = std::min<uint32_t>(properties.memoryTypeCount, VK_MAX_MEMORY_TYPES);
    for (auto i = 0u; i < count; i++) {
        auto heap = properties.memoryTypes[i].heapIndex;
        LUISA_ASSERT(heap < properties.memoryHeapCount,
                     "Vulkan memory type {} names heap {}, but the physical device "
                     "reports only {} heaps.",
                     i, heap, properties.memoryHeapCount);
        candidates[i] = detail::MemoryTypeCandidate{
            .property_flags = properties.memoryTypes[i].propertyFlags,
            .heap_flags = properties.memoryHeaps[heap].flags};
    }
    return count;
}

}// namespace vk_allocator_detail

void VkAllocator::apply_queue_sharing(
    VkBufferCreateInfo &info) const noexcept {
    info.sharingMode = _queue_family_sharing.sharing_mode();
    info.queueFamilyIndexCount =
        _queue_family_sharing.create_info_family_count();
    info.pQueueFamilyIndices =
        _queue_family_sharing.create_info_family_indices();
}

void VkAllocator::apply_queue_sharing(
    VkImageCreateInfo &info) const noexcept {
    info.sharingMode = _queue_family_sharing.sharing_mode();
    info.queueFamilyIndexCount =
        _queue_family_sharing.create_info_family_count();
    info.pQueueFamilyIndices =
        _queue_family_sharing.create_info_family_indices();
}

AllocatedBuffer VkAllocator::allocate_buffer(size_t byte_size, VkBufferUsageFlagBits usage, AccessType access) {
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byte_size,
        .usage = static_cast<VkBufferUsageFlags>(usage)};
    apply_queue_sharing(buffer_info);
    VmaAllocationCreateInfo alloc_info = {.flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT};
    switch (access) {
        case AccessType::kReadBack:
            alloc_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
            // Prefer HOST_CACHED memory for faster CPU readback.
            // VMA will fall back to uncached if cached is unavailable.
            alloc_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            break;
        case AccessType::kUpload:
            alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            break;
        default:
            alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            break;
    }
    AllocatedBuffer r;
    VK_CHECK_RESULT(vmaCreateBuffer(_allocator, &buffer_info, &alloc_info, &r.buffer, &r.allocation, nullptr));
    return r;
}
VkAllocator::VkAllocator(Device &device)
    : _allocator{nullptr},
      _queue_family_sharing{detail::plan_queue_family_sharing(
          {device.graphics_queue_index(),
           device.compute_queue_index(),
           device.copy_queue_index()})},
      _device{&device} {
    VmaAllocatorCreateInfo create_info{
        .flags = VmaAllocatorCreateFlags(device.enable_device_address() ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0),
        .physicalDevice = device.physical_device(),
        .device = device.logic_device(),
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks = nullptr,
        .pDeviceMemoryCallbacks = nullptr,
        .pHeapSizeLimit = nullptr,
        .pVulkanFunctions = nullptr,
        .instance = device.instance(),
        // The backend supports Vulkan 1.2 devices; VMA must not require 1.3.
        .vulkanApiVersion = VK_API_VERSION_1_2,
        .pTypeExternalMemoryHandleTypes = nullptr};
    VK_CHECK_RESULT(vmaCreateAllocator(&create_info, &_allocator));
    // Dedicated allocations select their own memory type, so the properties are
    // queried once here instead of per allocation.
    vkGetPhysicalDeviceMemoryProperties(device.physical_device(), &_memory_properties);
    LUISA_ASSERT(_memory_properties.memoryTypeCount > 0u &&
                     _memory_properties.memoryTypeCount <= VK_MAX_MEMORY_TYPES,
                 "Vulkan reported {} memory types, expected 1..{}.",
                 _memory_properties.memoryTypeCount, VK_MAX_MEMORY_TYPES);
}
AllocatedImage VkAllocator::allocate_image(
    VkImageType dimension,
    VkFormat format,
    uint3 size,
    uint mip_level,
    VkImageUsageFlags usage,
    detail::ImageTilingPolicy tiling_policy) {
    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = dimension,
        .format = format,
        .extent = VkExtent3D{
            .width = size.x,
            .height = size.y,
            .depth = size.z},
        .mipLevels = mip_level,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    apply_queue_sharing(image_info);
    // VK_EXT_image_tiling_control: the default policy is defined to be identical
    // to omitting the structure, so a default request (and a device without the
    // extension) leaves the pre-existing create info untouched.
    VkImageTilingControlCreateInfoEXT tiling_control{
        .sType = VK_STRUCTURE_TYPE_IMAGE_TILING_CONTROL_CREATE_INFO_EXT};
    auto tiling_plan = detail::plan_image_tiling_control(
        _device->optional_extension_enabled(
            detail::OptionalExtension::IMAGE_TILING_CONTROL),
        tiling_policy, image_info.tiling);
    if (tiling_plan.chain_create_info) {
        tiling_control.pNext = image_info.pNext;
        tiling_control.tilingControl = tiling_plan.tiling_control;
        image_info.pNext = &tiling_control;
    }
    VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    AllocatedImage r;
    VK_CHECK_RESULT(vmaCreateImage(_allocator, &image_info, &alloc_info, &r.image, &r.allocation, nullptr));
    if (tiling_plan.chain_create_info) {
        // The extension's tiling policy is really in use for this device.
        _device->mark_optional_extension_active(
            detail::OptionalExtension::IMAGE_TILING_CONTROL);
    }
    return r;
}
void VkAllocator::destroy_buffer(AllocatedBuffer const &buffer) {
    vmaDestroyBuffer(
        _allocator,
        buffer.buffer,
        buffer.allocation);
}
void VkAllocator::destroy_image(AllocatedImage const &img) {
    vmaDestroyImage(
        _allocator,
        img.image,
        img.allocation);
}
NativeBufferAllocation VkAllocator::allocate_native_buffer(
    size_t byte_size, VkBufferUsageFlags usage,
    VkBufferUsageFlags2KHR usage2, bool host_visible,
    MemoryInitPolicy init) noexcept {
    LUISA_ASSERT(byte_size > 0u,
                 "A Vulkan buffer must have a non-zero size.");
    LUISA_ASSERT(byte_size <= std::numeric_limits<VkDeviceSize>::max(),
                 "Vulkan buffer size {} exceeds VkDeviceSize.",
                 byte_size);
    auto device = _device->logic_device();
    // Per spec, VkBufferUsageFlags2CreateInfoKHR *replaces* the 32-bit usage
    // value, so a caller that chains it must repeat every low flag there.
    LUISA_ASSERT(
        usage2 == 0u ||
            static_cast<VkBufferUsageFlags>(usage2) == usage,
        "Vulkan 64-bit buffer usage {:#x} must repeat the 32-bit usage {:#x}.",
        static_cast<uint64_t>(usage2), static_cast<uint32_t>(usage));
    VkBufferUsageFlags2CreateInfoKHR usage2_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
        .pNext = nullptr,
        .usage = usage2};
    VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = usage2 == 0u ? nullptr : static_cast<const void *>(&usage2_info),
        .size = static_cast<VkDeviceSize>(byte_size),
        .usage = usage};
    apply_queue_sharing(buffer_info);
    VkBuffer buffer{};
    VK_CHECK_RESULT(vkCreateBuffer(
        device, &buffer_info, Device::alloc_callbacks(), &buffer));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);

    detail::MemoryTypeCandidate candidates[VK_MAX_MEMORY_TYPES]{};
    auto memory_type_count = vk_allocator_detail::flatten_memory_types(
        _memory_properties, candidates);
    // Lazily allocated and protected memory can never be a general purpose
    // destination buffer, and the zero-initialize flag is forbidden for the
    // latter (see detail::plan_zero_initialize_allocation).
    constexpr auto forbidden_flags = static_cast<uint32_t>(
        VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
    auto required_flags = static_cast<uint32_t>(
        host_visible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT :
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto preferred_flags = static_cast<uint32_t>(
        host_visible ? (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT) : 0u);
    auto selection = detail::select_memory_type(
        requirements.memoryTypeBits, memory_type_count, candidates,
        required_flags, preferred_flags, forbidden_flags);
    LUISA_ASSERT(
        static_cast<bool>(selection),
        "No Vulkan memory type can back a {}-byte buffer (memoryTypeBits {:#x}, "
        "{} types, host visible {}): {}.",
        byte_size, requirements.memoryTypeBits, memory_type_count,
        host_visible,
        detail::memory_type_selection_status_name(selection.status));
    auto zero_initialize = detail::plan_zero_initialize_allocation(
        _device->optional_extension_enabled(
            detail::OptionalExtension::ZERO_INITIALIZE_DEVICE_MEMORY),
        init == MemoryInitPolicy::kZeroInitialized,
        // This path never performs a memory import operation.
        false, selection.memory_type_index, memory_type_count,
        candidates[selection.memory_type_index].property_flags);

    // VK_KHR_extended_flags is deliberately not used here (or anywhere else in
    // the backend): it only widens flag *types* and defines no new bits, and
    // every allocation flag this path composes already fits in the 32-bit
    // VkMemoryAllocateFlags. The 64-bit buffer usage above comes from
    // VK_KHR_maintenance5's VkBufferUsageFlags2CreateInfoKHR, not from
    // VK_KHR_extended_flags.
    VkMemoryAllocateFlags allocate_flags{0u};
    if ((usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0u) {
        // VUID-VkMemoryAllocateInfo-pNext-03335: a buffer created with
        // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT needs the matching allocate
        // flag on its memory.
        allocate_flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }
    if (zero_initialize.use_flag()) {
        allocate_flags |= VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT;
    }
    VkMemoryAllocateFlagsInfo allocate_flags_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = nullptr,
        .flags = allocate_flags};
    VkMemoryAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = allocate_flags == 0u ? nullptr : static_cast<const void *>(&allocate_flags_info),
        .allocationSize = requirements.size,
        .memoryTypeIndex = selection.memory_type_index};
    VkDeviceMemory memory{};
    auto result = vkAllocateMemory(
        device, &allocate_info, Device::alloc_callbacks(), &memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, Device::alloc_callbacks());
        LUISA_ERROR(
            "Failed to allocate {} bytes of Vulkan memory (type {}, flags {:#x}): {}.",
            requirements.size, selection.memory_type_index, allocate_flags,
            vks::tools::error_string(result));
    }
    result = vkBindBufferMemory(device, buffer, memory, 0u);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device, memory, Device::alloc_callbacks());
        vkDestroyBuffer(device, buffer, Device::alloc_callbacks());
        LUISA_ERROR(
            "Failed to bind {} bytes of Vulkan memory (type {}) to a buffer: {}.",
            requirements.size, selection.memory_type_index,
            vks::tools::error_string(result));
    }
    if (zero_initialize.use_flag()) {
        // The flag is really in use for this device.
        _device->mark_optional_extension_active(
            detail::OptionalExtension::ZERO_INITIALIZE_DEVICE_MEMORY);
    }
    return NativeBufferAllocation{
        .buffer = buffer,
        .memory = memory,
        .allocation_size = requirements.size,
        .driver_zeroed = zero_initialize.use_flag()};
}
void VkAllocator::destroy_native_buffer(
    NativeBufferAllocation &allocation) noexcept {
    auto device = _device->logic_device();
    if (allocation.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, allocation.buffer, Device::alloc_callbacks());
    }
    if (allocation.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, allocation.memory, Device::alloc_callbacks());
    }
    allocation = {};
}
VkAllocator::~VkAllocator() {
    vmaDestroyAllocator(_allocator);
}
void VkAllocator::alloc_sparse(SparseAllocCmdList &&cmdlist, SparseAllocResult &result) {
    LUISA_DEBUG_ASSERT(cmdlist.create_info.empty() || cmdlist.create_info.size() == cmdlist.mem_requires.size());
    auto size = cmdlist.mem_requires.size();
    if (size == 0) [[unlikely]]
        return;
    if (cmdlist.create_info.empty()) {
        vstd::push_back_all(cmdlist.create_info, size, VmaAllocationCreateInfo{});
    }
    result.alloc_result.clear();
    result.alloc_result_info.clear();
    luisa::enlarge_by(result.alloc_result, size);
    luisa::enlarge_by(result.alloc_result_info, size);
    VK_CHECK_RESULT(
        vmaAllocateMemoryPages(_allocator, cmdlist.mem_requires.data(), cmdlist.create_info.data(), size, result.alloc_result.data(), result.alloc_result_info.data()));
    cmdlist.create_info.clear();
    cmdlist.mem_requires.clear();
}
void VkAllocator::alloc_sparse(
    VkMemoryRequirements const &require,
    VmaAllocationCreateInfo const *alloc_info,
    VmaAllocation &result,
    VmaAllocationInfo *result_info) {
    VK_CHECK_RESULT(vmaAllocateMemory(_allocator, &require, alloc_info, &result, result_info));
}
void VkAllocator::dealloc_sparse(vstd::vector<VmaAllocation> &alloc) {
    if (alloc.empty()) [[unlikely]]
        return;
    vmaFreeMemoryPages(_allocator, alloc.size(), alloc.data());
}
void VkAllocator::dealloc_sparse(VmaAllocation const &alloc) {
    vmaFreeMemory(_allocator, alloc);
}

}// namespace lc::vk
