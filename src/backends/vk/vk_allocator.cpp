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

namespace lc::vk {

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
           device.copy_queue_index()})} {
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
        .vulkanApiVersion = VK_API_VERSION_1_3,
        .pTypeExternalMemoryHandleTypes = nullptr};
    VK_CHECK_RESULT(vmaCreateAllocator(&create_info, &_allocator));
}
AllocatedImage VkAllocator::allocate_image(
    VkImageType dimension,
    VkFormat format,
    uint3 size,
    uint mip_level,
    VkImageUsageFlags usage) {
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
    VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    AllocatedImage r;
    VK_CHECK_RESULT(vmaCreateImage(_allocator, &image_info, &alloc_info, &r.image, &r.allocation, nullptr));
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
