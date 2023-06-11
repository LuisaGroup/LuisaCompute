#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "vk_allocator.h"
#include "device.h"
#include "log.h"
namespace lc::vk {
AllocatedBuffer VkAllocator::allocate_buffer(size_t byte_size, VkBufferUsageFlagBits usage, AccessType access) {
    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byte_size,
        .usage = static_cast<VkBufferUsageFlags>(usage)};
    VmaAllocationCreateInfo allocInfo = {
        .flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    switch (access) {
        case AccessType::ReadBack:
            allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            break;
        case AccessType::Upload:
            allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            break;
    }
    AllocatedBuffer r;
    VK_CHECK_RESULT(vmaCreateBuffer(_allocator, &bufferInfo, &allocInfo, &r.buffer, &r.allocation, nullptr));
    return r;
}
VkAllocator::VkAllocator(Device &device) {
    VmaAllocatorCreateInfo createInfo{
        .flags = {},
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
    VK_CHECK_RESULT(vmaCreateAllocator(&createInfo, &_allocator));
}
AllocatedImage VkAllocator::allocate_image(
    VkImageType dimension,
    VkFormat format,
    uint3 size,
    uint mip_level,
    VkImageUsageFlags usage) {
    VkImageCreateInfo imageInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = dimension,
        .format = format,
        .extent = VkExtent3D{
            .width = size.x,
            .height = size.y,
            .depth = size.z},
        .mipLevels = mip_level,
        .usage = usage};
    VmaAllocationCreateInfo allocInfo = {
        .flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO};
    AllocatedImage r;
    VK_CHECK_RESULT(vmaCreateImage(_allocator, &imageInfo, &allocInfo, &r.image, &r.allocation, nullptr));
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
}// namespace lc::vk
