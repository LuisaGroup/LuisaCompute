#pragma once
#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <luisa/core/basic_types.h>
namespace lc::vk {
using namespace luisa;
class Device;
struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
};
struct AllocatedImage {
    VkImage image;
    VmaAllocation allocation;
};
enum class AccessType {
    None,
    Upload,
    ReadBack
};
class VkAllocator {
    VmaAllocator _allocator;

public:
    auto allocator() const { return _allocator; }
    VkAllocator(Device &device);
    ~VkAllocator();
    AllocatedBuffer allocate_buffer(size_t byte_size, VkBufferUsageFlagBits usage, AccessType access);
    AllocatedImage allocate_image(
        VkImageType dimension,
        VkFormat format,
        uint3 size,
        uint mip_level,
        VkImageUsageFlags usage);
    void destroy_buffer(AllocatedBuffer const &buffer);
    void destroy_image(AllocatedImage const &img);
};
}// namespace lc::vk
