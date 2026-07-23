#pragma once

#include <volk.h>
#include "queue_family_contract.h"
#include "vk_mem_alloc.h"
#include <luisa/core/basic_types.h>
#include <luisa/vstl/vector.h>

namespace lc::vk {
using namespace luisa;
class Device;
struct AllocatedBuffer {
    VkBuffer buffer{};
    VmaAllocation allocation{};
};
struct AllocatedImage {
    VkImage image{};
    VmaAllocation allocation{};
};
enum class AccessType {
    kNone,
    kUpload,
    kReadBack
};
struct SparseAllocCmdList {
    vstd::vector<VkMemoryRequirements> mem_requires;
    vstd::vector<VmaAllocationCreateInfo> create_info;
};
struct SparseAllocResult {
    vstd::vector<VmaAllocation> alloc_result;
    vstd::vector<VmaAllocationInfo> alloc_result_info;
};
class VkAllocator {
    VmaAllocator _allocator;
    detail::QueueFamilySharingPlan _queue_family_sharing;
public:
    auto allocator() const { return _allocator; }
    VkAllocator(Device &device);
    ~VkAllocator();
    void apply_queue_sharing(VkBufferCreateInfo &info) const noexcept;
    void apply_queue_sharing(VkImageCreateInfo &info) const noexcept;
    AllocatedBuffer allocate_buffer(size_t byte_size, VkBufferUsageFlagBits usage, AccessType access);
    AllocatedImage allocate_image(
        VkImageType dimension,
        VkFormat format,
        uint3 size,
        uint mip_level,
        VkImageUsageFlags usage);
    void destroy_buffer(AllocatedBuffer const &buffer);
    void destroy_image(AllocatedImage const &img);
    void alloc_sparse(
        VkMemoryRequirements const &require,
        VmaAllocationCreateInfo const *alloc_info,
        VmaAllocation &result,
        VmaAllocationInfo *result_info);
    void alloc_sparse(SparseAllocCmdList &&cmdlist, SparseAllocResult &result);
    void dealloc_sparse(vstd::vector<VmaAllocation> &alloc);
    void dealloc_sparse(VmaAllocation const &alloc);
};

}// namespace lc::vk
