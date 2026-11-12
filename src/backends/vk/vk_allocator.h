#pragma once

#include <volk.h>
#include "queue_family_contract.h"
#include "vk_mem_alloc.h"
#include "memory_policy_plan.h"
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
// How the device memory of a new allocation should be initialized.
//
// `kUndefined` is the pre-existing behaviour: the contents of a fresh Vulkan
// allocation are undefined and the backend never reads them before writing.
// `kZeroInitialized` additionally asks the driver to zero the memory
// (VK_EXT_zero_initialize_device_memory). It is an optimization/capability only:
// when the extension is unsupported, switched off through
// VulkanDeviceConfigExt, or the selected memory type may not carry the flag, the
// allocation silently falls back to `kUndefined`.
enum class MemoryInitPolicy : uint32_t {
    kUndefined = 0u,
    kZeroInitialized,
};
// A VkBuffer bound to its own dedicated VkDeviceMemory.
//
// The optional device-memory policy extensions need allocation-time control that
// a VMA shared block cannot express per allocation (VkMemoryAllocateFlagsInfo
// flags, VkBufferUsageFlags2CreateInfoKHR), so those paths allocate a dedicated
// VkDeviceMemory instead of suballocating one.
struct NativeBufferAllocation {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    VkDeviceSize allocation_size{};
    // True when VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT was requested for
    // this allocation, i.e. the driver guarantees the memory is zeroed before
    // the application can access it.
    bool driver_zeroed{false};
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
    // Needed to read the negotiated optional-extension state and to record that
    // a fast path is really in use. Never null; the allocator is owned by Device.
    Device *_device{nullptr};
    // Cached once at construction: memory type selection for dedicated
    // allocations and the zero-initialize flag decision both need it.
    VkPhysicalDeviceMemoryProperties _memory_properties{};

public:
    auto allocator() const { return _allocator; }
    VkAllocator(Device &device);
    ~VkAllocator();
    void apply_queue_sharing(VkBufferCreateInfo &info) const noexcept;
    void apply_queue_sharing(VkImageCreateInfo &info) const noexcept;
    [[nodiscard]] VkPhysicalDeviceMemoryProperties const &memory_properties() const noexcept {
        return _memory_properties;
    }
    AllocatedBuffer allocate_buffer(size_t byte_size, VkBufferUsageFlagBits usage, AccessType access);
    // `tiling_policy` selects VK_EXT_image_tiling_control's tiling arrangement.
    // `detail::ImageTilingPolicy::kDefault` (and any policy on a device without
    // the extension) keeps the pre-existing VkImageCreateInfo untouched.
    AllocatedImage allocate_image(
        VkImageType dimension,
        VkFormat format,
        uint3 size,
        uint mip_level,
        VkImageUsageFlags usage,
        detail::ImageTilingPolicy tiling_policy = detail::ImageTilingPolicy::kDefault);
    // Creates a VkBuffer plus its own VkDeviceMemory.
    //
    // `usage2` is the 64-bit Vulkan usage mask; when it is non-zero the create
    // info chains VkBufferUsageFlags2CreateInfoKHR and the caller has already
    // proved (through `detail::plan_buffer_usage_flags2`) that the device offers
    // that channel. `init` selects the pre-existing undefined contents or the
    // driver-side zero initialization of VK_EXT_zero_initialize_device_memory.
    [[nodiscard]] NativeBufferAllocation allocate_native_buffer(
        size_t byte_size, VkBufferUsageFlags usage,
        VkBufferUsageFlags2KHR usage2, bool host_visible,
        MemoryInitPolicy init) noexcept;
    void destroy_native_buffer(NativeBufferAllocation &allocation) noexcept;
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
