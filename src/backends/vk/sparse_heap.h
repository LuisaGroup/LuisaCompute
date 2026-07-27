#pragma once

#include <mutex>

#include <volk.h>

#include "vk_mem_alloc.h"

namespace lc::vk {

class Device;

struct SparseHeapMemory {
    VkDeviceMemory memory{};
    VkDeviceSize offset{};
};

// A public sparse heap is created before the resource that will consume it, so
// its Vulkan memory requirements are not known yet. Defer the actual allocation
// until the first bind, then require every later resource to be compatible with
// the selected memory type and allocation alignment.
class VulkanSparseHeap {
private:
    Device *_device;
    size_t _byte_size;
    std::mutex _mutex;
    VmaAllocation _allocation{};
    VmaAllocationInfo _allocation_info{};

public:
    VulkanSparseHeap(Device *device, size_t byte_size) noexcept;
    ~VulkanSparseHeap() noexcept;
    VulkanSparseHeap(VulkanSparseHeap const &) = delete;
    VulkanSparseHeap(VulkanSparseHeap &&) = delete;
    VulkanSparseHeap &operator=(VulkanSparseHeap const &) = delete;
    VulkanSparseHeap &operator=(VulkanSparseHeap &&) = delete;

    [[nodiscard]] SparseHeapMemory acquire(
        VkMemoryRequirements requirements,
        VkDeviceSize required_binding_size) noexcept;
    [[nodiscard]] auto byte_size() const noexcept { return _byte_size; }
};

}// namespace lc::vk
