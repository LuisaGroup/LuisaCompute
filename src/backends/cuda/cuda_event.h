#pragma once

#include <cuda.h>
#include <luisa/core/stl/vector.h>

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include "../common/vulkan_instance.h"

namespace luisa::compute::cuda {

class CUDAEventManager;

class CUDAEvent {

    friend class CUDAEventManager;

private:
    VkDevice _device;
    VkSemaphore _vk_semaphore;
    CUexternalSemaphore _cuda_semaphore;

public:
    CUDAEvent(VkDevice device,
              VkSemaphore vk_semaphore,
              CUexternalSemaphore cuda_semaphore) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _cuda_semaphore; }
    void notify(uint64_t value) noexcept;
    void signal(CUstream stream, uint64_t value) noexcept;
    void wait(CUstream stream, uint64_t value) noexcept;
    void synchronize(uint64_t value) noexcept;
    [[nodiscard]] uint64_t signaled_value() noexcept;
    [[nodiscard]] bool is_completed(uint64_t value) noexcept;
};

class CUDAEventManager {

private:
    luisa::shared_ptr<VulkanInstance> _instance;
    VkPhysicalDevice _physical_device{nullptr};
    VkDevice _device{nullptr};
    uint64_t _addr_vkGetSemaphoreHandle{0u};
    std::atomic<size_t> _count{0u};

public:
    explicit CUDAEventManager(const CUuuid &uuid) noexcept;
    ~CUDAEventManager() noexcept;
    CUDAEventManager(CUDAEventManager &&) noexcept = delete;
    CUDAEventManager(const CUDAEventManager &) noexcept = delete;
    CUDAEventManager &operator=(CUDAEventManager &&) noexcept = delete;
    CUDAEventManager &operator=(const CUDAEventManager &) noexcept = delete;
    [[nodiscard]] CUDAEvent *create() noexcept;
    void destroy(CUDAEvent *event) noexcept;
};

}// namespace luisa::compute::cuda

#else

#error You cannot use CUDA backend without Vulkan. 😢😢😢. For Windows users, get Vulkan SDK from https://www.lunarg.com/vulkan-sdk/

#endif
