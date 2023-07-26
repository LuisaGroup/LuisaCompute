#pragma once

#include <vulkan/vulkan_core.h>

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>

namespace luisa::compute {

struct VulkanDeviceUUID;

class LC_BACKEND_API VulkanSwapchain {

public:
    class Impl;

private:
    luisa::unique_ptr<Impl> _impl;

public:
    VulkanSwapchain(const VulkanDeviceUUID &device_uuid, uint64_t window_handle,
                    uint width, uint height, bool allow_hdr,
                    bool vsync, uint back_buffer_count,
                    luisa::span<const char *const> required_device_extensions) noexcept;
    ~VulkanSwapchain() noexcept;
    [[nodiscard]] VkDevice device() const noexcept;
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept;
    [[nodiscard]] VkQueue queue() const noexcept;
    [[nodiscard]] VkExtent2D extent() const noexcept;
    [[nodiscard]] VkSurfaceFormatKHR format() const noexcept;
    [[nodiscard]] VkCommandPool command_pool() const noexcept;
    [[nodiscard]] size_t back_buffer_count() const noexcept;
    [[nodiscard]] bool is_hdr() const noexcept;
    void wait_for_fence() noexcept;
    void present(VkSemaphore wait, VkSemaphore signal,
                 VkImageView image, VkImageLayout image_layout) noexcept;
};

#ifdef LUISA_PLATFORM_APPLE
void *cocoa_window_content_view(uint64_t window_handle) noexcept;
#endif

}// namespace luisa::compute

