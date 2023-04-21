//
// Created by Mike on 3/22/2023.
//

#pragma once

#include <vulkan/vulkan_core.h>

#include <core/basic_types.h>
#include <core/stl/memory.h>
#include <core/stl/string.h>
#include <core/magic_enum.h>

#define LUISA_CHECK_VULKAN(x)                            \
    do {                                                 \
        auto ret = x;                                    \
        if (ret != VK_SUCCESS) [[unlikely]] {            \
            if (ret > 0) [[likely]] {                    \
                LUISA_WARNING_WITH_LOCATION(             \
                    "Vulkan call `" #x "` returned {}.", \
                    ::luisa::to_string(ret));            \
            } else [[unlikely]] {                        \
                LUISA_ERROR_WITH_LOCATION(               \
                    "Vulkan call `" #x "` failed: {}.",  \
                    ::luisa::to_string(ret));            \
            }                                            \
        }                                                \
    } while (false)

namespace luisa::compute {

class LC_BACKEND_API VulkanSwapchain {

public:
    class Impl;
    struct DeviceUUID {
        uint8_t bytes[16];

        [[nodiscard]] auto operator==(const DeviceUUID &rhs) const noexcept {
            return std::memcmp(bytes, rhs.bytes, sizeof(bytes)) == 0;
        }
    };

private:
    luisa::unique_ptr<Impl> _impl;

public:
    VulkanSwapchain(DeviceUUID device_uuid, uint64_t window_handle,
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
