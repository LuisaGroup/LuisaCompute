//
// Created by Mike on 3/22/2023.
//

#pragma once

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#include <core/basic_types.h>
#include <core/stl/memory.h>
#include <cstring>

#define LUISA_CHECK_VULKAN(x)                            \
    do {                                                 \
        auto ret = x;                                    \
        if (ret != VK_SUCCESS) [[unlikely]] {            \
            if (ret == VK_ERROR_OUT_OF_DATE_KHR ||       \
                ret == VK_SUBOPTIMAL_KHR) [[likely]] {   \
                LUISA_WARNING_WITH_LOCATION(             \
                    "Vulkan call `" #x "` returned {}.", \
                    string_VkResult(ret));               \
            } else [[unlikely]] {                        \
                LUISA_ERROR_WITH_LOCATION(               \
                    "Vulkan call `" #x "` failed: {}.",  \
                    string_VkResult(ret));               \
            }                                            \
        }                                                \
    } while (false)

namespace luisa::compute {

class VulkanSwapchain {

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

}// namespace luisa::compute
