/*
* Vulkan device class
*
* Encapsulates a physical Vulkan device and its logical representation
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "VulkanBuffer.h"
#include "VulkanTools.h"
#include <volk.h>
#include <algorithm>
#include <assert.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>

namespace vks {
struct VulkanDevice {
    /** @brief Physical device representation */
    VkPhysicalDevice physical_device;
    /** @brief Logical device representation (application's view of the device) */
    VkDevice logical_device;
    /** @brief Properties of the physical device including limits that the application can check against */
    VkPhysicalDeviceProperties properties;
    /** @brief Features of the physical device that an application can use to check if a feature is supported */
    VkPhysicalDeviceVulkan12Features features_12;
    VkPhysicalDeviceFeatures2 features;
    /** @brief Features that have been enabled for use on the physical device */
    VkPhysicalDeviceFeatures enabled_features;
    /** @brief Memory types and heaps of the physical device */
    VkPhysicalDeviceMemoryProperties memory_properties;
    /** @brief Queue family properties of the physical device */
    vstd::vector<VkQueueFamilyProperties> queue_family_properties;
    /** @brief List of extensions supported by the device */
    vstd::unordered_set<vstd::string> supported_extensions;
    /** @brief Set to true when the debug marker extension is detected */
    bool enable_debug_markers = false;
    /** @brief Contains queue family indices */
    struct
    {
        uint32_t graphics;
        uint32_t compute;
        uint32_t transfer;
        uint32_t sparse{VK_QUEUE_FAMILY_IGNORED};
    } queue_family_indices;
    operator VkDevice() const {
        return logical_device;
    }
    explicit VulkanDevice(VkPhysicalDevice physical_device);
    ~VulkanDevice();
    uint32_t get_memory_type(uint32_t type_bits, VkMemoryPropertyFlags properties, VkBool32 *mem_type_found = nullptr) const;
    uint32_t get_queue_family_index(VkQueueFlags queueFlags) const;
    VkResult create_logical_device(VkPhysicalDeviceFeatures &enabled_features, vstd::span<const vstd::string> enabled_extensions, void *p_next_chain, bool use_swap_chain = true, VkQueueFlags requested_queue_types = VK_QUEUE_FLAG_BITS_MAX_ENUM);
    bool extension_supported(vstd::string_view extension);
    VkFormat get_supported_depth_format(bool check_sampling_support);
    static void init_volk(luisa::filesystem::path const &custom_path, luisa::string_view lib_name);
};
}// namespace vks
