/*
* Vulkan device class
*
* Encapsulates a physical Vulkan device and its logical representation
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
// SRS - Enable beta extensions and make VK_KHR_portability_subset visible
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include "VulkanDevice.h"
#include <luisa/core/logging.h>
#include "log.h"
#include "device.h"
namespace vks {
/**
	* Default constructor
	*
	* @param physicalDevice Physical device that is to be used
	*/
VulkanDevice::VulkanDevice(VkPhysicalDevice physicalDevice) {
    assert(physicalDevice);
    this->physicalDevice = physicalDevice;

    // Store Properties features, limits and properties of the physical device for later use
    // Device properties also contain limits and sparse properties
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    // Features should be checked by the examples before using them
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);
    // Memory properties are used regularly for creating all kinds of buffers
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    // Queue family properties, used for setting up requested queues upon device creation
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    assert(queueFamilyCount > 0);
    queueFamilyProperties.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

    // Get list of supported extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    if (extCount > 0) {
        vstd::vector<VkExtensionProperties> extensions(extCount);
        if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, &extensions.front()) == VK_SUCCESS) {
            for (auto ext : extensions) {
                supportedExtensions.emplace(ext.extensionName);
            }
        }
    }
}

/** 
	* Default destructor
	*
	* @note Frees the logical device
	*/
VulkanDevice::~VulkanDevice() {
    if (logicalDevice) {
        vkDestroyDevice(logicalDevice, lc::vk::Device::alloc_callbacks());
    }
}

/**
	* Get the index of a memory type that has all the requested property bits set
	*
	* @param typeBits Bit mask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
	* @param properties Bit mask of properties for the memory type to request
	* @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
	* 
	* @return Index of the requested memory type
	*
	* @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested properties
	*/
uint32_t VulkanDevice::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound) const {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeBits & 1) == 1) {
            if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                if (memTypeFound) {
                    *memTypeFound = true;
                }
                return i;
            }
        }
        typeBits >>= 1;
    }

    if (memTypeFound) {
        *memTypeFound = false;
        return 0;
    } else {
        LUISA_ERROR("Could not find a matching memory type");
    }
}

/**
	* Get the index of a queue family that supports the requested queue flags
	* SRS - support VkQueueFlags parameter for requesting multiple flags vs. VkQueueFlagBits for a single flag only
	*
	* @param queueFlags Queue flags to find a queue family index for
	*
	* @return Index of the queue family index that matches the flags
	*
	* @throw Throws an exception if no queue family index could be found that supports the requested flags
	*/
uint32_t VulkanDevice::getQueueFamilyIndex(VkQueueFlags queueFlags) const {
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if ((queueFlags & VK_QUEUE_COMPUTE_BIT) == queueFlags) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
            if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)) {
                return i;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if ((queueFlags & VK_QUEUE_TRANSFER_BIT) == queueFlags) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
            if ((queueFamilyProperties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)) {
                return i;
            }
        }
    }

    // For other queue types or if no separate compute queue is present, return the first one to support the requested flags
    for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
        if ((queueFamilyProperties[i].queueFlags & queueFlags) == queueFlags) {
            return i;
        }
    }

    LUISA_ERROR("Could not find a matching queue family index");
}

/**
	* Create the logical device based on the assigned physical device, also gets default queue family indices
	*
	* @param enabledFeatures Can be used to enable certain features upon device creation
	* @param pNextChain Optional chain of pointer to extension structures
	* @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
	* @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device  
	*
	* @return VkResult of the device creation call
	*/
VkResult VulkanDevice::createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, vstd::span<const vstd::string> enabledExtensions, void *pNextChain, bool useSwapChain, VkQueueFlags requestedQueueTypes) {
    // Desired queues need to be requested upon logical device creation
    // Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
    // requests different queue types

    vstd::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};

    // Get queue family indices for the requested queue family types
    // Note that the indices may overlap depending on the implementation

    const float defaultQueuePriority(0.0f);

    // Graphics queue
    if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT) {
        queueFamilyIndices.graphics = getQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT);
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndices.graphics;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &defaultQueuePriority;
        queueCreateInfos.push_back(queueInfo);
    } else {
        queueFamilyIndices.graphics = 0;
    }

    // Dedicated compute queue
    if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT) {
        queueFamilyIndices.compute = getQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT);
        if (queueFamilyIndices.compute != queueFamilyIndices.graphics) {
            // If compute family index differs, we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queueFamilyIndices.compute;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &defaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    } else {
        // Else we use the same queue
        queueFamilyIndices.compute = queueFamilyIndices.graphics;
    }

    // Dedicated transfer queue
    if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT) {
        queueFamilyIndices.transfer = getQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT);
        if ((queueFamilyIndices.transfer != queueFamilyIndices.graphics) && (queueFamilyIndices.transfer != queueFamilyIndices.compute)) {
            // If transfer family index differs, we need an additional queue create info for the transfer queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queueFamilyIndices.transfer;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &defaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    } else {
        // Else we use the same queue
        queueFamilyIndices.transfer = queueFamilyIndices.graphics;
    }

    // Create the logical device representation
    vstd::vector<const char *> deviceExtensions;
    vstd::push_back_func(
        deviceExtensions,
        enabledExtensions.size(),
        [&](size_t i) {
            return enabledExtensions[i].c_str();
        });
		
    if (useSwapChain) {
        // If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

    // If a pNext(Chain) has been passed, we need to add it to the device creation info
    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    if (pNextChain) {
        physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        physicalDeviceFeatures2.features = enabledFeatures;
        physicalDeviceFeatures2.pNext = pNextChain;
        deviceCreateInfo.pEnabledFeatures = nullptr;
        deviceCreateInfo.pNext = &physicalDeviceFeatures2;
    }

    // Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
    if (extensionSupported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        deviceExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
        enableDebugMarkers = true;
    }

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK)) && defined(VK_KHR_portability_subset)
    // SRS - When running on iOS/macOS with MoltenVK and VK_KHR_portability_subset is defined and supported by the device, enable the extension
    if (extensionSupported(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif

    if (deviceExtensions.size() > 0) {
        for (const char *enabledExtension : deviceExtensions) {
            if (!extensionSupported(enabledExtension)) {
                LUISA_ERROR("Enabled device extension \"{}\" is not present at device level", enabledExtension);
            }
        }

        deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }

    this->enabledFeatures = enabledFeatures;

    VkResult result = vkCreateDevice(physicalDevice, &deviceCreateInfo, lc::vk::Device::alloc_callbacks(), &logicalDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    // Create a default command pool for graphics command buffers

    return result;
}

bool VulkanDevice::extensionSupported(vstd::string_view extension) {
    return supportedExtensions.find(extension) != supportedExtensions.end();
}

VkFormat VulkanDevice::getSupportedDepthFormat(bool checkSamplingSupport) {
    // All depth formats may be optional, so we need to find a suitable depth format to use
    vstd::vector<VkFormat> depthFormats = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D16_UNORM};
    for (auto &format : depthFormats) {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
        // Format must support depth stencil attachment for optimal tiling
        if (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            if (checkSamplingSupport) {
                if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
                    continue;
                }
            }
            return format;
        }
    }
    LUISA_ERROR("Could not find a matching depth format");
}

};// namespace vks

