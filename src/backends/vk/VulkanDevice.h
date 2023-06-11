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
#include "vulkan/vulkan.h"
#include <algorithm>
#include <assert.h>
#include <luisa/vstl/common.h>

namespace vks
{
struct VulkanDevice
{
	/** @brief Physical device representation */
	VkPhysicalDevice physicalDevice;
	/** @brief Logical device representation (application's view of the device) */
	VkDevice logicalDevice;
	/** @brief Properties of the physical device including limits that the application can check against */
	VkPhysicalDeviceProperties properties;
	/** @brief Features of the physical device that an application can use to check if a feature is supported */
	VkPhysicalDeviceFeatures features;
	/** @brief Features that have been enabled for use on the physical device */
	VkPhysicalDeviceFeatures enabledFeatures;
	/** @brief Memory types and heaps of the physical device */
	VkPhysicalDeviceMemoryProperties memoryProperties;
	/** @brief Queue family properties of the physical device */
	vstd::vector<VkQueueFamilyProperties> queueFamilyProperties;
	/** @brief List of extensions supported by the device */
	vstd::unordered_set<vstd::string> supportedExtensions;
	/** @brief Default command pool for the graphics queue family index */
	VkCommandPool commandPool = VK_NULL_HANDLE;
	/** @brief Set to true when the debug marker extension is detected */
	bool enableDebugMarkers = false;
	/** @brief Contains queue family indices */
	struct
	{
		uint32_t graphics;
		uint32_t compute;
		uint32_t transfer;
	} queueFamilyIndices;
	operator VkDevice() const
	{
		return logicalDevice;
	};
	explicit VulkanDevice(VkPhysicalDevice physicalDevice);
	~VulkanDevice();
	uint32_t        getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr) const;
	uint32_t        getQueueFamilyIndex(VkQueueFlags queueFlags) const;
	VkResult        createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, vstd::span<const vstd::string> enabledExtensions, void *pNextChain, bool useSwapChain = true, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
	void            copyBuffer(vks::Buffer *src, vks::Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion = nullptr);
	VkCommandPool   createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);
	VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin = false);
	void            flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);
	void            flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);
	bool            extensionSupported(vstd::string_view extension);
	VkFormat        getSupportedDepthFormat(bool checkSamplingSupport);
};
}        // namespace vks

