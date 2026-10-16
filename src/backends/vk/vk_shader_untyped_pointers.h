#pragma once
// VK_KHR_shader_untyped_pointers was released in Vulkan 1.4.325. The bundled
// volk headers (VK_HEADER_VERSION 321) and current system SDKs predate it, so
// vendor the definitions here. Skipped automatically when the build already
// provides them (headers >= 1.4.325).
#ifndef VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME
#define VK_KHR_SHADER_UNTYPED_POINTERS_SPEC_VERSION 1
#define VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME "VK_KHR_shader_untyped_pointers"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR \
    ((VkStructureType)1000387000)
typedef struct VkPhysicalDeviceShaderUntypedPointersFeaturesKHR {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderUntypedPointers;
} VkPhysicalDeviceShaderUntypedPointersFeaturesKHR;
#endif
