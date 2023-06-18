#pragma once
#include <vulkan/vulkan.h>
#include <luisa/core/logging.h>
#include "VulkanTools.h"

namespace lc::vk {
#ifdef NDEBUG
#define VK_CHECK_RESULT(f) (f)
#else
#define VK_CHECK_RESULT(f)                                                                                    \
    {                                                                                                         \
        VkResult res = (f);                                                                                   \
        if (res != VK_SUCCESS) [[unlikely]] {                                                                 \
            LUISA_ERROR("Fatal : VkResult is \"{}\" in {} at line {}", vks::tools::errorString(res), __FILE__, __LINE__); \
        }                                                                                                     \
    }
#endif
}// namespace lc::vk
