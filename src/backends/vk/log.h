#pragma once
#include <volk.h>
#include <luisa/core/logging.h>
#include "VulkanTools.h"

namespace lc::vk {
#define VK_CHECK_RESULT(f)                                                              \
    do {                                                                                \
        const VkResult vk_check_result = (f);                                           \
        if (vk_check_result != VK_SUCCESS) [[unlikely]] {                               \
            LUISA_ERROR(                                                               \
                "Fatal: Vulkan call '{}' returned \"{}\" in {} at line {}.",            \
                #f, vks::tools::error_string(vk_check_result), __FILE__, __LINE__);      \
        }                                                                               \
    } while (false)
}// namespace lc::vk
