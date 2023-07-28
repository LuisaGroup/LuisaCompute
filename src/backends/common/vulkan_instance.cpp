#include <vulkan/vulkan.h>

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_APPLE)
#include <vulkan/vulkan_macos.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#else
#error "Unsupported platform"
#endif

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/logging.h>

#include "vulkan_instance.h"

namespace luisa::compute {

namespace detail {
#ifndef NDEBUG
static VkBool32 vulkan_validation_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                           VkDebugUtilsMessageTypeFlagsEXT /* types */,
                                           const VkDebugUtilsMessengerCallbackDataEXT *data,
                                           void * /* user data */) noexcept {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LUISA_WARNING("Vulkan Validation Error: {}", data->pMessage);
    } else {
        LUISA_WARNING("Vulkan Validation Warning: {}", data->pMessage);
    }
    return VK_FALSE;
}
#endif
}// namespace detail

VulkanInstance::VulkanInstance() noexcept {

    luisa::vector<const char *> extensions;
    extensions.reserve(4u);
    extensions.emplace_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(LUISA_PLATFORM_WINDOWS)
    extensions.emplace_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(LUISA_PLATFORM_APPLE)
    extensions.emplace_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#else
    extensions.emplace_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

#ifndef NDEBUG
    static constexpr std::array validation_layers{"VK_LAYER_KHRONOS_validation"};
    auto supports_validation = [] {
        auto layer_count = 0u;
        LUISA_CHECK_VULKAN(vkEnumerateInstanceLayerProperties(&layer_count, nullptr));
        luisa::vector<VkLayerProperties> available_layers(layer_count);
        LUISA_CHECK_VULKAN(vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data()));
        for (auto layer : validation_layers) {
            if (!std::any_of(available_layers.cbegin(), available_layers.cend(),
                             [layer = luisa::string_view{layer}](auto available) noexcept {
                                 return layer == available.layerName;
                             })) {
                return false;
            }
        }
        return true;
    }();
    if (supports_validation) {
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "LuisaCompute Vulkan Extension";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "LuisaCompute";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = LUISA_REQUIRED_VULKAN_VERSION;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

#ifdef LUISA_PLATFORM_APPLE
#define ENUMERATE_PORTABILITY_BIT (0x01)
    create_info.flags = ENUMERATE_PORTABILITY_BIT;
#endif

#ifndef NDEBUG
    VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
    if (supports_validation) {
        debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_create_info.pfnUserCallback = detail::vulkan_validation_callback;
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();
        create_info.pNext = &debug_create_info;
    }
#endif
    LUISA_CHECK_VULKAN(vkCreateInstance(&create_info, nullptr, &_instance));

#ifndef NDEBUG
    if (supports_validation) {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(_instance, "vkCreateDebugUtilsMessengerEXT");
        LUISA_ASSERT(func != nullptr, "Failed to load vkCreateDebugUtilsMessengerEXT.");
        LUISA_CHECK_VULKAN(func(_instance, &debug_create_info, nullptr, &_debug_messenger));
    }
#endif
    LUISA_VERBOSE_WITH_LOCATION("Created vulkan instance.");
}

VulkanInstance::~VulkanInstance() noexcept {
#ifndef NDEBUG
    if (_instance != nullptr) {
        if (auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(_instance, "vkDestroyDebugUtilsMessengerEXT")) {
            func(_instance, _debug_messenger, nullptr);
        }
    }
#endif
    vkDestroyInstance(_instance, nullptr);
    LUISA_VERBOSE_WITH_LOCATION("Destroyed vulkan instance.");
}

luisa::shared_ptr<VulkanInstance> VulkanInstance::retain() noexcept {
    static std::mutex mutex;
    static luisa::weak_ptr<VulkanInstance> weak_instance;
    std::scoped_lock lock{mutex};
    if (auto ptr = weak_instance.lock()) { return ptr; }
    luisa::shared_ptr<VulkanInstance> instance{
        new (luisa::allocate_with_allocator<VulkanInstance>()) VulkanInstance,
        [](auto *ptr) noexcept { luisa::delete_with_allocator(ptr); }};
    weak_instance = instance;
    return instance;
}

}// namespace luisa::compute
