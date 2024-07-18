#include <array>
#include <algorithm>
#include <luisa/core/platform.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/rhi/pixel.h>

#include "vulkan_instance.h"
#include <luisa/backends/common/vulkan_swapchain.h>

namespace luisa::compute {

// source:
// #version 450
// layout(location = 0) in vec2 inPosition;
// layout(location = 1) in vec3 inColor;
// layout(location = 0) out vec2 fragTexCoord;
// void main() {
//     gl_Position = vec4(inPosition * 2.0 - 1.0, 0.0, 1.0);
//     fragTexCoord = inPosition;
// }
static const std::array vulkan_swapchain_screen_shader_vertex_bytecode = {
    0x07230203u, 0x00010300u, 0x000d000bu, 0x00000026u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0008000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000du, 0x00000012u, 0x00000020u,
    0x00050048u, 0x0000000bu, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00050048u, 0x0000000bu, 0x00000001u,
    0x0000000bu, 0x00000001u, 0x00050048u, 0x0000000bu, 0x00000002u, 0x0000000bu, 0x00000003u, 0x00050048u,
    0x0000000bu, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00030047u, 0x0000000bu, 0x00000002u, 0x00040047u,
    0x00000012u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x00000020u, 0x0000001eu, 0x00000000u, 0x00020013u,
    0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u,
    0x00000007u, 0x00000006u, 0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u, 0x0004002bu,
    0x00000008u, 0x00000009u, 0x00000001u, 0x0004001cu, 0x0000000au, 0x00000006u, 0x00000009u, 0x0006001eu,
    0x0000000bu, 0x00000007u, 0x00000006u, 0x0000000au, 0x0000000au, 0x00040020u, 0x0000000cu, 0x00000003u,
    0x0000000bu, 0x0004003bu, 0x0000000cu, 0x0000000du, 0x00000003u, 0x00040015u, 0x0000000eu, 0x00000020u,
    0x00000001u, 0x0004002bu, 0x0000000eu, 0x0000000fu, 0x00000000u, 0x00040017u, 0x00000010u, 0x00000006u,
    0x00000002u, 0x00040020u, 0x00000011u, 0x00000001u, 0x00000010u, 0x0004003bu, 0x00000011u, 0x00000012u,
    0x00000001u, 0x0004002bu, 0x00000006u, 0x00000014u, 0x40000000u, 0x0004002bu, 0x00000006u, 0x00000016u,
    0x3f800000u, 0x0004002bu, 0x00000006u, 0x00000019u, 0x00000000u, 0x00040020u, 0x0000001du, 0x00000003u,
    0x00000007u, 0x00040020u, 0x0000001fu, 0x00000003u, 0x00000010u, 0x0004003bu, 0x0000001fu, 0x00000020u,
    0x00000003u, 0x0005002cu, 0x00000010u, 0x00000025u, 0x00000016u, 0x00000016u, 0x00050036u, 0x00000002u,
    0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du, 0x00000010u, 0x00000013u,
    0x00000012u, 0x0005008eu, 0x00000010u, 0x00000015u, 0x00000013u, 0x00000014u, 0x00050083u, 0x00000010u,
    0x00000018u, 0x00000015u, 0x00000025u, 0x00050051u, 0x00000006u, 0x0000001au, 0x00000018u, 0x00000000u,
    0x00050051u, 0x00000006u, 0x0000001bu, 0x00000018u, 0x00000001u, 0x00070050u, 0x00000007u, 0x0000001cu,
    0x0000001au, 0x0000001bu, 0x00000019u, 0x00000016u, 0x00050041u, 0x0000001du, 0x0000001eu, 0x0000000du,
    0x0000000fu, 0x0003003eu, 0x0000001eu, 0x0000001cu, 0x0003003eu, 0x00000020u, 0x00000013u, 0x000100fdu,
    0x00010038u};

// source:
// #version 450
// layout(binding = 0) uniform texture2D tex;
// layout(binding = 1) uniform sampler sam;
// layout(location = 0) in vec2 fragTexCoord;
// layout(location = 0) out vec4 outColor;
// void main() {
//     outColor = texture(sampler2D(tex, sam), fragTexCoord);
// }
static const std::array vulkan_swapchain_screen_shader_fragment_bytecode = {
    0x07230203u, 0x00010300u, 0x000d000bu, 0x00000019u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00000016u, 0x00030010u,
    0x00000004u, 0x00000007u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x0000000cu,
    0x00000022u, 0x00000000u, 0x00040047u, 0x0000000cu, 0x00000021u, 0x00000000u, 0x00040047u, 0x00000010u,
    0x00000022u, 0x00000000u, 0x00040047u, 0x00000010u, 0x00000021u, 0x00000001u, 0x00040047u, 0x00000016u,
    0x0000001eu, 0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
    0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
    0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x00090019u, 0x0000000au,
    0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00040020u,
    0x0000000bu, 0x00000000u, 0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000000u, 0x0002001au,
    0x0000000eu, 0x00040020u, 0x0000000fu, 0x00000000u, 0x0000000eu, 0x0004003bu, 0x0000000fu, 0x00000010u,
    0x00000000u, 0x0003001bu, 0x00000012u, 0x0000000au, 0x00040017u, 0x00000014u, 0x00000006u, 0x00000002u,
    0x00040020u, 0x00000015u, 0x00000001u, 0x00000014u, 0x0004003bu, 0x00000015u, 0x00000016u, 0x00000001u,
    0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003du,
    0x0000000au, 0x0000000du, 0x0000000cu, 0x0004003du, 0x0000000eu, 0x00000011u, 0x00000010u, 0x00050056u,
    0x00000012u, 0x00000013u, 0x0000000du, 0x00000011u, 0x0004003du, 0x00000014u, 0x00000017u, 0x00000016u,
    0x00050057u, 0x00000007u, 0x00000018u, 0x00000013u, 0x00000017u, 0x0003003eu, 0x00000009u, 0x00000018u,
    0x000100fdu, 0x00010038u};

#ifdef LUISA_PLATFORM_APPLE
void *cocoa_window_content_view(uint64_t window_handle) noexcept;
#endif

class VulkanSwapchain::Impl {

private:
    // instance, must go first to ensure destruction order
    luisa::shared_ptr<VulkanInstance> _instance;

    // surface
    VkSurfaceKHR _surface{nullptr};

    // device
    VkPhysicalDevice _physical_device{nullptr};
    VkDevice _device{nullptr};

    // queue
    VkQueue _queue{nullptr};

    // swapchain
    VkSwapchainKHR _swapchain{nullptr};
    VkSurfaceFormatKHR _swapchain_format{};
    VkExtent2D _swapchain_extent{};
    luisa::vector<VkImage> _swapchain_images;
    luisa::vector<VkImageView> _swapchain_image_views;
    luisa::vector<VkFramebuffer> _swapchain_framebuffers;

    // render pass
    VkRenderPass _render_pass{nullptr};
    VkDescriptorSetLayout _descriptor_set_layout{nullptr};
    VkPipelineLayout _pipeline_layout{nullptr};
    VkPipeline _graphics_pipeline{nullptr};
    VkSampler _texture_sampler{nullptr};

    VkBuffer _vertex_buffer{nullptr};
    VkDeviceMemory _vertex_buffer_memory{nullptr};

    // descriptor
    VkDescriptorPool _descriptor_pool{nullptr};
    luisa::vector<VkDescriptorSet> _descriptor_sets;
    luisa::vector<VkDescriptorImageInfo> _cached_image_infos;

    // command
    VkCommandPool _command_pool{nullptr};
    luisa::vector<VkCommandBuffer> _command_buffers;

    // sync objects
    luisa::vector<VkFence> _in_flight_fences;
    luisa::vector<VkSemaphore> _image_available_semaphores;
    luisa::vector<VkSemaphore> _render_finished_semaphores;
    size_t _current_frame{0u};

private:
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities{};
        luisa::vector<VkSurfaceFormatKHR> formats;
        luisa::vector<VkPresentModeKHR> present_modes;
    };

    [[nodiscard]] auto _query_swapchain_support(VkPhysicalDevice device) noexcept {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _surface, &details.capabilities);
        auto format_count = 0u;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count, nullptr);
        if (format_count != 0u) {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, _surface, &format_count, details.formats.data());
        }
        auto present_mode_count = 0u;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &present_mode_count, nullptr);
        if (present_mode_count != 0u) {
            details.present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, _surface, &present_mode_count, details.present_modes.data());
        }
        return details;
    }

    void _create_surface(uint64_t display_handle, uint64_t window_handle) noexcept {
#if defined(LUISA_PLATFORM_WINDOWS)
        VkWin32SurfaceCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        create_info.hwnd = reinterpret_cast<HWND>(window_handle);
        create_info.hinstance = GetModuleHandle(nullptr);
        LUISA_CHECK_VULKAN(vkCreateWin32SurfaceKHR(_instance->handle(), &create_info, nullptr, &_surface));
#elif defined(LUISA_PLATFORM_APPLE)
        VkMacOSSurfaceCreateInfoMVK create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK;
        create_info.pView = cocoa_window_content_view(window_handle);
        LUISA_CHECK_VULKAN(vkCreateMacOSSurfaceMVK(_instance->handle(), &create_info, nullptr, &_surface));
#else
        static std::once_flag set_xlib_error_handler;
        std::call_once(set_xlib_error_handler, [] {
            XSetErrorHandler([](Display *display, XErrorEvent *error) noexcept {
                char buffer[256] = {};
                XGetErrorText(display, error->error_code, buffer, sizeof(buffer));
                LUISA_WARNING_WITH_LOCATION("Xlib error: {}", buffer);
                return 0;
            });
        });
        auto create_surface_xlib = [&] {
            VkXlibSurfaceCreateInfoKHR create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
            create_info.dpy = display_handle ? reinterpret_cast<Display *>(display_handle) : XOpenDisplay(nullptr);
            create_info.window = static_cast<Window>(window_handle);
            LUISA_CHECK_VULKAN(vkCreateXlibSurfaceKHR(_instance->handle(), &create_info, nullptr, &_surface));
        };
#if LUISA_ENABLE_WAYLAND
        if (window_handle & 0xffff'ffff'0000'0000ull) {// 64-bit pointer, so likely wayland
            VkWaylandSurfaceCreateInfoKHR create_info_wl{};
            create_info_wl.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
            create_info_wl.display = display_handle ? reinterpret_cast<wl_display *>(display_handle) : wl_display_connect(nullptr);
            create_info_wl.surface = reinterpret_cast<wl_surface *>(window_handle);
            LUISA_CHECK_VULKAN(vkCreateWaylandSurfaceKHR(_instance->handle(), &create_info_wl, nullptr, &_surface));
        } else {// X uses 32-bit IDs
            create_surface_xlib();
        }
#else
        create_surface_xlib();
#endif
#endif
    }

    [[nodiscard]] static auto _is_hdr_colorspace(VkColorSpaceKHR colorspace) noexcept {
        return colorspace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
    }

    [[nodiscard]] static auto _colorspace_name(VkColorSpaceKHR colorspace) noexcept {
        switch (colorspace) {
            case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
                return "sRGB (non-linear)";
            case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
                return "Display P3 (non-linear)";
            case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
                return "Extended sRGB (linear)";
            case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
                return "Display P3 (linear)";
            case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
                return "DCI P3 (non-linear)";
            case VK_COLOR_SPACE_BT709_LINEAR_EXT:
                return "BT709 (linear)";
            case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
                return "BT709 (non-linear)";
            case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
                return "BT2020 (linear)";
            case VK_COLOR_SPACE_HDR10_ST2084_EXT:
                return "HDR10 (ST2084)";
            case VK_COLOR_SPACE_DOLBYVISION_EXT:
                return "Dolby Vision";
            case VK_COLOR_SPACE_HDR10_HLG_EXT:
                return "HDR10 (HLG)";
            case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
                return "Adobe RGB (linear)";
            case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
                return "Adobe RGB (non-linear)";
            case VK_COLOR_SPACE_PASS_THROUGH_EXT:
                return "Pass-through";
            case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
                return "Extended sRGB (non-linear)";
            case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
                return "Display Native";
            default:
                break;
        }
        return "Unknown";
    }

    void _create_device(const VulkanDeviceUUID &device_uuid,
                        luisa::span<const char *const> required_device_extensions,
                        bool allow_hdr) noexcept {

        luisa::vector<const char *> device_extensions;
        device_extensions.reserve(required_device_extensions.size() + 2u);
        device_extensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

#ifdef LUISA_PLATFORM_APPLE
        device_extensions.emplace_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif
        for (auto ext : required_device_extensions) {
#ifdef LUISA_PLATFORM_APPLE
            if (luisa::string_view{ext} != VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME &&
                luisa::string_view{ext} != VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
                device_extensions.emplace_back(ext);
            }
#else
            if (luisa::string_view{ext} != VK_KHR_SWAPCHAIN_EXTENSION_NAME) {
                device_extensions.emplace_back(ext);
            }
#endif
        }

        auto check_properties = [&device_uuid](auto device) noexcept {
            VkPhysicalDeviceIDProperties id_properties{};
            id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            VkPhysicalDeviceProperties2 properties2{};
            properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties2.pNext = &id_properties;

            vkGetPhysicalDeviceProperties2(device, &properties2);
            if (properties2.properties.apiVersion < LUISA_REQUIRED_VULKAN_VERSION) { return false; }
            if (device_uuid == VulkanDeviceUUID{} /* any uuid */) { return true; }
            return std::memcmp(id_properties.deviceUUID, device_uuid.bytes, sizeof(device_uuid.bytes)) == 0;
        };

        auto check_extensions = [&device_extensions](auto device) noexcept {
            auto extension_count = 0u;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);
            luisa::vector<VkExtensionProperties> available_extensions(extension_count);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());
            luisa::unordered_set<luisa::string_view> required_extensions(device_extensions.begin(), device_extensions.end());
            for (const auto &extension : available_extensions) {
                required_extensions.erase(extension.extensionName);
            }
            return required_extensions.empty();
        };

        auto find_queue_family = [surface = _surface](auto device) noexcept -> luisa::optional<uint32_t> {
            auto queue_family_count = 0u;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
            luisa::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
            for (auto i = 0u; i < queue_family_count; i++) {
                if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    VkBool32 present_support = false;
                    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
                    if (present_support) { return i; }
                }
            }
            return luisa::nullopt;
        };

        auto check_swapchain_support = [this](auto device, bool requires_hdr) noexcept {
            auto details = _query_swapchain_support(device);
            if (requires_hdr) {
                details.formats.erase(
                    std::remove_if(details.formats.begin(), details.formats.end(), [](auto format) noexcept {
                        return _is_hdr_colorspace(format.colorSpace);
                    }),
                    details.formats.end());
            }
            return !details.formats.empty() && !details.present_modes.empty();
        };

        // find the suitable physical device
        auto device_count = 0u;
        LUISA_CHECK_VULKAN(vkEnumeratePhysicalDevices(_instance->handle(), &device_count, nullptr));
        LUISA_ASSERT(device_count > 0u, "Failed to find GPUs with Vulkan support.");
        luisa::vector<VkPhysicalDevice> devices(device_count);
        LUISA_CHECK_VULKAN(vkEnumeratePhysicalDevices(_instance->handle(), &device_count, devices.data()));
        luisa::optional<uint32_t> queue_family;
        for (auto device : devices) {
            if (check_properties(device) && check_extensions(device) && check_swapchain_support(device, allow_hdr)) {
                if (auto present_queue_family = find_queue_family(device)) {
                    _physical_device = device;
                    queue_family = present_queue_family;
                    break;
                }
            }
        }
        // maybe the device doesn't support HDR, try again without HDR
        if (_physical_device == nullptr && allow_hdr) {
            for (auto device : devices) {
                if (check_properties(device) && check_extensions(device) && check_swapchain_support(device, false)) {
                    if (auto present_queue_family = find_queue_family(device)) {
                        _physical_device = device;
                        queue_family = present_queue_family;
                        break;
                    }
                }
            }
        }
        LUISA_ASSERT(_physical_device != nullptr, "Failed to find a suitable GPU.");

        // report device properties
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(_physical_device, &properties);
        auto device_type = [&properties] {
            switch (properties.deviceType) {
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    return "Integrated";
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    return "Discrete";
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    return "Virtual";
                case VK_PHYSICAL_DEVICE_TYPE_CPU:
                    return "CPU";
                default:
                    return "Other";
            }
        }();
        LUISA_VERBOSE_WITH_LOCATION(
            "Vulkan Device: {} ({}), Driver: {}.{}.{}, API: {}.{}.{}",
            properties.deviceName, device_type,
            VK_VERSION_MAJOR(properties.driverVersion),
            VK_VERSION_MINOR(properties.driverVersion),
            VK_VERSION_PATCH(properties.driverVersion),
            VK_VERSION_MAJOR(properties.apiVersion),
            VK_VERSION_MINOR(properties.apiVersion),
            VK_VERSION_PATCH(properties.apiVersion));

        // create the logical device
        VkDeviceQueueCreateInfo queue_create_info{};
        auto queue_priority = 1.0f;
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = *queue_family;
        queue_create_info.queueCount = 1;
        queue_create_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_create_info{};
        device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_create_info.queueCreateInfoCount = 1u;
        device_create_info.pQueueCreateInfos = &queue_create_info;
        device_create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        device_create_info.ppEnabledExtensionNames = device_extensions.data();

        // enable validation layers if necessary
        static constexpr std::array validation_layers{"VK_LAYER_KHRONOS_validation"};
        if (_instance->has_debug_layer()) {
            device_create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
            device_create_info.ppEnabledLayerNames = validation_layers.data();
        } else {
            device_create_info.enabledLayerCount = 0u;
        }
        LUISA_CHECK_VULKAN(vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));

        // get the queue
        vkGetDeviceQueue(_device, *queue_family, 0u, &_queue);

        // create the command pool
        VkCommandPoolCreateInfo pool_create_info{};
        pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_create_info.queueFamilyIndex = *queue_family;
        pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        LUISA_CHECK_VULKAN(vkCreateCommandPool(_device, &pool_create_info, nullptr, &_command_pool));
    }

    void _create_swapchain(uint width, uint height, uint back_buffers,
                           bool is_recreation, bool allow_hdr, bool vsync) noexcept {

        auto support = _query_swapchain_support(_physical_device);
        if (support.capabilities.maxImageCount == 0u) { support.capabilities.maxImageCount = back_buffers; }
        if (!is_recreation) {// only allow change back buffer count and swapchain format on first creation
            back_buffers = std::clamp(
                back_buffers,
                support.capabilities.minImageCount,
                support.capabilities.maxImageCount);
            _swapchain_format = [&formats = support.formats, allow_hdr] {
                for (auto f : formats) {
                    LUISA_VERBOSE_WITH_LOCATION(
                        "Supported swapchain format: "
                        "colorspace = {}, format = {}",
                        luisa::to_string(f.colorSpace),
                        luisa::to_string(f.format));
                }
                if (allow_hdr) {
                    for (auto format : formats) {
                        if (format.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) { return format; }
                    }
                }
                for (auto format : formats) {
                    if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
                        (format.format == VK_FORMAT_R8G8B8A8_SRGB ||
                         format.format == VK_FORMAT_B8G8R8A8_SRGB)) { return format; }
                }
                for (auto format : formats) {
                    if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { return format; }
                }
                return formats.front();
            }();
        }

        _swapchain_extent = [&capabilities = support.capabilities, width, height] {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() &&
                capabilities.currentExtent.height != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            }
            VkExtent2D actual_extent{width, height};
            actual_extent.width = std::clamp(actual_extent.width,
                                             capabilities.minImageExtent.width,
                                             capabilities.maxImageExtent.width);
            actual_extent.height = std::clamp(actual_extent.height,
                                              capabilities.minImageExtent.height,
                                              capabilities.maxImageExtent.height);
            return actual_extent;
        }();

        auto present_mode = [&present_modes = support.present_modes, vsync] {
            if (!vsync) {// try to use mailbox mode if vsync is disabled
                for (auto mode : present_modes) {
                    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) { return mode; }
                }
                for (auto mode : present_modes) {
                    if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) { return mode; }
                }
            }
            LUISA_ASSERT(std::find(present_modes.cbegin(),
                                   present_modes.cend(),
                                   VK_PRESENT_MODE_FIFO_KHR) != present_modes.cend(),
                         "FIFO present mode is not supported.");
            return VK_PRESENT_MODE_FIFO_KHR;
        }();

        // create the swapchain
        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = _surface;
        create_info.minImageCount = back_buffers;
        create_info.imageFormat = _swapchain_format.format;
        create_info.imageColorSpace = _swapchain_format.colorSpace;
        create_info.imageExtent = _swapchain_extent;
        create_info.imageArrayLayers = 1u;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info.preTransform = support.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        LUISA_CHECK_VULKAN(vkCreateSwapchainKHR(_device, &create_info, nullptr, &_swapchain));

        // get the swapchain images
        auto image_count = back_buffers;
        _swapchain_images.resize(image_count);
        LUISA_CHECK_VULKAN(vkGetSwapchainImagesKHR(_device, _swapchain, &image_count, _swapchain_images.data()));
        LUISA_ASSERT(image_count == back_buffers, "Swapchain image count mismatch.");

        // create the swapchain image views
        _swapchain_image_views.resize(image_count);
        for (auto i = 0u; i < image_count; i++) {
            VkImageViewCreateInfo image_view_create_info{};
            image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            image_view_create_info.image = _swapchain_images[i];
            image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            image_view_create_info.format = _swapchain_format.format;
            image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_view_create_info.subresourceRange.baseMipLevel = 0u;
            image_view_create_info.subresourceRange.levelCount = 1u;
            image_view_create_info.subresourceRange.baseArrayLayer = 0u;
            image_view_create_info.subresourceRange.layerCount = 1u;
            LUISA_CHECK_VULKAN(vkCreateImageView(_device, &image_view_create_info, nullptr, &_swapchain_image_views[i]));
        }
        LUISA_INFO("Created swapchain: {}x{} with {} back buffer(s) in {} (format = {}, mode = {}).",
                   _swapchain_extent.width, _swapchain_extent.height, back_buffers,
                   _colorspace_name(_swapchain_format.colorSpace),
                   luisa::to_string(_swapchain_format.format),
                   luisa::to_string(present_mode));
    }

    void _create_render_pass() noexcept {
        VkAttachmentDescription color_attachment{};
        color_attachment.format = _swapchain_format.format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref{};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;
        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;
        LUISA_CHECK_VULKAN(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_render_pass));
    }

    void _create_descriptor_set_layout() noexcept {

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.anisotropyEnable = VK_FALSE;
        sampler_info.maxAnisotropy = 0;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable = VK_FALSE;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        LUISA_CHECK_VULKAN(vkCreateSampler(_device, &sampler_info, nullptr, &_texture_sampler));

        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};

        // binding 0: image
        bindings[0].binding = 0;
        bindings[0].descriptorCount = 1;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].pImmutableSamplers = nullptr;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        // binding 1: immutable sampler
        bindings[1].binding = 1;
        bindings[1].descriptorCount = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[1].pImmutableSamplers = &_texture_sampler;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        LUISA_CHECK_VULKAN(vkCreateDescriptorSetLayout(_device, &layout_info, nullptr, &_descriptor_set_layout));
    }

    void _create_pipeline() noexcept {

        auto create_shader_module = [this](auto code) noexcept {
            VkShaderModuleCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            create_info.codeSize = code.size_bytes();
            create_info.pCode = code.data();
            VkShaderModule shader_module;
            LUISA_CHECK_VULKAN(vkCreateShaderModule(_device, &create_info, nullptr, &shader_module));
            return shader_module;
        };
        auto vert_shader = create_shader_module(luisa::span{vulkan_swapchain_screen_shader_vertex_bytecode});
        auto frag_shader = create_shader_module(luisa::span{vulkan_swapchain_screen_shader_fragment_bytecode});

        VkPipelineShaderStageCreateInfo vertex_stage_info{};
        vertex_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertex_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertex_stage_info.module = vert_shader;
        vertex_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo fragment_stage_info{};
        fragment_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragment_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragment_stage_info.module = frag_shader;
        fragment_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo shader_stages[] = {vertex_stage_info, fragment_stage_info};

        VkVertexInputBindingDescription vertex_description{};
        vertex_description.binding = 0;
        vertex_description.stride = sizeof(float) * 2;
        vertex_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attribute_description{};
        attribute_description.binding = 0;
        attribute_description.location = 0;
        attribute_description.format = VK_FORMAT_R32G32_SFLOAT;
        attribute_description.offset = 0;

        VkPipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.vertexAttributeDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &vertex_description;
        vertex_input_info.pVertexAttributeDescriptions = &attribute_description;

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
        color_blending.blendConstants[0] = 0.0f;
        color_blending.blendConstants[1] = 0.0f;
        color_blending.blendConstants[2] = 0.0f;
        color_blending.blendConstants[3] = 0.0f;

        std::array dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamicState.pDynamicStates = dynamic_states.data();

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &_descriptor_set_layout;
        LUISA_CHECK_VULKAN(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_pipeline_layout));

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages;
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamicState;
        pipeline_info.layout = _pipeline_layout;
        pipeline_info.renderPass = _render_pass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        LUISA_CHECK_VULKAN(vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &_graphics_pipeline));

        vkDestroyShaderModule(_device, vert_shader, nullptr);
        vkDestroyShaderModule(_device, frag_shader, nullptr);
    }

    void _create_framebuffers() noexcept {
        _swapchain_framebuffers.resize(_swapchain_image_views.size());
        for (auto i = 0u; i < _swapchain_image_views.size(); i++) {
            VkImageView attachments[] = {_swapchain_image_views[i]};
            VkFramebufferCreateInfo framebuffer_create_info{};
            framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_create_info.renderPass = _render_pass;
            framebuffer_create_info.attachmentCount = 1u;
            framebuffer_create_info.pAttachments = attachments;
            framebuffer_create_info.width = std::max(1u, _swapchain_extent.width);
            framebuffer_create_info.height = std::max(1u, _swapchain_extent.height);
            framebuffer_create_info.layers = 1u;
            LUISA_CHECK_VULKAN(vkCreateFramebuffer(_device, &framebuffer_create_info, nullptr, &_swapchain_framebuffers[i]));
        }
    }

    void _create_vertex_buffer() noexcept {
        const std::array vertices = {1.f, 0.f,
                                     0.f, 0.f,
                                     0.f, 1.f,
                                     1.f, 0.f,
                                     0.f, 1.f,
                                     1.f, 1.f};
        auto buffer_size = sizeof(vertices);

        auto find_memory_type = [this](uint type_filter, VkMemoryPropertyFlags properties) noexcept {
            VkPhysicalDeviceMemoryProperties memory_properties;
            vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);
            for (auto i = 0u; i < memory_properties.memoryTypeCount; i++) {
                if ((type_filter & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }
            LUISA_ERROR_WITH_LOCATION("Failed to find suitable memory type.");
        };

        auto create_buffer = [this, &find_memory_type](VkDeviceSize size,
                                                       VkBufferUsageFlags usage,
                                                       VkMemoryPropertyFlags properties) noexcept {
            VkBuffer buffer;
            VkDeviceMemory buffer_memory;
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            LUISA_CHECK_VULKAN(vkCreateBuffer(_device, &buffer_info, nullptr, &buffer));
            VkMemoryRequirements memory_requirements;
            vkGetBufferMemoryRequirements(_device, buffer, &memory_requirements);
            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = memory_requirements.size;
            alloc_info.memoryTypeIndex = find_memory_type(memory_requirements.memoryTypeBits, properties);
            LUISA_CHECK_VULKAN(vkAllocateMemory(_device, &alloc_info, nullptr, &buffer_memory));
            LUISA_CHECK_VULKAN(vkBindBufferMemory(_device, buffer, buffer_memory, 0));
            return std::make_pair(buffer, buffer_memory);
        };

        auto [staging_buffer, staging_buffer_memory] = create_buffer(
            buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void *data = nullptr;
        LUISA_CHECK_VULKAN(vkMapMemory(_device, staging_buffer_memory, 0, buffer_size, 0, &data));
        std::memcpy(data, vertices.data(), buffer_size);
        vkUnmapMemory(_device, staging_buffer_memory);

        std::tie(_vertex_buffer, _vertex_buffer_memory) = create_buffer(
            buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        // copy buffer
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = _command_pool;
        alloc_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer;
        LUISA_CHECK_VULKAN(vkAllocateCommandBuffers(_device, &alloc_info, &command_buffer));
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        LUISA_CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));
        VkBufferCopy copy_region{};
        copy_region.size = buffer_size;
        vkCmdCopyBuffer(command_buffer, staging_buffer, _vertex_buffer, 1, &copy_region);
        LUISA_CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        LUISA_CHECK_VULKAN(vkQueueSubmit(_queue, 1, &submit_info, nullptr));
        LUISA_CHECK_VULKAN(vkQueueWaitIdle(_queue));
        vkFreeCommandBuffers(_device, _command_pool, 1, &command_buffer);

        vkDestroyBuffer(_device, staging_buffer, nullptr);
        vkFreeMemory(_device, staging_buffer_memory, nullptr);
    }

    void _create_descriptor_pool() noexcept {
        std::array<VkDescriptorPoolSize, 2> pool_sizes{};
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        pool_sizes[0].descriptorCount = static_cast<uint32_t>(_swapchain_images.size());
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        pool_sizes[1].descriptorCount = static_cast<uint32_t>(_swapchain_images.size());
        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = static_cast<uint32_t>(_swapchain_images.size());
        LUISA_CHECK_VULKAN(vkCreateDescriptorPool(_device, &pool_info, nullptr, &_descriptor_pool));
    }

    void _create_descriptor_sets() noexcept {
        luisa::vector<VkDescriptorSetLayout> layouts(_swapchain_images.size(), _descriptor_set_layout);
        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = _descriptor_pool;
        alloc_info.descriptorSetCount = static_cast<uint32_t>(_swapchain_images.size());
        alloc_info.pSetLayouts = layouts.data();
        _descriptor_sets.resize(_swapchain_images.size());
        LUISA_CHECK_VULKAN(vkAllocateDescriptorSets(_device, &alloc_info, _descriptor_sets.data()));
        _cached_image_infos.resize(_swapchain_images.size());
    }

    void _create_command_buffers() noexcept {
        _command_buffers.resize(_swapchain_images.size());
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = _command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<uint32_t>(_command_buffers.size());
        LUISA_CHECK_VULKAN(vkAllocateCommandBuffers(_device, &alloc_info, _command_buffers.data()));
    }

    void _create_sync_objects() noexcept {
        _in_flight_fences.resize(_swapchain_images.size());
        _image_available_semaphores.resize(_swapchain_images.size());
        _render_finished_semaphores.resize(_swapchain_images.size());
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (auto i = 0u; i < _swapchain_images.size(); i++) {
            LUISA_CHECK_VULKAN(vkCreateFence(_device, &fence_info, nullptr, &_in_flight_fences[i]));
            LUISA_CHECK_VULKAN(vkCreateSemaphore(_device, &semaphore_info, nullptr, &_image_available_semaphores[i]));
            LUISA_CHECK_VULKAN(vkCreateSemaphore(_device, &semaphore_info, nullptr, &_render_finished_semaphores[i]));
        }
    }

    void _record_command_buffer(VkCommandBuffer command_buffer, uint image_index) const noexcept {

        LUISA_CHECK_VULKAN(vkResetCommandBuffer(command_buffer, 0));

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        LUISA_CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = _render_pass;
        render_pass_info.framebuffer = _swapchain_framebuffers[image_index];
        render_pass_info.renderArea.offset = {0, 0};
        render_pass_info.renderArea.extent = _swapchain_extent;
        VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        render_pass_info.clearValueCount = 1;
        render_pass_info.pClearValues = &clear_color;
        vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _graphics_pipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = std::max(1.f, static_cast<float>(_swapchain_extent.width));
        viewport.height = std::max(1.f, static_cast<float>(_swapchain_extent.height));
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = _swapchain_extent;
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        auto vertex_buffer = _vertex_buffer;
        auto vertex_buffer_offset = static_cast<VkDeviceSize>(0u);
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &vertex_buffer_offset);
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline_layout, 0, 1, &_descriptor_sets[_current_frame], 0, nullptr);

        vkCmdDraw(command_buffer, 6u, 1, 0, 0);
        vkCmdEndRenderPass(command_buffer);

        LUISA_CHECK_VULKAN(vkEndCommandBuffer(command_buffer));
    }

private:
    uint2 _requested_size{};
    bool _requested_hdr{false};
    bool _requested_vsync{false};

private:
    void _destroy_swapchain() noexcept {
        for (auto i = 0u; i < _swapchain_images.size(); i++) {
            vkDestroyFramebuffer(_device, _swapchain_framebuffers[i], nullptr);
            vkDestroyImageView(_device, _swapchain_image_views[i], nullptr);
        }
        _swapchain_images.clear();
        vkDestroySwapchainKHR(_device, _swapchain, nullptr);
    }

    void _recreate_swapchain() noexcept {
        auto back_buffers = _swapchain_framebuffers.size();
        vkDeviceWaitIdle(_device);
        _destroy_swapchain();
        _create_swapchain(_requested_size.x, _requested_size.y,
                          back_buffers, true,
                          _requested_hdr, _requested_vsync);
        _create_framebuffers();
    }

public:
    Impl(VulkanDeviceUUID device_uuid,
         uint64_t display_handle,
         uint64_t window_handle,
         uint width, uint height,
         bool allow_hdr, bool vsync,
         uint back_buffer_count,
         luisa::span<const char *const> required_device_extensions) noexcept
        : _instance{VulkanInstance::retain()},
          _requested_size{width, height},
          _requested_hdr{allow_hdr},
          _requested_vsync{vsync} {
        _create_surface(display_handle, window_handle);
        _create_device(device_uuid, required_device_extensions, allow_hdr);
        _create_swapchain(width, height, back_buffer_count, false, allow_hdr, vsync);
        _create_render_pass();
        _create_descriptor_set_layout();
        _create_pipeline();
        _create_framebuffers();
        _create_vertex_buffer();
        _create_descriptor_pool();
        _create_descriptor_sets();
        _create_command_buffers();
        _create_sync_objects();
    }

    ~Impl() noexcept {
        vkDeviceWaitIdle(_device);
        for (auto i = 0u; i < _swapchain_images.size(); i++) {
            vkDestroyFence(_device, _in_flight_fences[i], nullptr);
            vkDestroySemaphore(_device, _image_available_semaphores[i], nullptr);
            vkDestroySemaphore(_device, _render_finished_semaphores[i], nullptr);
        }
        _destroy_swapchain();
        vkDestroyPipeline(_device, _graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
        vkDestroyRenderPass(_device, _render_pass, nullptr);
        vkDestroyDescriptorPool(_device, _descriptor_pool, nullptr);
        vkDestroySampler(_device, _texture_sampler, nullptr);
        vkDestroyDescriptorSetLayout(_device, _descriptor_set_layout, nullptr);
        vkDestroyBuffer(_device, _vertex_buffer, nullptr);
        vkFreeMemory(_device, _vertex_buffer_memory, nullptr);
        vkDestroyCommandPool(_device, _command_pool, nullptr);
        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance->handle(), _surface, nullptr);
    }

    void wait_for_fence() noexcept {
        // wait for command buffer to finish
        LUISA_CHECK_VULKAN(vkWaitForFences(
            _device, 1, &_in_flight_fences[_current_frame],
            VK_TRUE, UINT64_MAX));
    }

    void present(VkSemaphore wait, VkSemaphore signal,
                 VkImageView image, VkImageLayout image_layout) noexcept {

        wait_for_fence();

        // acquire next image
        auto image_index = 0u;
        if (auto ret = vkAcquireNextImageKHR(
                _device, _swapchain, UINT64_MAX,
                _image_available_semaphores[_current_frame],
                VK_NULL_HANDLE, &image_index);
            ret == VK_ERROR_OUT_OF_DATE_KHR) {
            _recreate_swapchain();
            return;
        } else if (ret != VK_SUCCESS && ret != VK_SUBOPTIMAL_KHR) {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to acquire swapchain image: {}.",
                luisa::to_string(ret));
        }
        LUISA_CHECK_VULKAN(vkResetFences(_device, 1, &_in_flight_fences[_current_frame]));

        // update descriptor set if necessary
        if (image != _cached_image_infos[_current_frame].imageView ||
            image_layout != _cached_image_infos[_current_frame].imageLayout) {
            _cached_image_infos[_current_frame].imageView = image;
            _cached_image_infos[_current_frame].imageLayout = image_layout;
            VkWriteDescriptorSet descriptor_write{};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstBinding = 0u;
            descriptor_write.dstArrayElement = 0u;
            descriptor_write.dstSet = _descriptor_sets[_current_frame];
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            descriptor_write.descriptorCount = 1u;
            descriptor_write.pImageInfo = &_cached_image_infos[_current_frame];
            vkUpdateDescriptorSets(_device, 1u, &descriptor_write, 0u, nullptr);
        }

        // record command buffer
        _record_command_buffer(_command_buffers[_current_frame], image_index);

        // submit command buffer
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        std::array wait_semaphores = {_image_available_semaphores[_current_frame], wait};
        std::array wait_stages = {static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
                                  static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)};
        submit_info.waitSemaphoreCount = wait == nullptr ? 1u : 2u;
        submit_info.pWaitSemaphores = wait_semaphores.data();
        submit_info.pWaitDstStageMask = wait_stages.data();
        std::array signal_semaphores = {_render_finished_semaphores[_current_frame], signal};
        submit_info.signalSemaphoreCount = signal == nullptr ? 1u : 2u;
        submit_info.pSignalSemaphores = signal_semaphores.data();
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &_command_buffers[_current_frame];
        LUISA_CHECK_VULKAN(vkQueueSubmit(_queue, 1u, &submit_info, _in_flight_fences[_current_frame]));

        // present
        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.pSwapchains = &_swapchain;
        present_info.swapchainCount = 1u;
        present_info.waitSemaphoreCount = 1u;
        present_info.pWaitSemaphores = &_render_finished_semaphores[_current_frame];
        present_info.pImageIndices = &image_index;
        LUISA_CHECK_VULKAN(vkQueuePresentKHR(_queue, &present_info));

        // update current frame index
        _current_frame = (_current_frame + 1u) % _swapchain_images.size();
    }
    [[nodiscard]] auto instance() const noexcept { return _instance->handle(); }
    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto swapchain_extent() const noexcept { return _swapchain_extent; }
    [[nodiscard]] auto swapchain_format() const noexcept { return _swapchain_format; }
    [[nodiscard]] auto back_buffer_count() const noexcept { return _swapchain_images.size(); }
    [[nodiscard]] auto command_pool() const noexcept { return _command_pool; }
    [[nodiscard]] auto physical_device() const noexcept { return _physical_device; }
    [[nodiscard]] auto queue() const noexcept { return _queue; }
    [[nodiscard]] auto is_hdr() const noexcept { return _is_hdr_colorspace(_swapchain_format.colorSpace); }
};

VulkanSwapchain::VulkanSwapchain(const VulkanDeviceUUID &device_uuid,
                                 uint64_t display_handle,
                                 uint64_t window_handle,
                                 uint width, uint height,
                                 bool allow_hdr, bool vsync,
                                 uint back_buffer_count,
                                 luisa::span<const char *const> required_device_extensions) noexcept
    : _impl{luisa::make_unique<Impl>(device_uuid, display_handle, window_handle,
                                     width, height, allow_hdr,
                                     vsync, back_buffer_count,
                                     required_device_extensions)} {}

VulkanSwapchain::~VulkanSwapchain() noexcept = default;
VkInstance VulkanSwapchain::instance() const noexcept { return _impl->instance(); }
VkDevice VulkanSwapchain::device() const noexcept { return _impl->device(); }
VkPhysicalDevice VulkanSwapchain::physical_device() const noexcept { return _impl->physical_device(); }
VkExtent2D VulkanSwapchain::extent() const noexcept { return _impl->swapchain_extent(); }
VkSurfaceFormatKHR VulkanSwapchain::format() const noexcept { return _impl->swapchain_format(); }
size_t VulkanSwapchain::back_buffer_count() const noexcept { return _impl->back_buffer_count(); }
VkCommandPool VulkanSwapchain::command_pool() const noexcept { return _impl->command_pool(); }
VkQueue VulkanSwapchain::queue() const noexcept { return _impl->queue(); }
bool VulkanSwapchain::is_hdr() const noexcept { return _impl->is_hdr(); }
void VulkanSwapchain::wait_for_fence() noexcept { _impl->wait_for_fence(); }
void VulkanSwapchain::present(VkSemaphore wait, VkSemaphore signal,
                              VkImageView image, VkImageLayout image_layout) noexcept {
    _impl->present(wait, signal, image, image_layout);
}

class VulkanSwapchainForCPU {

private:
    VulkanSwapchain _base;
    size_t _stage_buffer_size{};
    luisa::vector<VkBuffer> _stage_buffers;
    luisa::vector<VkDeviceMemory> _stage_buffer_memories;

    VkFormat _image_format{};
    VkImage _image{nullptr};
    VkDeviceMemory _image_memory{nullptr};
    VkImageView _image_view{nullptr};
    luisa::vector<VkCommandBuffer> _command_buffers;
    uint _current_frame{0u};
    VkExtent2D _image_extent;

private:
    [[nodiscard]] auto _find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) noexcept {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(_base.physical_device(), &memory_properties);
        for (auto i = 0u; i < memory_properties.memoryTypeCount; i++) {
            if ((type_filter & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        LUISA_ERROR_WITH_LOCATION("Failed to find suitable memory type.");
    }

    void _create_image() noexcept {

        // choose format
        _image_format = _base.is_hdr() ?
                            VK_FORMAT_R16G16B16A16_SFLOAT :
                            VK_FORMAT_R8G8B8A8_SRGB;

        // create image
        VkImageCreateInfo image_info{};
        image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.extent.width = _image_extent.width;
        image_info.extent.height = _image_extent.height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.format = _image_format;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        LUISA_CHECK_VULKAN(vkCreateImage(_base.device(), &image_info, nullptr, &_image));

        // allocate memory
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(_base.device(), _image, &mem_requirements);
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = _find_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        LUISA_CHECK_VULKAN(vkAllocateMemory(_base.device(), &alloc_info, nullptr, &_image_memory));
        LUISA_CHECK_VULKAN(vkBindImageMemory(_base.device(), _image, _image_memory, 0));
    }

    void _transition_image_layout() noexcept {

        // create a single-use command buffer
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandPool = _base.command_pool();
        alloc_info.commandBufferCount = 1;
        VkCommandBuffer command_buffer;
        LUISA_CHECK_VULKAN(vkAllocateCommandBuffers(_base.device(), &alloc_info, &command_buffer));

        // begin recording
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        LUISA_CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        // transition image layout
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = _image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);

        // end recording
        LUISA_CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        // submit command buffer
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        LUISA_CHECK_VULKAN(vkQueueSubmit(_base.queue(), 1, &submit_info, VK_NULL_HANDLE));
        LUISA_CHECK_VULKAN(vkQueueWaitIdle(_base.queue()));

        // free command buffer
        vkFreeCommandBuffers(_base.device(), _base.command_pool(), 1, &command_buffer);
    }

    void _create_image_view() noexcept {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = _image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = _image_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        LUISA_CHECK_VULKAN(vkCreateImageView(_base.device(), &view_info, nullptr, &_image_view));
    }

    void _create_buffers() noexcept {

        // compute stage buffer size
        auto pixel_size = [this] {
            switch (_image_format) {
                case VK_FORMAT_R8G8B8A8_SRGB:
                    return 4u;
                case VK_FORMAT_R16G16B16A16_SFLOAT:
                    return 8u;
                default:
                    break;
            }
            LUISA_ERROR_WITH_LOCATION("Unsupported image format.");
        }();
        _stage_buffer_size = _image_extent.width * _image_extent.height * pixel_size;

        // create stage buffers
        _stage_buffers.resize(_base.back_buffer_count());
        _stage_buffer_memories.resize(_base.back_buffer_count());
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = _stage_buffer_size;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        for (auto i = 0u; i < _base.back_buffer_count(); i++) {
            LUISA_CHECK_VULKAN(vkCreateBuffer(_base.device(), &buffer_info, nullptr, &_stage_buffers[i]));
            VkMemoryRequirements mem_requirements;
            vkGetBufferMemoryRequirements(_base.device(), _stage_buffers[i], &mem_requirements);
            VkMemoryAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = mem_requirements.size;
            alloc_info.memoryTypeIndex = _find_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            LUISA_CHECK_VULKAN(vkAllocateMemory(_base.device(), &alloc_info, nullptr, &_stage_buffer_memories[i]));
            LUISA_CHECK_VULKAN(vkBindBufferMemory(_base.device(), _stage_buffers[i], _stage_buffer_memories[i], 0));
        }
    }

    void _create_command_buffers() noexcept {
        _command_buffers.resize(_base.back_buffer_count());
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = _base.command_pool();
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<uint>(_command_buffers.size());
        LUISA_CHECK_VULKAN(vkAllocateCommandBuffers(_base.device(), &alloc_info, _command_buffers.data()));
    }

    void _initialize() noexcept {
        _create_image();
        _transition_image_layout();
        _create_image_view();
        _create_buffers();
        _create_command_buffers();
    }

    void _clean_up() noexcept {
        vkDeviceWaitIdle(_base.device());
        auto device = _base.device();
        for (auto i = 0u; i < _base.back_buffer_count(); i++) {
            vkDestroyBuffer(device, _stage_buffers[i], nullptr);
            vkFreeMemory(device, _stage_buffer_memories[i], nullptr);
        }
        vkDestroyImageView(device, _image_view, nullptr);
        vkDestroyImage(device, _image, nullptr);
        vkFreeMemory(device, _image_memory, nullptr);
    }

public:
    VulkanSwapchainForCPU(uint64_t display_handle, uint64_t window_handle, uint width, uint height,
                          bool allow_hdr, bool vsync, uint back_buffer_count) noexcept
        : _base{VulkanDeviceUUID{/* any */},
                display_handle,
                window_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_count,
                {/* not required */}},
          _image_extent{width, height} {
        _initialize();
    }

    ~VulkanSwapchainForCPU() noexcept { _clean_up(); }

    [[nodiscard]] auto format() const noexcept { return _image_format; }

    [[nodiscard]] auto pixel_storage() const noexcept {
        LUISA_ASSERT(_image_format == VK_FORMAT_R8G8B8A8_SRGB ||
                         _image_format == VK_FORMAT_R16G16B16A16_SFLOAT,
                     "Unsupported image format.");
        return _image_format == VK_FORMAT_R8G8B8A8_SRGB ?
                   PixelStorage::BYTE4 :
                   PixelStorage::HALF4;
    }

    using BlitCallback = void (*)(void *ctx, void *mapped_pixels);

    void present(void *ctx, BlitCallback blit) noexcept {
        _base.wait_for_fence();

        // update stage buffer
        void *mapped = nullptr;
        LUISA_CHECK_VULKAN(vkMapMemory(_base.device(), _stage_buffer_memories[_current_frame], 0u, _stage_buffer_size, 0u, &mapped));
        blit(ctx, mapped);
        vkUnmapMemory(_base.device(), _stage_buffer_memories[_current_frame]);

        // copy buffer to image
        auto command_buffer = _command_buffers[_current_frame];
        LUISA_CHECK_VULKAN(vkResetCommandBuffer(command_buffer, 0u));

        // encode command buffer
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        LUISA_CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));
        VkBufferImageCopy region{};
        region.bufferOffset = 0u;
        region.bufferRowLength = 0u;
        region.bufferImageHeight = 0u;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0u;
        region.imageSubresource.baseArrayLayer = 0u;
        region.imageSubresource.layerCount = 1u;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {_image_extent.width, _image_extent.height, 1u};
        vkCmdCopyBufferToImage(command_buffer, _stage_buffers[_current_frame], _image, VK_IMAGE_LAYOUT_GENERAL, 1u, &region);
        vkEndCommandBuffer(command_buffer);

        // submit command buffer
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1u;
        submit_info.pCommandBuffers = &command_buffer;
        LUISA_CHECK_VULKAN(vkQueueSubmit(_base.queue(), 1u, &submit_info, nullptr));

        // present
        _base.present(nullptr, nullptr, _image_view, VK_IMAGE_LAYOUT_GENERAL);

        // update frame index
        _current_frame = (_current_frame + 1u) % _base.back_buffer_count();
    }

    void present(luisa::span<const std::byte> pixels) noexcept {
        LUISA_ASSERT(pixels.size_bytes() >= _stage_buffer_size,
                     "Pixel buffer is too small.");
        present(&pixels, [](void *ctx, void *mapped) noexcept {
            auto pixels = static_cast<luisa::span<const std::byte> *>(ctx);
            std::memcpy(mapped, pixels->data(), pixels->size_bytes());
        });
    }

    [[nodiscard]] VulkanSwapchain *native_handle() noexcept { return &_base; }
    [[nodiscard]] const VulkanSwapchain *native_handle() const noexcept { return &_base; }
};

LUISA_EXPORT_API void *luisa_compute_create_cpu_swapchain(uint64_t display_handle, uint64_t window_handle,
                                                          uint width, uint height, bool allow_hdr, bool vsync,
                                                          uint back_buffer_count) noexcept {
    return new VulkanSwapchainForCPU{display_handle, window_handle, width, height, allow_hdr, vsync, back_buffer_count};
}

LUISA_EXPORT_API uint8_t luisa_compute_cpu_swapchain_storage(void *swapchain) noexcept {
    return static_cast<uint8_t>(static_cast<VulkanSwapchainForCPU *>(swapchain)->pixel_storage());
}

LUISA_EXPORT_API void *luisa_compute_cpu_swapchain_native_handle(void *swapchain) noexcept {
    return static_cast<VulkanSwapchainForCPU *>(swapchain)->native_handle();
}

LUISA_EXPORT_API void luisa_compute_destroy_cpu_swapchain(void *swapchain) noexcept {
    delete static_cast<VulkanSwapchainForCPU *>(swapchain);
}

LUISA_EXPORT_API void luisa_compute_cpu_swapchain_present(void *swapchain, const void *pixels, uint64_t size) noexcept {
    static_cast<VulkanSwapchainForCPU *>(swapchain)->present(
        luisa::span{static_cast<const std::byte *>(pixels),
                    static_cast<unsigned long>(size)});
}

LUISA_EXPORT_API void luisa_compute_cpu_swapchain_present_with_callback(void *swapchain, void *ctx, void (*blit)(void *ctx, void *mapped_pixels)) noexcept {
    static_cast<VulkanSwapchainForCPU *>(swapchain)->present(ctx, blit);
}

}// namespace luisa::compute
