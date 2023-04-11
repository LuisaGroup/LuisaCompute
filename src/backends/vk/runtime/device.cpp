#include "device.h"
#include <core/logging.h>
#include "../log.h"
namespace lc::vk {
namespace detail {
static VkInstance vk_instance{nullptr};
static std::mutex instance_mtx;
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;
static VkDebugUtilsMessengerEXT debugUtilsMessenger;

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData) {
    // Select prefix depending on flags passed to the callback
    vstd::string prefix;

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        prefix = "VERBOSE: ";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        prefix = "INFO: ";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        prefix = "WARNING: ";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        prefix = "ERROR: ";
    }

    // Display message to default output (console/logcat)
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        vstd::string debugMessage;
        debugMessage << prefix << "[" << vstd::to_string(pCallbackData->messageIdNumber) << "][" << pCallbackData->pMessageIdName << "] : " << pCallbackData->pMessage;
        LUISA_ERROR("{}", debugMessage);
    }
    // The return value of this callback controls whether the Vulkan call that caused the validation message will be aborted or not
    // We return VK_FALSE as we DON'T want Vulkan calls that cause a validation message to abort
    // If you instead want to have calls abort, pass in VK_TRUE and the function will return VK_ERROR_VALIDATION_FAILED_EXT
    return VK_FALSE;
}

void setupDebugging(VkInstance instance) {

    vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCI{};
    debugUtilsMessengerCI.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugUtilsMessengerCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugUtilsMessengerCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debugUtilsMessengerCI.pfnUserCallback = debugUtilsMessengerCallback;
    VkResult result = vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsMessengerCI, nullptr, &debugUtilsMessenger);
    assert(result == VK_SUCCESS);
}
}// namespace detail
//////////////// Not implemented area
ResourceCreationInfo Device::create_mesh(
    const AccelOption &option) noexcept {
    LUISA_ERROR("mesh not implemented.");
    return ResourceCreationInfo::make_invalid();
}
void Device::destroy_mesh(uint64_t handle) noexcept {
    LUISA_ERROR("mesh not implemented.");
}

ResourceCreationInfo Device::create_procedural_primitive(
    const AccelOption &option) noexcept {
    LUISA_ERROR("procedural primitive not implemented.");
    return ResourceCreationInfo::make_invalid();
}
void Device::destroy_procedural_primitive(uint64_t handle) noexcept {
    LUISA_ERROR("procedural primitive not implemented.");
}

ResourceCreationInfo Device::create_accel(const AccelOption &option) noexcept {
    LUISA_ERROR("accel not implemented.");
    return ResourceCreationInfo::make_invalid();
}
void Device::destroy_accel(uint64_t handle) noexcept {
    LUISA_ERROR("accel not implemented.");
}
//////////////// Not implemented area

VkInstance Device::create_instance(bool enableValidation, Settings &settings) {
    // Validation can also be forced via a define
    settings.validation = enableValidation;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "luisa_compute";
    appInfo.pEngineName = appInfo.pApplicationName;
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // Enable surface extensions depending on os
#if defined(_WIN32)
    instance_exts.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
    instance_exts.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
    instance_exts.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
    instance_exts.push_back(VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
    instance_exts.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
    instance_exts.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
    instance_exts.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
    instance_exts.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_HEADLESS_EXT)
    instance_exts.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
#endif

    // Get extensions supported by the instance and store for later use
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    if (extCount > 0) {
        vstd::vector<VkExtensionProperties> extensions(extCount);
        if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS) {
            for (VkExtensionProperties extension : extensions) {
                supported_instance_exts.push_back(extension.extensionName);
            }
        }
    }

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
    // SRS - When running on iOS/macOS with MoltenVK, enable VK_KHR_get_physical_device_properties2 if not already enabled by the example (required by VK_KHR_portability_subset)
    if (std::find(enable_inst_ext.begin(), enable_inst_ext.end(), VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == enable_inst_ext.end()) {
        enable_inst_ext.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#endif

    // Enabled requested instance extensions
    if (enable_inst_ext.size() > 0) {
        for (const char *enabledExtension : enable_inst_ext) {
            // Output message if requested extension is not available
            if (std::find(supported_instance_exts.begin(), supported_instance_exts.end(), enabledExtension) == supported_instance_exts.end()) {
                LUISA_ERROR("Enabled instance extension \"{}\"  is not present at instance level", enabledExtension);
            }
            instance_exts.push_back(enabledExtension);
        }
    }

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = NULL;
    instanceCreateInfo.pApplicationInfo = &appInfo;

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK)) && defined(VK_KHR_portability_enumeration)
    // SRS - When running on iOS/macOS with MoltenVK and VK_KHR_portability_enumeration is defined and supported by the instance, enable the extension and the flag
    if (std::find(supported_instance_exts.begin(), supported_instance_exts.end(), VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != supported_instance_exts.end()) {
        instance_exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instanceCreateInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif

    if (instance_exts.size() > 0) {
        if (settings.validation) {
            instance_exts.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);// SRS - Dependency when VK_EXT_DEBUG_MARKER is enabled
            instance_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        instanceCreateInfo.enabledExtensionCount = (uint32_t)instance_exts.size();
        instanceCreateInfo.ppEnabledExtensionNames = instance_exts.data();
    }

    // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
    // Note that on Android this layer requires at least NDK r20
    const char *validationLayerName = "VK_LAYER_KHRONOS_validation";
    if (settings.validation) {
        // Check if this layer is available at instance level
        uint32_t instanceLayerCount;
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
        vstd::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
        vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());
        bool validationLayerPresent = false;
        for (VkLayerProperties layer : instanceLayerProperties) {
            if (strcmp(layer.layerName, validationLayerName) == 0) {
                validationLayerPresent = true;
                break;
            }
        }
        if (validationLayerPresent) {
            instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
            instanceCreateInfo.enabledLayerCount = 1;
        } else {
            LUISA_ERROR("Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled");
        }
    }
    VkInstance instance;
    VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));
    return instance;
}
Device::Device(Context &&ctx) : DeviceInterface{std::move(ctx)} {
    // init instance
    Settings settings{};
    {
        std::lock_guard lck{detail::instance_mtx};
        if (!detail::vk_instance) {
#ifdef NDEBUG
            constexpr bool enableValidation = false;
#else
            constexpr bool enableValidation = true;
#endif
            detail::vk_instance = create_instance(enableValidation, settings);
        }
    }
}
bool Device::init_device(Settings &settings, uint32_t selectedDevice) {
    VkResult err;

    // If requested, we enable the default validation layers for debugging
    if (settings.validation) {
        detail::setupDebugging(detail::vk_instance);
    }

    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, nullptr));
    if (gpuCount == 0) {
        LUISA_ERROR("No device with Vulkan support found");
        return false;
    }
    // Enumerate devices
    physical_devices.push_back_uninitialized(gpuCount);
    err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, physical_devices.data());
    if (err) {
        LUISA_ERROR("Could not enumerate physical devices : {}", err);
        return false;
    }

    // GPU selection

    // Select physical device to be used for the Vulkan example
    // Defaults to the first device unless specified by command line

    auto physicalDevice = physical_devices[selectedDevice];

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    vkGetPhysicalDeviceProperties(physicalDevice, &device_properties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &device_features);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &device_memory_properties);

    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation

    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    vk_device = vstd::make_unique<vks::VulkanDevice>(physicalDevice);

    VkResult res = vk_device->createLogicalDevice(device_features, enable_device_exts, nullptr);
    if (res != VK_SUCCESS) {
        LUISA_ERROR("Could not create Vulkan device: {}, {}", vks::tools::errorString(res), res);
        return false;
    }
    auto device = vk_device->logicalDevice;

    // Get a graphics queue from the device
    vkGetDeviceQueue(device, vk_device->queueFamilyIndices.graphics, 0, &graphics_queue);
    vkGetDeviceQueue(device, vk_device->queueFamilyIndices.compute, 0, &compute_queue);
    vkGetDeviceQueue(device, vk_device->queueFamilyIndices.transfer, 0, &copy_queue);

    return true;
}
Device::~Device() {
}
void *Device::native_handle() const noexcept { return vk_device->logicalDevice; }
BufferCreationInfo Device::create_buffer(const Type *element, size_t elem_count) noexcept { return BufferCreationInfo::make_invalid(); }
BufferCreationInfo Device::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept { return BufferCreationInfo::make_invalid(); }
void Device::destroy_buffer(uint64_t handle) noexcept {}

// texture
ResourceCreationInfo Device::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels) noexcept { return ResourceCreationInfo::make_invalid(); }
void Device::destroy_texture(uint64_t handle) noexcept {}

// bindless array
ResourceCreationInfo Device::create_bindless_array(size_t size) noexcept { return ResourceCreationInfo::make_invalid(); }
void Device::destroy_bindless_array(uint64_t handle) noexcept {}

// stream
ResourceCreationInfo Device::create_stream(StreamTag stream_tag) noexcept { return ResourceCreationInfo::make_invalid(); }
void Device::destroy_stream(uint64_t handle) noexcept {}
void Device::synchronize_stream(uint64_t stream_handle) noexcept {}
void Device::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {}

// swap chain
SwapChainCreationInfo Device::create_swap_chain(
    uint64_t window_handle, uint64_t stream_handle,
    uint width, uint height, bool allow_hdr,
    bool vsync, uint back_buffer_size) noexcept { return SwapChainCreationInfo{ResourceCreationInfo::make_invalid()}; }
void Device::destroy_swap_chain(uint64_t handle) noexcept {}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {}

// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept { return ShaderCreationInfo::make_invalid(); }
ShaderCreationInfo Device::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept { return ShaderCreationInfo::make_invalid(); }
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept { return ShaderCreationInfo::make_invalid(); }
Usage Device::shader_argument_usage(uint64_t handle, size_t index) noexcept { return Usage::NONE; }
void Device::destroy_shader(uint64_t handle) noexcept {}

// event
ResourceCreationInfo Device::create_event() noexcept { return ResourceCreationInfo::make_invalid(); }
void Device::destroy_event(uint64_t handle) noexcept {}
void Device::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {}
void Device::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {}
void Device::synchronize_event(uint64_t handle) noexcept {}
void Device::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {}
}// namespace lc::vk