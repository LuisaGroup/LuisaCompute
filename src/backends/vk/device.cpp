#include "device.h"
#include <luisa/core/logging.h>
#include "log.h"
#include <luisa/vstl/config.h>
#include <luisa/core/binary_file_stream.h>
#include "compute_shader.h"
#include "../common/hlsl/hlsl_codegen.h"
#include "serde_type.h"
#include "../common/hlsl/binding_to_arg.h"
#include <luisa/runtime/context.h>
#include "../common/hlsl/shader_compiler.h"
#include "shader_serializer.h"

namespace lc::vk {
static std::mutex gDxcMutex;
static vstd::optional<hlsl::ShaderCompiler> gDxcCompiler;
static constexpr uint k_shader_model = 65u;
static int32 gDxcRefCount = 0;

using namespace std::string_literals;
namespace detail {
struct Settings {
    bool validation;
    bool fullscreen{false};
    bool vsync{false};
    bool overlay{true};
};
static VkInstance vk_instance{nullptr};
static std::mutex instance_mtx;
static Settings settings{};
static PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
static PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;
static VkDebugUtilsMessengerEXT debugUtilsMessenger;

struct InstanceDestructor {
    ~InstanceDestructor() {
        if (vk_instance) {
            vkDestroyInstance(vk_instance, nullptr);
        }
    }
};
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
vstd::vector<VkExtensionProperties> supported_exts(VkPhysicalDevice physical_device) {
    uint extensions_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, nullptr);
    vstd::vector<VkExtensionProperties> props;
    props.push_back_uninitialized(extensions_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, props.data());
    return props;
}
VkInstance create_instance(bool enableValidation) {
    vstd::vector<const char *> instance_exts = {VK_KHR_SURFACE_EXTENSION_NAME};
    vstd::vector<const char *> enable_inst_ext;
    vstd::unordered_set<vstd::string> supported_instance_exts;

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
                supported_instance_exts.emplace(extension.extensionName);
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
            if (supported_instance_exts.find(enabledExtension) == supported_instance_exts.end()) {
                LUISA_ERROR("Enabled instance extension \"{}\"  is not present at instance level", enabledExtension);
            }
            instance_exts.push_back(enabledExtension);
        }
    }

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = NULL;
    instanceCreateInfo.pApplicationInfo = &appInfo;

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
            LUISA_WARNING("Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled");
            settings.validation = false;
        }
    }

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK)) && defined(VK_KHR_portability_enumeration)
    // SRS - When running on iOS/macOS with MoltenVK and VK_KHR_portability_enumeration is defined and supported by the instance, enable the extension and the flag
    if (supported_instance_exts.find(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == supported_instance_exts.end()) {
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

    VkInstance instance;
    VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));
    return instance;
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
Device::Device(Context &&ctx, DeviceConfig const *configs)
    : DeviceInterface{std::move(ctx)} {
    bool headless = false;
    uint device_idx = 0;
    if (configs) {
        headless = configs->headless;
        device_idx = configs->device_index;
        _binary_io = configs->binary_io;
    }
    if (!headless) {
        // init instance
        {
            std::lock_guard lck{detail::instance_mtx};
            if (!detail::vk_instance) {
#ifdef NDEBUG
                constexpr bool enableValidation = false;
#else
                constexpr bool enableValidation = true;
#endif
                detail::vk_instance = detail::create_instance(enableValidation);
            }
        }
        _init_device(device_idx);
    }
    // auto exts = detail::supported_exts(physical_device());
    // for(auto&& i : exts){
    //     LUISA_INFO("{}", i.extensionName);
    // }
    auto ctx_inst = context();
    {
        std::lock_guard lck(gDxcMutex);
        if (gDxcRefCount == 0) {
            gDxcCompiler.create(ctx.runtime_directory());
        }
        gDxcRefCount++;
    }
    if (!_binary_io) {
        _default_file_io = vstd::make_unique<DefaultBinaryIO>(std::move(ctx_inst));
        _binary_io = _default_file_io.get();
    }
}
void Device::_init_device(uint32_t selectedDevice) {
    VkResult err;

    // If requested, we enable the default validation layers for debugging
    if (detail::settings.validation) {
        detail::setupDebugging(detail::vk_instance);
    }

    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, nullptr));
    if (gpuCount == 0) {
        LUISA_ERROR("No device with Vulkan support found");
        return;
    }
    vstd::vector<VkPhysicalDevice> physical_devices;
    // Enumerate devices
    physical_devices.push_back_uninitialized(gpuCount);
    err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, physical_devices.data());
    if (err) {
        LUISA_ERROR("Could not enumerate physical devices : {}", err);
        return;
    }

    // GPU selection

    // Select physical device to be used for the Vulkan example
    // Defaults to the first device unless specified by command line

    auto physicalDevice = physical_devices[selectedDevice];

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    vkGetPhysicalDeviceProperties(physicalDevice, &_device_properties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &_device_features);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &_device_memory_properties);

    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation

    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    _vk_device.create(physicalDevice);
    _enable_device_exts.emplace_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    _enable_device_exts.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    _enable_device_exts.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    _enable_device_exts.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    VkPhysicalDeviceDescriptorIndexingFeatures enableBindlessFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        .shaderInputAttachmentArrayDynamicIndexing = VK_TRUE,
        .shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
        .shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
        .shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
        .shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE,
        .shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE};
    VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQueryFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = &enableBindlessFeatures,
        .rayQuery = VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &enabledRayQueryFeatures,
        .accelerationStructure = VK_TRUE};

    VK_CHECK_RESULT(_vk_device->createLogicalDevice(_device_features, _enable_device_exts, &enabledAccelerationStructureFeatures));
    auto device = _vk_device->logicalDevice;
    // Get a graphics queue from the device
    vkGetDeviceQueue(device, _vk_device->queueFamilyIndices.graphics, 0, &_graphics_queue);
    vkGetDeviceQueue(device, _vk_device->queueFamilyIndices.compute, 0, &_compute_queue);
    vkGetDeviceQueue(device, _vk_device->queueFamilyIndices.transfer, 0, &_copy_queue);
    _pso_header.headerSize = sizeof(VkPipelineCacheHeaderVersionOne);
    _pso_header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    _pso_header.vendorID = _vk_device->properties.vendorID;
    _pso_header.deviceID = _vk_device->properties.deviceID;
    memcpy(_pso_header.pipelineCacheUUID, _vk_device->properties.pipelineCacheUUID, VK_UUID_SIZE);
    _allocator.create(*this);
}
bool Device::is_pso_same(VkPipelineCacheHeaderVersionOne const &pso) {
    return memcmp(&pso, &_pso_header, sizeof(VkPipelineCacheHeaderVersionOne)) == 0;
}
Device::~Device() {
    std::lock_guard lck(gDxcMutex);
    if (--gDxcRefCount == 0) {
        gDxcCompiler.destroy();
    }
}
void *Device::native_handle() const noexcept { return _vk_device->logicalDevice; }
BufferCreationInfo Device::create_buffer(const Type *element, size_t elem_count) noexcept { return BufferCreationInfo::make_invalid(); }
BufferCreationInfo Device::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept { return BufferCreationInfo::make_invalid(); }
void Device::destroy_buffer(uint64_t handle) noexcept {}

// texture
ResourceCreationInfo Device::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept { return ResourceCreationInfo::make_invalid(); }
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
SwapchainCreationInfo Device::create_swapchain(
    uint64_t window_handle, uint64_t stream_handle,
    uint width, uint height, bool allow_hdr,
    bool vsync, uint back_buffer_size) noexcept { return SwapchainCreationInfo{ResourceCreationInfo::make_invalid()}; }
void Device::destroy_swap_chain(uint64_t handle) noexcept {}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {}

// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    ShaderCreationInfo info;
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= 1;
    }
    if (option.enable_debug_info) {
        mask |= 2;
    }
    // Clock clk;
    auto code = hlsl::CodegenUtility{}.Codegen(kernel, _binary_io, option.native_include, mask, true);
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    if (option.compile_only) {
        assert(!option.name.empty());
        auto comp_result = Device::Compiler()->compile_compute(
            code.result.view(),
            true,
            k_shader_model,
            option.enable_fast_math,
            true);
        comp_result.multi_visit(
            [&](vstd::unique_ptr<hlsl::DxcByteBlob> const &buffer) {
                ShaderSerializer::serialize_bytecode(
                    code.properties,
                    check_md5,
                    code.typeMD5,
                    kernel.block_size(),
                    option.name,
                    {reinterpret_cast<const uint *>(buffer->data()), buffer->size() / sizeof(uint)},
                    SerdeType::ByteCode,
                    _binary_io);
            },
            [](auto &&err) {
                LUISA_ERROR("Compile Error: {}", err);
                return nullptr;
            });

    } else {
        vstd::string_view file_name;
        vstd::string str_cache;
        SerdeType serde_type;
        if (option.enable_cache) {
            if (option.name.empty()) {
                str_cache << check_md5.to_string(false) << ".dxil"sv;
                file_name = str_cache;
                serde_type = SerdeType::Cache;
            } else {
                file_name = option.name;
                serde_type = SerdeType::ByteCode;
            }
        }
        auto shader = ComputeShader::compile(
            _binary_io,
            this,
            kernel,
            [&]() { return std::move(code); },
            check_md5,
            hlsl::binding_to_arg(kernel.bound_arguments()),
            kernel.block_size(),
            file_name,
            serde_type,
            k_shader_model,
            option.enable_fast_math);
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pipeline();
    }
    info.block_size = kernel.block_size();
    return info;
}
ShaderCreationInfo Device::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept { return ShaderCreationInfo::make_invalid(); }
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept { return ShaderCreationInfo::make_invalid(); }
Usage Device::shader_argument_usage(uint64_t handle, size_t index) noexcept { return Usage::NONE; }
void Device::destroy_shader(uint64_t handle) noexcept {
    delete reinterpret_cast<ComputeShader *>(handle);
}

// event
ResourceCreationInfo Device::create_event() noexcept { return ResourceCreationInfo::make_invalid(); }
void Device::destroy_event(uint64_t handle) noexcept {}
void Device::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {}
void Device::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {}
void Device::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {}
void Device::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {}
bool Device::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {return false;}
VSTL_EXPORT_C void backend_device_names(luisa::vector<luisa::string> &r) {
    {
        std::lock_guard lck{detail::instance_mtx};
        if (!detail::vk_instance) {
#ifdef NDEBUG
            constexpr bool enableValidation = false;
#else
            constexpr bool enableValidation = true;
#endif
            detail::vk_instance = detail::create_instance(enableValidation);
        }
    }
    vstd::vector<VkPhysicalDevice> physical_devices;
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, nullptr));
    if (gpuCount == 0) {
        return;
    }
    // Enumerate devices
    physical_devices.push_back_uninitialized(gpuCount);
    auto err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, physical_devices.data());
    if (err) {
        LUISA_ERROR("Could not enumerate physical devices : {}", err);
        return;
    }
    r.reserve(physical_devices.size());
    VkPhysicalDeviceProperties _device_properties;
    for (auto &&i : physical_devices) {
        vkGetPhysicalDeviceProperties(i, &_device_properties);
        r.emplace_back(_device_properties.deviceName);
    }
}
hlsl::ShaderCompiler *Device::Compiler() {
    return gDxcCompiler.ptr();
}
VkInstance Device::instance() const {
    return detail::vk_instance;
}

VSTL_EXPORT_C DeviceInterface *create(Context &&c, DeviceConfig const *settings) {
    return new Device(std::move(c), settings);
}
VSTL_EXPORT_C void destroy(DeviceInterface *device) {
    delete static_cast<Device *>(device);
}
}// namespace lc::vk
