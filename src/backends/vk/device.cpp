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
#include "builtin_kernel.h"
#include "shader_serializer.h"
#include "default_buffer.h"
#include "stream.h"
#include "event.h"
#include "texture.h"
#include "bindless_array.h"
#include "blas.h"
#include "tlas.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include "pinned_memory_ext.h"
#include "vk_raster_ext.h"
#include "vk_native_res_ext.h"
#include <luisa/backends/ext/raster_ext_interface.h>
#ifdef LCVK_ENABLE_CUDA
#include "vk_cuda_interop_ext.h"
#endif
namespace lc::vk {
static constexpr uint k_shader_model = 65u;

using namespace std::string_literals;
static luisa::spin_mutex gDxcMutex;
static vstd::StackObject<hlsl::ShaderCompiler, false> gDxcCompiler;
static int32 gDxcRefCount = 0;

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
struct AllocCallbacks {
    VkAllocationCallbacks callbacks;
    AllocCallbacks() {
        callbacks.pfnAllocation = [](
                                      void *pUserData,
                                      size_t size,
                                      size_t alignment,
                                      VkSystemAllocationScope allocationScope) -> void * {
            return luisa::detail::allocator_allocate(size, alignment);
        };
        callbacks.pfnFree = [](void *pUserData,
                               void *pMemory) {
            luisa::detail::allocator_deallocate(pMemory, 0);
        };
        callbacks.pfnReallocation = [](
                                        void *pUserData,
                                        void *pOriginal,
                                        size_t size,
                                        size_t alignment,
                                        VkSystemAllocationScope allocationScope) -> void * {
            return luisa::detail::allocator_reallocate(pOriginal, size, alignment);
        };
    }
};
static AllocCallbacks alloc;
struct InstanceDestructor {
    ~InstanceDestructor() {
        if (vk_instance) {
            vkDestroyInstance(vk_instance, Device::alloc_callbacks());
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
    VkResult result = vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsMessengerCI, Device::alloc_callbacks(), &debugUtilsMessenger);
    assert(result == VK_SUCCESS);
}
vstd::vector<VkExtensionProperties> supported_exts(VkPhysicalDevice physical_device) {
    uint extensions_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, nullptr);
    vstd::vector<VkExtensionProperties> props;
    luisa::enlarge_by(props, extensions_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, props.data());
    return props;
}
void create_instance(bool enableValidation, VkInstance &instance) {
    volkInitialize();
    if (!instance) {
        vstd::vector<const char *> instance_exts;
        instance_exts.reserve(8);
        vstd::unordered_set<vstd::string> supported_instance_exts;

        // Validation can also be forced via a define
        settings.validation = enableValidation;

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "luisa_compute";
        appInfo.pEngineName = appInfo.pApplicationName;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        // Enable surface extensions depending on os
        instance_exts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        instance_exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
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
        if (std::find(instance_exts.begin(), instance_exts.end(), VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == instance_exts.end()) {
            instance_exts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
#endif
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
        VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, Device::alloc_callbacks(), &instance));
    }
    volkLoadInstance(instance);
}

}// namespace detail
VkAllocationCallbacks *Device::alloc_callbacks() {
    return &detail::alloc.callbacks;
}
//////////////// Not implemented area
ResourceCreationInfo Device::create_mesh(
    const AccelOption &option) noexcept {
    auto mesh = new Blas(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(mesh),
        .native_handle = mesh->accel()};
}
void Device::destroy_mesh(uint64_t handle) noexcept {
    delete reinterpret_cast<Blas *>(handle);
}

ResourceCreationInfo Device::create_procedural_primitive(
    const AccelOption &option) noexcept {
    return create_mesh(option);
}

uint Device::compute_warp_size() const noexcept {
    return 32;// TODO
}
uint64_t Device::memory_granularity() const noexcept {
    return sparse_buffer_size;
}

void Device::destroy_procedural_primitive(uint64_t handle) noexcept {
    destroy_mesh(handle);
}

ResourceCreationInfo Device::create_accel(const AccelOption &option) noexcept {
    auto accel = new Tlas(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(accel),
        .native_handle = accel->accel()};
}
void Device::destroy_accel(uint64_t handle) noexcept {
    delete reinterpret_cast<Tlas *>(handle);
}
//////////////// Not implemented area
Device::Device(Context &&ctx_arg, DeviceConfig const *configs)
    : DeviceInterface{std::move(ctx_arg)},
      set_bindless_kernel(BuiltinKernel::LoadBindlessSetKernel),
      set_accel_kernel(BuiltinKernel::LoadAccelSetKernel) {
    bool headless = false;
    bool use_lmdb = false;
    bool load_dxc = true;
    uint device_idx = -1;
    if (configs) {
        if (configs->extension) {
            _config_ext = luisa::unique_ptr<VulkanDeviceConfigExt>{reinterpret_cast<VulkanDeviceConfigExt *>(configs->extension.release())};
        }
        headless = configs->headless;
        use_lmdb = configs->use_lmdb;
        device_idx = configs->device_index;
        _binary_io = configs->binary_io;
    }
    VkPhysicalDevice ext_phy_device{};
    VkDevice ext_device{};
    if (_config_ext) {
        auto external_device = _config_ext->create_external_device();
        ext_phy_device = external_device.physical_device;
        ext_device = external_device.device;
        if ((ext_phy_device != nullptr) != (ext_device != nullptr)) [[unlikely]] {
            LUISA_ERROR("External physical device must all have instance or all be null.");
        }
        if (external_device.instance) {
            std::lock_guard lck{detail::instance_mtx};
            detail::vk_instance = external_device.instance;
        }
        load_dxc = _config_ext->load_dxc();
        _graphics_queue = external_device.graphics_queue;
        _compute_queue = external_device.compute_queue;
        _copy_queue = external_device.copy_queue;
        _external_instance = external_device.instance;
        _external_device = external_device.device;
        _external_graphics_queue = external_device.graphics_queue;
        _external_compute_queue = external_device.compute_queue;
        _external_copy_queue = external_device.copy_queue;
    }
    Context ctx{this->_ctx_impl};
#ifndef LC_NO_HLSL_BUILTIN
    if (load_dxc) {
        std::lock_guard lck(gDxcMutex);
        if (gDxcRefCount == 0)
            gDxcCompiler.create(ctx.runtime_directory());
        gDxcRefCount++;
    }
#endif
    if (!_binary_io) {
        _default_file_io = vstd::make_unique<DefaultBinaryIO>(context(), headless, use_lmdb);
        _binary_io = _default_file_io.get();
    }
    if (!headless) {
        // init instance
        {
            std::lock_guard lck{detail::instance_mtx};
            if (!detail::vk_instance || _external_instance) {
#ifdef NDEBUG
                constexpr bool enableValidation = false;
#else
                constexpr bool enableValidation = true;
#endif
                detail::create_instance(enableValidation, detail::vk_instance);
            }
        }
        bool fallback = false;
        if (_config_ext) {
            fallback = _config_ext->enable_fallback();
        }
        _init_device(ext_phy_device, ext_device, device_idx, fallback);

        if (_config_ext) {
            _config_ext->readback_vulkan_device(instance(), physical_device(), logic_device(), alloc_callbacks(), _pso_header, _graphics_queue, _compute_queue, _copy_queue, gDxcCompiler->compiler(), gDxcCompiler->library(), gDxcCompiler->utils());
        }
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{PinnedMemoryExt::name},
#else
            PinnedMemoryExt::name,
#endif
            [](Device *device) -> DeviceExtension * {
                return new VkPinnedMemoryExt(device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<VkPinnedMemoryExt *>(ext);
            });
    }
    exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{RasterExt::name},
#else
        RasterExt::name,
#endif
        [](Device *device) -> DeviceExtension * {
            return new VkRasterExt(device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<VkRasterExt *>(ext);
        });
    exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{NativeResourceExt::name},
#else
        NativeResourceExt::name,
#endif
        [](Device *device) -> DeviceExtension * {
            return new VkNativeResourceExt(device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<VkNativeResourceExt *>(ext);
        });

#ifdef LCVK_ENABLE_CUDA
    exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{VkCudaInterop::name},
#else
        VkCudaInterop::name,
#endif
        [](Device *device) -> DeviceExtension * {
            return new VkCudaInteropImpl(device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<VkCudaInteropImpl *>(ext);
        });
#endif
    // auto exts = detail::supported_exts(physical_device());
    // for(auto&& i : exts){
    //     LUISA_INFO("{}", i.extensionName);
    // }

    // func_table.init(this);
}

void Device::_init_device(VkPhysicalDevice external_physical_device, VkDevice external_device, uint32_t selectedDevice, bool fallback) {
    VkPhysicalDevice physical_device = external_physical_device;
    if (!physical_device) {
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
        luisa::enlarge_by(physical_devices, gpuCount);
        err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, physical_devices.data());
        if (err) [[unlikely]] {
            LUISA_ERROR("Could not enumerate physical devices : {}", (int)err);
            return;
        }

        // GPU selection

        // Select physical device to be used for the Vulkan example
        // Defaults to the first device unless specified by command line
        if (selectedDevice == -1) {
            selectedDevice = 0;
            for (auto &&i : physical_devices) {
                vkGetPhysicalDeviceProperties(i, &_device_properties);
                luisa::string device_name{_device_properties.deviceName};
                if (device_name.find("GeForce") != luisa::string::npos ||
                    device_name.find("Radeon") != luisa::string::npos ||
                    device_name.find("Arc") != luisa::string::npos) {
                    LUISA_INFO("Select device: {}", device_name);
                    break;
                }
                selectedDevice++;
            }
        }
        physical_device = physical_devices[selectedDevice];
    }

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    vkGetPhysicalDeviceFeatures(physical_device, &_device_features);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &_device_memory_properties);

    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation

    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    _vk_device.create(physical_device);
    _vk_device->logicalDevice = external_device;
    _enable_device_exts.emplace_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (!fallback) {
        // _enable_device_exts.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
#ifdef LCVK_ENABLE_CUDA
        _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef LUISA_PLATFORM_WINDOWS
        _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
        _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
#endif
    }

    VkPhysicalDeviceBufferDeviceAddressFeatures device_buffer_feature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        nullptr,
        VK_TRUE};

    VkPhysicalDeviceDescriptorIndexingFeatures enable_bindless_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        .pNext = &device_buffer_feature,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,

        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,

        .runtimeDescriptorArray = VK_TRUE,
    };

    VkPhysicalDeviceRayQueryFeaturesKHR enable_rayquery_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = &enable_bindless_features,
        .rayQuery = VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabledAccelerationStructureFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &enable_rayquery_features,
        .accelerationStructure = VK_TRUE};

    VkPhysicalDeviceTimelineSemaphoreFeatures enable_timeline_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .pNext = &enabledAccelerationStructureFeatures,
        .timelineSemaphore = VK_TRUE};
    VkPhysicalDeviceSynchronization2Features barrier_feature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        &enable_timeline_feature,
        true};
    void *ext_chain = &barrier_feature;
    VK_CHECK_RESULT(_vk_device->createLogicalDevice(_device_features, _enable_device_exts, ext_chain));
    auto device = _vk_device->logicalDevice;
    volkLoadDevice(device);

    // Get a graphics queue from the device
    if (!_external_graphics_queue)
        vkGetDeviceQueue(device, _vk_device->queueFamilyIndices.graphics, 0, &_graphics_queue);
    if (!_external_compute_queue)
        vkGetDeviceQueue(device, _vk_device->queueFamilyIndices.compute, 0, &_compute_queue);
    if (!_external_copy_queue)
        vkGetDeviceQueue(device, _vk_device->queueFamilyIndices.transfer, 0, &_copy_queue);
    _pso_header.headerSize = sizeof(VkPipelineCacheHeaderVersionOne);
    _pso_header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    _pso_header.vendorID = _vk_device->properties.vendorID;
    _pso_header.deviceID = _vk_device->properties.deviceID;
    memcpy(_pso_header.pipelineCacheUUID, _vk_device->properties.pipelineCacheUUID, VK_UUID_SIZE);
    _allocator.create(*this);
    // bind desc_pool
    VkDescriptorBindingFlags desc_binding_flag =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindless_binding_flags{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 1,
        .pBindingFlags = &desc_binding_flag};
    // bindless buffer desc_pool
    {
        buffer_heap_pool.full_size = 262144;
        VkDescriptorPoolSize pool_size;
        pool_size.descriptorCount = buffer_heap_pool.full_size;
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size};
        VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &createInfo, alloc_callbacks(), &_bdls_buffer_desc_pool));
        VkDescriptorSetLayoutBinding binding{
            0,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            buffer_heap_pool.full_size,
            VK_SHADER_STAGE_ALL,
            nullptr};
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &bindless_binding_flags,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = 1,
            .pBindings = &binding};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptorLayout, alloc_callbacks(), &_bdls_buffer_set_layout));
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _bdls_buffer_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &_bdls_buffer_set_layout};
        VK_CHECK_RESULT(vkAllocateDescriptorSets(logic_device(), &alloc_info, &_bdls_buffer_set));
    }
    // bindless tex2d desc_pool
    {
        tex2d_heap_pool.full_size = 262144;
        VkDescriptorPoolSize pool_size;
        pool_size.descriptorCount = tex2d_heap_pool.full_size;
        pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size};
        VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &createInfo, alloc_callbacks(), &_bdls_tex2d_desc_pool));
        VkDescriptorSetLayoutBinding binding{
            0,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            tex2d_heap_pool.full_size,
            VK_SHADER_STAGE_ALL,
            nullptr};
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &bindless_binding_flags,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = 1,
            .pBindings = &binding};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptorLayout, alloc_callbacks(), &_bdls_tex2d_set_layout));
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _bdls_tex2d_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &_bdls_tex2d_set_layout};
        VK_CHECK_RESULT(vkAllocateDescriptorSets(logic_device(), &alloc_info, &_bdls_tex2d_set));
        tex2d_bindless_imgview.resize(tex2d_heap_pool.full_size);
    }
    // bindless tex3d desc_pool
    {
        tex3d_heap_pool.full_size = 262144;
        VkDescriptorPoolSize pool_size;
        pool_size.descriptorCount = tex3d_heap_pool.full_size;
        pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size};
        VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &createInfo, alloc_callbacks(), &_bdls_tex3d_desc_pool));
        VkDescriptorSetLayoutBinding binding{
            0,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            tex3d_heap_pool.full_size,
            VK_SHADER_STAGE_ALL,
            nullptr};
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &bindless_binding_flags,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = 1,
            .pBindings = &binding};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptorLayout, alloc_callbacks(), &_bdls_tex3d_set_layout));
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _bdls_tex3d_desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &_bdls_tex3d_set_layout};
        VK_CHECK_RESULT(vkAllocateDescriptorSets(logic_device(), &alloc_info, &_bdls_tex3d_set));
        tex3d_bindless_imgview.resize(tex3d_heap_pool.full_size);
    }
    // sampler desc_pool
    {
        VkDescriptorPoolSize pool_size;
        pool_size.descriptorCount = 16;
        pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        VkDescriptorPoolCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size};
        VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &createInfo, alloc_callbacks(), &_sampler_pool));
        _samplers.resize(16);
        size_t idx = 0;
        for (auto x : vstd::range(4))
            for (auto y : vstd::range(4)) {
                auto d = vstd::scope_exit([&] { ++idx; });
                VkSamplerCreateInfo info{
                    VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    nullptr,
                    0};

                switch ((Sampler::Filter)y) {
                    case Sampler::Filter::POINT:
                        info.minFilter = VK_FILTER_NEAREST;
                        info.magFilter = VK_FILTER_NEAREST;
                        break;
                    case Sampler::Filter::LINEAR_POINT:
                        info.minFilter = VK_FILTER_LINEAR;
                        info.magFilter = VK_FILTER_LINEAR;
                        break;
                    case Sampler::Filter::LINEAR_LINEAR:
                        info.minFilter = VK_FILTER_LINEAR;
                        info.magFilter = VK_FILTER_LINEAR;
                        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                        break;
                    case Sampler::Filter::ANISOTROPIC:
                        info.minFilter = VK_FILTER_LINEAR;
                        info.magFilter = VK_FILTER_LINEAR;
                        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                        info.anisotropyEnable = VK_TRUE;
                        info.maxAnisotropy = 16;
                        break;
                    default: LUISA_ASSUME(false); break;
                }

                VkSamplerAddressMode address = [&] {
                    switch ((Sampler::Address)x) {
                        case Sampler::Address::EDGE:
                            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                        case Sampler::Address::REPEAT:
                            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                        case Sampler::Address::MIRROR:
                            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                        default:
                            info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                    }
                }();
                info.addressModeU = address;
                info.addressModeV = address;
                info.addressModeW = address;

                info.mipLodBias = 0;
                info.minLod = 0;
                info.maxLod = VK_LOD_CLAMP_NONE;
                VK_CHECK_RESULT(vkCreateSampler(logic_device(), &info, alloc_callbacks(), &_samplers[idx]));
            }
        VkDescriptorSetLayoutBinding binding{
            0,
            VK_DESCRIPTOR_TYPE_SAMPLER,
            16,
            VK_SHADER_STAGE_ALL,
            _samplers.data()};
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptorLayout, alloc_callbacks(), &_sampler_set_layout));
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _sampler_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &_sampler_set_layout};
        VK_CHECK_RESULT(vkAllocateDescriptorSets(logic_device(), &alloc_info, &_sampler_set));
    }
}
bool Device::is_pso_same(VkPipelineCacheHeaderVersionOne const &pso) {
    return std::memcmp(&pso, &_pso_header, sizeof(VkPipelineCacheHeaderVersionOne)) == 0;
}
Device::~Device() {
    if (_vk_device) {
        vkDestroyDescriptorSetLayout(logic_device(), _sampler_set_layout, alloc_callbacks());
        vkDestroyDescriptorSetLayout(logic_device(), _bdls_buffer_set_layout, alloc_callbacks());
        vkDestroyDescriptorSetLayout(logic_device(), _bdls_tex2d_set_layout, alloc_callbacks());
        vkDestroyDescriptorSetLayout(logic_device(), _bdls_tex3d_set_layout, alloc_callbacks());
        vkDestroyDescriptorPool(logic_device(), _sampler_pool, alloc_callbacks());
        vkDestroyDescriptorPool(logic_device(), _bdls_tex3d_desc_pool, alloc_callbacks());
        vkDestroyDescriptorPool(logic_device(), _bdls_tex2d_desc_pool, alloc_callbacks());
        vkDestroyDescriptorPool(logic_device(), _bdls_buffer_desc_pool, alloc_callbacks());
        for (auto &i : tex2d_bindless_imgview) {
            if (i) vkDestroyImageView(logic_device(), i, alloc_callbacks());
        }
        for (auto &i : tex3d_bindless_imgview) {
            if (i) vkDestroyImageView(logic_device(), i, alloc_callbacks());
        }
        for (auto &i : _samplers) {
            vkDestroySampler(logic_device(), i, alloc_callbacks());
        }
    }
    _default_file_io = nullptr;
#ifndef LC_NO_HLSL_BUILTIN
    if (gDxcCompiler) {
        std::lock_guard lck(gDxcMutex);
        if (--gDxcRefCount == 0) {
            gDxcCompiler.destroy();
        }
    }
#endif
    if (_external_device) {
        _vk_device->logicalDevice = nullptr;
        _vk_device->physicalDevice = nullptr;
    }
}
void *Device::native_handle() const noexcept { return _vk_device->logicalDevice; }
BufferCreationInfo Device::create_buffer(const Type *element, size_t elem_count, void *external_ptr) noexcept {
    if (element->is_custom()) [[unlikely]] {
        LUISA_ERROR("Indirect buffer not supported.");
    }
    BufferCreationInfo info;
    info.element_stride = (element == Type::of<void>()) ? 1 : element->size();
    auto ptr = new DefaultBuffer(this, info.element_stride * elem_count, true);
    info.handle = reinterpret_cast<uint64_t>(ptr);
    info.native_handle = ptr->vk_buffer();
    info.total_size_bytes = ptr->byte_size();
    return info;
}
BufferCreationInfo Device::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_ptr) noexcept { return BufferCreationInfo::make_invalid(); }
void Device::destroy_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<Buffer *>(handle);
}

// texture
ResourceCreationInfo Device::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth, uint mipmap_levels,
    void *, bool simultaneous_access, bool allow_raster_target) noexcept {

    auto ptr = new Texture(
        this,
        dimension,
        format,
        uint3(width, height, depth),
        mipmap_levels,
        simultaneous_access,
        allow_raster_target);
    ResourceCreationInfo r{
        .handle = reinterpret_cast<uint64_t>(ptr),
        .native_handle = ptr->vk_image()};
    return r;
}
void Device::destroy_texture(uint64_t handle) noexcept {
    delete reinterpret_cast<Texture *>(handle);
}
luisa::FirstFit::Node *Device::HeapAlloc::sub_alloc(uint32_t size) {
    auto ptr = sub_allocator.allocate_best_fit(size);
    if (ptr->offset() + ptr->size() > (full_size - count)) [[unlikely]] {
        vengine_log("bindless allocator out or range!\n");
    }
    return ptr;
}
void Device::HeapAlloc::free(luisa::FirstFit::Node *ptr) {
    sub_allocator.free(ptr);
}
uint Device::HeapAlloc::get_index(luisa::FirstFit::Node const *ptr) const {
    return full_size - (ptr->offset() + ptr->size());
}
// bindless array
ResourceCreationInfo Device::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    auto r = new BindlessArray(this, type, size);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(r),
        .native_handle = &r->indices_buffer()};
}
void Device::destroy_bindless_array(uint64_t handle) noexcept {
    delete reinterpret_cast<BindlessArray *>(handle);
}

// stream
ResourceCreationInfo Device::create_stream(StreamTag stream_tag) noexcept {
    auto ptr = new Stream(this, stream_tag);
    ResourceCreationInfo info{
        .handle = reinterpret_cast<uint64_t>(ptr),
        .native_handle = ptr->queue()};
    return info;
}
void Device::destroy_stream(uint64_t handle) noexcept {
    delete reinterpret_cast<Stream *>(handle);
}
void Device::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->sync();
}
void Device::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->dispatch(list.commands(), list.steal_callbacks(), list.presents(), inqueue_limit);
}

// swap chain
SwapchainCreationInfo Device::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
    auto ptr = new Swapchain(this);
    ptr->create_swapchain(
        option.display,
        option.window,
        option.size.x,
        option.size.y,
        option.back_buffer_count + 1,
        false,
        option.wants_hdr,
        option.wants_vsync);
    SwapchainCreationInfo r;
    r.handle = reinterpret_cast<uint64_t>(ptr);
    r.storage = option.wants_hdr ? PixelStorage::HALF4 : PixelStorage::BYTE4;
    r.native_handle = ptr->swapchain();
    return r;
}
void Device::destroy_swap_chain(uint64_t handle) noexcept {
    delete reinterpret_cast<Swapchain *>(handle);
}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->present(reinterpret_cast<Texture const *>(image_handle), 0, reinterpret_cast<Swapchain *>(swapchain_handle), inqueue_limit);
}

// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    LUISA_ASSERT(Device::Compiler(), "Shader compiler not loaded.");
    ShaderCreationInfo info;
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= 1;
    }
    if (option.enable_debug_info) {
        mask |= 2;
    }
    // Clock clk;
    auto code = hlsl::CodegenUtility{}.Codegen(kernel, option.native_include, mask, true);
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    if (option.compile_only) {
        assert(!option.name.empty());
        info.invalidate();
        auto comp_result = Device::Compiler()->compile_compute(
            code.result.view(),
            !option.enable_debug_info,
            k_shader_model,
            option.enable_fast_math,
            true,
            option.enable_debug_info);
        comp_result.multi_visit(
            [&](hlsl::ComUniquePtr<IDxcBlob> const &buffer) {
                auto saved_args = ShaderSerializer::serialize_saved_args(kernel);
                ShaderSerializer::serialize_bytecode(
                    code.properties,
                    saved_args,
                    check_md5,
                    code.typeMD5,
                    kernel.block_size(),
                    option.name,
                    {reinterpret_cast<const uint *>(buffer->GetBufferPointer()), buffer->GetBufferSize() / sizeof(uint)},
                    SerdeType::ByteCode,
                    _binary_io, code.useTex2DBindless,
                    code.useTex3DBindless,
                    code.useBufferBindless,
                    code.printers);
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
                str_cache << check_md5.to_string(false) << ".spv"sv;
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
            ShaderSerializer::serialize_saved_args(kernel),
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
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    ShaderCreationInfo info;
    auto deser_result = ShaderSerializer::try_deser_compute(this, {}, {}, name, SerdeType::ByteCode, _binary_io);
    if (!deser_result.shader) {
        info.invalidate();
        return info;
    }
    if (!ComputeShader::verify_type_md5(arg_types, deser_result.type_md5)) {
        LUISA_WARNING("Shader {} arguments not match.", name);
        info.invalidate();
        return info;
    }
    auto shader = static_cast<ComputeShader *>(deser_result.shader);
    info.handle = reinterpret_cast<uint64_t>(deser_result.shader);
    info.native_handle = shader->pipeline();
    info.block_size = shader->block_size();
    return info;
}
Usage Device::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    auto shader = reinterpret_cast<Shader const *>(handle);
    return shader->saved_arguments()[index].varUsage;
}
void Device::destroy_shader(uint64_t handle) noexcept {
    delete reinterpret_cast<ComputeShader *>(handle);
}

// event
ResourceCreationInfo Device::create_event() noexcept {
    auto ptr = new Event(this);
    ResourceCreationInfo r{
        .handle = reinterpret_cast<uint64_t>(ptr),
        .native_handle = ptr->semaphore()};
    return r;
}
void Device::destroy_event(uint64_t handle) noexcept {
    delete reinterpret_cast<Event *>(handle);
}
void Device::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->signal(reinterpret_cast<Event *>(handle), fence_value);
}
void Device::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->wait(reinterpret_cast<Event *>(handle), fence_value);
}
void Device::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {
    reinterpret_cast<Event *>(handle)->sync(fence_value);
}
void Device::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {}
bool Device::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    return reinterpret_cast<Event *>(handle)->is_complete(fence_value);
}
VSTL_EXPORT_C void backend_device_names(luisa::vector<luisa::string> &r) {
    {
        std::lock_guard lck{detail::instance_mtx};
        if (!detail::vk_instance) {
#ifdef NDEBUG
            constexpr bool enableValidation = false;
#else
            constexpr bool enableValidation = true;
#endif
            detail::create_instance(enableValidation, detail::vk_instance);
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
    luisa::enlarge_by(physical_devices, gpuCount);
    auto err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpuCount, physical_devices.data());
    if (err) {
        LUISA_ERROR("Could not enumerate physical devices : {}", (int)err);
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
    return gDxcCompiler ? gDxcCompiler.ptr() : nullptr;
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
uint Device::HeapAlloc::alloc() {
    std::lock_guard lck{mtx};
    if (release_pool.empty()) {
        return count++;
    }
    auto r = release_pool.back();
    release_pool.pop_back();
    return r;
}
void Device::HeapAlloc::dealloc(uint idx) {
    std::lock_guard lck{mtx};
    release_pool.emplace_back(idx);
}
Device::HeapAlloc::HeapAlloc() : sub_allocator(std::numeric_limits<uint32_t>::max(), 1) {}
Device::HeapAlloc::~HeapAlloc() = default;
Device::LazyLoadShader::LazyLoadShader(LoadFunc loadFunc) : loadFunc(loadFunc) {}
Device::LazyLoadShader::~LazyLoadShader() {}
ComputeShader *Device::LazyLoadShader::Get(Device *self) {
    if (!shader) {
        shader = vstd::create_unique(loadFunc(self));
    }
    return shader.get();
}
bool Device::LazyLoadShader::Check(Device *self) {
    if (shader) return true;
    shader = vstd::create_unique(loadFunc(self));
    if (shader) {
        auto afterExit = vstd::scope_exit([&] { shader = nullptr; });
        return true;
    }
    return false;
}
ResourceCreationInfo Device::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    VkMemoryRequirements req{
        .size = byte_size,
        .alignment = sparse_buffer_size,
        .memoryTypeBits = std::numeric_limits<uint>::max()};
    auto allocation = vengine_new<std::pair<VmaAllocation, VmaAllocationInfo>>();
    VmaAllocationCreateInfo allocInfo = {
        .flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
        .usage = VMA_MEMORY_USAGE_GPU_ONLY};
    _allocator->alloc_sparse(req, &allocInfo, allocation->first, &allocation->second);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(allocation),
        .native_handle = allocation->first};
}
void Device::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    auto ptr = reinterpret_cast<std::pair<VmaAllocation, VmaAllocationInfo> *>(handle);
    _allocator->dealloc_sparse(ptr->first);
    vengine_delete(ptr);
}
ResourceCreationInfo Device::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    return allocate_sparse_texture_heap(byte_size);
}
void Device::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    deallocate_sparse_texture_heap(handle);
}
void Device::update_sparse_resources(
    uint64_t stream_handle,
    luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->update_sparse_resources(std::move(textures_update));
}
SparseBufferCreationInfo Device::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    if (element->is_custom()) [[unlikely]] {
        LUISA_ERROR("Indirect buffer not supported.");
    }
    SparseBufferCreationInfo info;
    auto ptr = new SparseBuffer(this, element->size() * elem_count, true);
    info.element_stride = (element == Type::of<void>()) ? 1 : element->size();
    info.handle = reinterpret_cast<uint64_t>(ptr);
    info.native_handle = ptr->vk_buffer();
    info.total_size_bytes = ptr->byte_size();
    info.tile_size_bytes = sparse_buffer_size;
    return info;
}
SparseTextureCreationInfo Device::create_sparse_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    auto ptr = new Texture(this);
    ptr->init_as_sparse(dimension, format, uint3(width, height, depth), mipmap_levels, simultaneous_access);
    SparseTextureCreationInfo r;
    r.handle = reinterpret_cast<uint64_t>(ptr);
    r.native_handle = ptr->vk_image();
    r.tile_size_bytes = sparse_buffer_size;
    r.tile_size = [&]() {
        if (dimension == 2) {
            return make_uint3(Texture::tex2d_tile_size(pixel_format_to_storage(format)), 1);
        } else {
            return Texture::tex3d_tile_size(pixel_format_to_storage(format));
        }
    }();
    return r;
}
DeviceExtension *Device::extension(vstd::string_view name) noexcept {
    auto ite = exts.find(name);
    if (ite == exts.end()) return nullptr;
    auto &v = ite->second;
    {
        std::lock_guard lck{ext_mtx};
        if (v.ext == nullptr) {
            v.ext = v.ctor(this);
        }
    }
    return v.ext;
}
void Device::destroy_sparse_texture(uint64_t handle) noexcept {
    delete reinterpret_cast<Texture *>(handle);
}
void Device::destroy_sparse_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<SparseBuffer *>(handle);
}
void Device::set_stream_log_callback(uint64_t stream_handle,
                                     const StreamLogCallback &callback) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->logger = callback;
}
}// namespace lc::vk
