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
#include "motion_instance.h"
#include "rt_shader.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include "pinned_memory_ext.h"
#include "vk_raster_ext.h"
#include "vk_native_res_ext.h"
#include <luisa/backends/ext/raster_ext_interface.h>
#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
#include "vk_cuda_interop_ext.h"
#endif
#ifdef LUISA_XIR_TO_SPIRV
#include <spirv_codegen/entry.h>
#include <spirv_codegen/utils.h>
#include <SPIRV/disassemble.h>
#include <fstream>
#endif

namespace lc::vk {
using namespace std::string_literals;
static luisa::spin_mutex g_dxc_mutex;
static vstd::StackObject<hlsl::ShaderCompiler, false> g_dxc_compiler;
static int32 g_dxc_ref_count = 0;

namespace detail {
struct Settings {
    bool validation{false};
    bool fullscreen{false};
    bool vsync{false};
    bool overlay{true};
};

static VkInstance vk_instance{nullptr};
static std::mutex instance_mtx;
static Settings settings{};
static PFN_vkCreateDebugUtilsMessengerEXT vk_create_debug_utils_messenger_ext;
static PFN_vkDestroyDebugUtilsMessengerEXT vk_destroy_debug_utils_messenger_ext;
static VkDebugUtilsMessengerEXT debug_utils_messenger;
struct AllocCallbacks {
    VkAllocationCallbacks callbacks{};
    AllocCallbacks() {
        callbacks.pfnAllocation = [](
                                      void *p_user_data,
                                      size_t size,
                                      size_t alignment,
                                      VkSystemAllocationScope allocationScope) -> void * {
            return luisa::detail::allocator_allocate(size, alignment);
        };
        callbacks.pfnFree = [](void *p_user_data,
                               void *pMemory) {
            luisa::detail::allocator_deallocate(pMemory, 0);
        };
        callbacks.pfnReallocation = [](
                                        void *p_user_data,
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
VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_messenger_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
    void *p_user_data) {
    // Select prefix depending on flags passed to the callback
    vstd::string prefix;
    bool is_error{false};
    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        prefix = "VERBOSE: ";
    } else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        prefix = "INFO: ";
    } else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        prefix = "WARNING: ";
    } else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        is_error = true;
        prefix = "ERROR: ";
    }

    // Display message to default output (console/logcat)
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        vstd::string debug_message;
        debug_message << prefix << "[" << vstd::to_string(p_callback_data->messageIdNumber) << "][" << p_callback_data->pMessageIdName << "] : " << p_callback_data->pMessage;
        if (is_error)
            LUISA_ERROR("{}", debug_message);
        else
            LUISA_WARNING("{}", debug_message);
    }
    // The return value of this callback controls whether the Vulkan call that caused the validation message will be aborted or not
    // We return VK_FALSE as we DON'T want Vulkan calls that cause a validation message to abort
    // If you instead want to have calls abort, pass in VK_TRUE and the function will return VK_ERROR_VALIDATION_FAILED_EXT
    return VK_FALSE;
}

void setup_debugging(VkInstance instance) {

    vk_create_debug_utils_messenger_ext = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    vk_destroy_debug_utils_messenger_ext = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    VkDebugUtilsMessengerCreateInfoEXT debug_utils_messenger_ci{};
    debug_utils_messenger_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_utils_messenger_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_utils_messenger_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debug_utils_messenger_ci.pfnUserCallback = debug_utils_messenger_callback;
    VkResult result = vk_create_debug_utils_messenger_ext(instance, &debug_utils_messenger_ci, Device::alloc_callbacks(), &debug_utils_messenger);
    assert(result == VK_SUCCESS);
}
vstd::unordered_set<luisa::string> supported_exts(VkPhysicalDevice physical_device) {
    uint extensions_count;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, nullptr);
    vstd::vector<VkExtensionProperties> props;
    luisa::enlarge_by(props, extensions_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, props.data());
    vstd::unordered_set<luisa::string> result;
    result.reserve(props.size());
    for (auto &i : props) {
        result.emplace(i.extensionName);
    }
    return result;
}
void create_instance(bool enable_validation, bool &enable_surface, VkInstance &instance, luisa::filesystem::path const &custom_path, luisa::string_view lib_name, luisa::span<luisa::string const> extra_exts) {
    vks::VulkanDevice::init_volk(custom_path, lib_name);
    if (!instance) {
        vstd::vector<const char *> instance_exts;
        instance_exts.reserve(8);
        vstd::unordered_set<vstd::string> supported_instance_exts;
        // Validation can also be forced via a define
        settings.validation = enable_validation;

        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "luisa_compute";
        app_info.pEngineName = app_info.pApplicationName;
        app_info.apiVersion = VK_API_VERSION_1_3;
        // Get extensions supported by the instance and store for later use
        uint32_t ext_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
        if (ext_count > 0) {
            vstd::vector<VkExtensionProperties> extensions(ext_count);
            if (vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, &extensions.front()) == VK_SUCCESS) {
                supported_instance_exts.reserve(extensions.size());
                for (VkExtensionProperties extension : extensions) {
                    supported_instance_exts.emplace(extension.extensionName);
                }
            }
        }
        // Enable surface extensions depending on os
        auto emplace_instance_ext = [&](const char *name) {
            if (supported_instance_exts.find(name) != supported_instance_exts.end()) {
                instance_exts.push_back(name);
                return true;
            } else {
                return false;
            }
        };
        if (enable_surface) {
            enable_surface &= emplace_instance_ext(VK_KHR_SURFACE_EXTENSION_NAME);
            enable_surface &= emplace_instance_ext(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
#if defined(_WIN32)
            enable_surface &= emplace_instance_ext(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
            enable_surface &= emplace_instance_ext(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
            enable_surface &= emplace_instance_ext(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
            enable_surface &= emplace_instance_ext(VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
            enable_surface &= emplace_instance_ext(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
            enable_surface &= emplace_instance_ext(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
#if LUISA_ENABLE_WAYLAND && !defined(VK_USE_PLATFORM_WAYLAND_KHR)
            emplace_instance_ext("VK_KHR_wayland_surface");
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
            enable_surface &= emplace_instance_ext(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
            enable_surface &= emplace_instance_ext(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
            enable_surface &= emplace_instance_ext(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_HEADLESS_EXT)
            enable_surface &= emplace_instance_ext(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
#endif
        }
        for (auto &i : extra_exts) {
            instance_exts.emplace_back(i.c_str());
        }
#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
        // SRS - When running on iOS/macOS with MoltenVK, enable VK_KHR_get_physical_device_properties2 if not already enabled by the example (required by VK_KHR_portability_subset)
        if (std::find(instance_exts.begin(), instance_exts.end(), VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == instance_exts.end()) {
            emplace_instance_ext(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }
#endif
        VkInstanceCreateInfo instance_create_info = {};
        instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_create_info.pNext = nullptr;
        instance_create_info.pApplicationInfo = &app_info;

        // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
        // Note that on Android this layer requires at least NDK r20
        const char *validation_layer_name = "VK_LAYER_KHRONOS_validation";
        if (settings.validation) {
            // Check if this layer is available at instance level
            uint32_t instance_layer_count;
            vkEnumerateInstanceLayerProperties(&instance_layer_count, nullptr);
            vstd::vector<VkLayerProperties> instance_layer_properties(instance_layer_count);
            vkEnumerateInstanceLayerProperties(&instance_layer_count, instance_layer_properties.data());
            bool validation_layer_present = false;
            for (VkLayerProperties layer : instance_layer_properties) {
                if (strcmp(layer.layerName, validation_layer_name) == 0) {
                    validation_layer_present = true;
                    break;
                }
            }
            if (validation_layer_present) {
                instance_create_info.ppEnabledLayerNames = &validation_layer_name;
                instance_create_info.enabledLayerCount = 1;
            } else {
                LUISA_WARNING("Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled");
                settings.validation = false;
            }
        }

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK)) && defined(VK_KHR_portability_enumeration)
        // SRS - When running on iOS/macOS with MoltenVK and VK_KHR_portability_enumeration is defined and supported by the instance, enable the extension and the flag
        if (supported_instance_exts.find(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == supported_instance_exts.end()) {
            emplace_instance_ext(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instance_create_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif
        if (settings.validation) {
            emplace_instance_ext(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);// SRS - Dependency when VK_EXT_DEBUG_MARKER is enabled
            emplace_instance_ext(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        instance_create_info.enabledExtensionCount = (uint32_t)instance_exts.size();
        instance_create_info.ppEnabledExtensionNames = instance_exts.size() > 0 ? instance_exts.data() : nullptr;
        VK_CHECK_RESULT(vkCreateInstance(&instance_create_info, Device::alloc_callbacks(), &instance));
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
        .native_handle = nullptr};
}
void Device::destroy_mesh(uint64_t handle) noexcept {
    delete reinterpret_cast<Blas *>(handle);
}

ResourceCreationInfo Device::create_procedural_primitive(
    const AccelOption &option) noexcept {
    return create_mesh(option);
}

uint Device::compute_warp_size() const noexcept {
    return 32;
}
uint64_t Device::memory_granularity() const noexcept {
    return kSparseBufferSize;
}

void Device::destroy_procedural_primitive(uint64_t handle) noexcept {
    destroy_mesh(handle);
}

ResourceCreationInfo Device::create_accel(const AccelOption &option) noexcept {
    auto accel = new Tlas(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(accel),
        .native_handle = nullptr};
}
void Device::destroy_accel(uint64_t handle) noexcept {
    delete reinterpret_cast<Tlas *>(handle);
}

ResourceCreationInfo Device::create_motion_instance(const AccelMotionOption &option) noexcept {
    auto instance = new MotionInstance(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(instance),
        .native_handle = nullptr};
}

void Device::destroy_motion_instance(uint64_t handle) noexcept {
    delete reinterpret_cast<MotionInstance *>(handle);
}
//////////////// Not implemented area
Device::Device(Context &&ctx_arg, DeviceConfig const *configs)
    : DeviceInterface{std::move(ctx_arg)},
      set_bindless_kernel(BuiltinKernel::load_bindless_set_kernel),
      set_accel_kernel(BuiltinKernel::load_accel_set_kernel) {
    bool headless = false;
    bool use_lmdb = false;
    bool load_dxc = true;
    uint device_idx = -1;
    if (configs) {
        if (configs->extension) {
            _config_ext = luisa::unique_ptr<VulkanDeviceConfigExt>{reinterpret_cast<VulkanDeviceConfigExt *>(configs->extension.release())};
            _config_ext->get_defragment_function([this] {
                vma_defragment(this);
            });
        }
        headless = configs->headless;
        use_lmdb = configs->use_lmdb;
        device_idx = configs->device_index;
        _binary_io = configs->binary_io;
        _inqueue_limit = configs->inqueue_buffer_limit;
    }
    VkPhysicalDevice ext_phy_device{};
    VkDevice ext_device{};
    luisa::filesystem::path custom_path;
    luisa::string_view lib_name;

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
        auto ext_path = _config_ext->external_vulkan_lib_path();
        custom_path = std::move(ext_path.lib_path);
        lib_name = std::move(ext_path.lib_name);
        _compute_queue = external_device.compute_queue;
        _copy_queue = external_device.copy_queue;
        external_instance = external_device.instance;
        this->external_device = external_device.device;
        external_graphics_queue = external_device.graphics_queue;
        external_compute_queue = external_device.compute_queue;
        external_copy_queue = external_device.copy_queue;
        bindless_enabled = _config_ext->enable_bindless_feature();
        raytracing_enabled = _config_ext->enable_raytracing_feature();
        interop_enabled = _config_ext->enable_interop_feature();
        device_address_enabled = _config_ext->enable_device_address_feature();
        surface_enabled = _config_ext->enable_surface_feature();
    }
    device_address_enabled |= raytracing_enabled;

    Context ctx{this->_ctx_impl};
#ifndef LC_NO_HLSL_BUILTIN
    if (load_dxc) {
        std::lock_guard lck(g_dxc_mutex);
        if (g_dxc_ref_count == 0)
            g_dxc_compiler.create(ctx.runtime_directory(), true);
        g_dxc_ref_count++;
    }
#endif
    if (!_binary_io) {
        _default_file_io = vstd::make_unique<DefaultBinaryIO>(context(), headless, use_lmdb);
        _binary_io = _default_file_io.get();
    }
    if (headless) {
        surface_enabled = false;
    }
    // init instance
    {
        std::lock_guard lck{detail::instance_mtx};
        if (!detail::vk_instance || external_instance) {
#ifdef NDEBUG
            constexpr bool enable_validation = false;
#else
            constexpr bool enable_validation = true;
#endif
            luisa::vector<luisa::string> extra_exts = [&]() {
                if (_config_ext) {
                    return _config_ext->extra_instance_exts();
                } else {
                    return luisa::vector<luisa::string>{};
                }
            }();
            bool enable_surface = surface_enabled;
            detail::create_instance(enable_validation, enable_surface, detail::vk_instance, custom_path, lib_name, extra_exts);
            surface_enabled = enable_surface;
        }
    }
#ifndef LUISA_VULKAN_ENABLE_CUDA_INTEROP
    interop_enabled = false;
#endif
    _init_device(ext_phy_device, ext_device, device_idx);

    if (_config_ext) {
        _config_ext->init_volk(vkGetInstanceProcAddr);
        _config_ext->readback_vulkan_device(instance(), physical_device(), logic_device(), alloc_callbacks(), _pso_header, _graphics_queue, _compute_queue, _copy_queue, graphics_queue_index(), compute_queue_index(), copy_queue_index(), g_dxc_compiler->compiler(), g_dxc_compiler->library(), g_dxc_compiler->utils());
    }
    _exts.try_emplace(
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
    _exts.try_emplace(
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
    _exts.try_emplace(
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

#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
    _exts.try_emplace(
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
    // auto _exts = detail::supported_exts(physical_device());
    // for(auto&& i : _exts){
    //     LUISA_INFO("{}", i.extensionName);
    // }

    // func_table.init(this);
}

void Device::_init_device(VkPhysicalDevice external_physical_device, VkDevice external_device, uint32_t selected_device) {
    VkPhysicalDevice physical_device = external_physical_device;
    if (!physical_device) {
        VkResult err;

        // If requested, we enable the default validation layers for debugging
        if (detail::settings.validation) {
            detail::setup_debugging(detail::vk_instance);
        }

        // Physical device
        uint32_t gpu_count = 0;
        // Get number of available physical devices
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(detail::vk_instance, &gpu_count, nullptr));
        if (gpu_count == 0) {
            LUISA_ERROR("No device with Vulkan support found");
            return;
        }
        vstd::vector<VkPhysicalDevice> physical_devices;
        // Enumerate devices
        luisa::enlarge_by(physical_devices, gpu_count);
        err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpu_count, physical_devices.data());
        if (err) [[unlikely]] {
            LUISA_ERROR("Could not enumerate physical devices : {}", (int)err);
            return;
        }
        if (physical_devices.empty()) [[unlikely]] {
            LUISA_ERROR("Vulkan physical device not found.");
            return;
        }

        // GPU selection

        // Select physical device to be used for the Vulkan example
        // Defaults to the first device unless specified by command line
        VkPhysicalDeviceProperties device_properties;
        if (selected_device == -1) {
            selected_device = 0;
            for (auto &&i : physical_devices) {
                vkGetPhysicalDeviceProperties(i, &device_properties);
                luisa::string device_name{device_properties.deviceName};
                if (device_name.find("GeForce") != luisa::string::npos ||
                    device_name.find("Radeon") != luisa::string::npos ||
                    device_name.find("Arc") != luisa::string::npos) {
                    LUISA_INFO("Select device: {}", device_name);
                    break;
                }
                selected_device++;
            }
        }
        physical_device = physical_devices[std::min<uint32_t>(selected_device, physical_devices.size() - 1)];
    }

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    auto supported_ext = detail::supported_exts(physical_device);
    VkPhysicalDeviceFeatures device_features{};
    vkGetPhysicalDeviceFeatures(physical_device, &device_features);
    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation

    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    _vk_device.create(physical_device);
    _vk_device->logical_device = external_device;
    if (supported_ext.find(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == supported_ext.end()) [[unlikely]] {
        LUISA_ERROR("Necessary extension \"VK_KHR_timeline_semaphore\" is unsupported.");
    }
    if (supported_ext.find(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == supported_ext.end()) [[unlikely]] {
        LUISA_ERROR("Necessary extension \"VK_KHR_synchronization2\" is unsupported.");
    }
    bool enable_16bit = false;
    bool enable_8bit = false;
    bool enable_atomic64_bit = false;
    bool enable_barycentric = false;
    bool enable_motion_blur = false;
    bool enable_float_atomic_add = false;
    bool enable_float_shared_atomic = false;
    if (supported_ext.find(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME) != supported_ext.end()) {
        _enable_device_exts.emplace_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
        enable_barycentric = true;
    }
    if (supported_ext.find(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) != supported_ext.end()) {
        _enable_device_exts.emplace_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
        enable_atomic64_bit = true;
    }
    if (supported_ext.find(VK_KHR_16BIT_STORAGE_EXTENSION_NAME) != supported_ext.end()) {
        _enable_device_exts.emplace_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        enable_16bit = true;
    }
    if (supported_ext.find(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME) != supported_ext.end()) {
        _enable_device_exts.emplace_back(VK_AMD_GPU_SHADER_HALF_FLOAT_EXTENSION_NAME);
        enable_16bit = true;
    }
    if (supported_ext.find(VK_KHR_8BIT_STORAGE_EXTENSION_NAME) != supported_ext.end()) {
        _enable_device_exts.emplace_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
        enable_8bit = true;
    }
    if (supported_ext.find(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT float_atomic_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
            nullptr,
            VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE,
            VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE,
            VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE};
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &float_atomic_features;
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        bool needs_float_atomic_ext = false;
        if (float_atomic_features.shaderBufferFloat32AtomicAdd == VK_TRUE) {
            needs_float_atomic_ext = true;
            enable_float_atomic_add = true;
        }
        if (float_atomic_features.shaderSharedFloat32Atomics == VK_TRUE ||
            float_atomic_features.shaderSharedFloat32AtomicAdd == VK_TRUE) {
            needs_float_atomic_ext = true;
            enable_float_shared_atomic = true;
        }
        if (needs_float_atomic_ext) {
            _enable_device_exts.emplace_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }
    }
    _enable_device_exts.emplace_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    _enable_device_exts.emplace_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    if (bindless_enabled) {
        if (supported_ext.find(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) != supported_ext.end() &&
            supported_ext.find(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) != supported_ext.end()) {
            _enable_device_exts.emplace_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            _enable_device_exts.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        } else {
            bindless_enabled = false;
        }
    }
    if (raytracing_enabled) {
        if (supported_ext.find(VK_KHR_RAY_QUERY_EXTENSION_NAME) != supported_ext.end() &&
            supported_ext.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != supported_ext.end()) {
            _enable_device_exts.emplace_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            _enable_device_exts.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            // Enable motion blur extension if available (NVIDIA only)
            // VK_NV_ray_tracing_motion_blur requires VK_KHR_ray_tracing_pipeline
            if (supported_ext.find(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME) != supported_ext.end() &&
                supported_ext.find(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) != supported_ext.end()) {
                _enable_device_exts.emplace_back(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
                _enable_device_exts.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
                enable_motion_blur = true;
                motion_blur_enabled = true;
            }
        } else {
            raytracing_enabled = false;
        }
    }
    if (interop_enabled) {
        if (supported_ext.find(VK_KHR_RAY_QUERY_EXTENSION_NAME) != supported_ext.end() &&
            supported_ext.find(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) != supported_ext.end()
#ifdef LUISA_PLATFORM_WINDOWS
            && supported_ext.find(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) != supported_ext.end() && supported_ext.find(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME) != supported_ext.end()
#else
            && supported_ext.find(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) != supported_ext.end() && supported_ext.find(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) != supported_ext.end()
#endif
        ) {
            _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
            _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef LUISA_PLATFORM_WINDOWS
            _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
            _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            _enable_device_exts.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
        } else {
            interop_enabled = false;
        }
    }
    luisa::vector<luisa::string> extra_exts = [&]() {
        if (_config_ext) {
            return _config_ext->extra_device_exts();
        } else {
            return luisa::vector<luisa::string>{};
        }
    }();
    for (auto &i : extra_exts) {
        if (supported_ext.find(i) != supported_ext.end())
            _enable_device_exts.emplace_back(i.c_str());
    }
    void *feature_next{nullptr};
    if (_config_ext) {
        feature_next = _config_ext->device_feature_settings();
    }
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR raster_bary{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
        .pNext = feature_next,
        .fragmentShaderBarycentric = VK_TRUE};
    if (enable_barycentric) {
        feature_next = &raster_bary;
    }

    // 16-bit storage features are set in VkPhysicalDeviceVulkan11Features below
    VkPhysicalDeviceRayQueryFeaturesKHR enable_rayquery_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = feature_next,
        .rayQuery = VK_TRUE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabled_acceleration_structure_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &enable_rayquery_features,
        .accelerationStructure = VK_TRUE};
    VkPhysicalDeviceRayTracingMotionBlurFeaturesNV motion_blur_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
        .pNext = &enabled_acceleration_structure_features,
        .rayTracingMotionBlur = VK_TRUE,
        .rayTracingMotionBlurPipelineTraceRaysIndirect = VK_FALSE};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = &motion_blur_features,
        .rayTracingPipeline = VK_TRUE};
    if (raytracing_enabled) {
        if (enable_motion_blur) {
            feature_next = &rt_pipeline_features;
        } else {
            feature_next = &enabled_acceleration_structure_features;
        }
    }
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT float_atomic_feature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        feature_next,
        VK_FALSE,
        enable_float_atomic_add ? VK_TRUE : VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        enable_float_shared_atomic ? VK_TRUE : VK_FALSE,
        enable_float_shared_atomic ? VK_TRUE : VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        VK_FALSE};
    if (enable_float_atomic_add || enable_float_shared_atomic) {
        feature_next = &float_atomic_feature;
    }
    VkPhysicalDeviceSynchronization2Features barrier_feature{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        feature_next,
        true};
    VkPhysicalDeviceVulkan11Features vk11_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &barrier_feature,
        .storageBuffer16BitAccess = enable_16bit ? VK_TRUE : VK_FALSE,
        .uniformAndStorageBuffer16BitAccess = enable_16bit ? VK_TRUE : VK_FALSE,
        .variablePointersStorageBuffer = VK_TRUE};
    auto vk_bindless_enabled = bindless_enabled ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan12Features vk12_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vk11_feature,
        .storageBuffer8BitAccess = enable_8bit ? VK_TRUE : VK_FALSE,
        .shaderBufferInt64Atomics = enable_atomic64_bit ? VK_TRUE : VK_FALSE,
        .shaderSharedInt64Atomics = enable_atomic64_bit ? VK_TRUE : VK_FALSE,
        .shaderFloat16 = enable_16bit ? VK_TRUE : VK_FALSE,
        .shaderInt8 = enable_8bit ? VK_TRUE : VK_FALSE,
        .descriptorIndexing = vk_bindless_enabled,
        .shaderUniformBufferArrayNonUniformIndexing = vk_bindless_enabled,
        .shaderSampledImageArrayNonUniformIndexing = vk_bindless_enabled,
        .shaderStorageBufferArrayNonUniformIndexing = vk_bindless_enabled,
        .shaderStorageImageArrayNonUniformIndexing = vk_bindless_enabled,
        .descriptorBindingUniformBufferUpdateAfterBind = vk_bindless_enabled,
        .descriptorBindingSampledImageUpdateAfterBind = vk_bindless_enabled,
        .descriptorBindingStorageImageUpdateAfterBind = vk_bindless_enabled,
        .descriptorBindingStorageBufferUpdateAfterBind = vk_bindless_enabled,
        .runtimeDescriptorArray = vk_bindless_enabled,

        .shaderSubgroupExtendedTypes = (enable_atomic64_bit || enable_16bit) ? VK_TRUE : VK_FALSE,

        .timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = device_address_enabled ? VK_TRUE : VK_FALSE};
    VK_CHECK_RESULT(_vk_device->create_logical_device(device_features, _enable_device_exts, &vk12_feature, surface_enabled));
    auto device = _vk_device->logical_device;
    volkLoadDevice(device);

    // Get a graphics queue from the device
    if (!external_graphics_queue)
        vkGetDeviceQueue(device, _vk_device->queue_family_indices.graphics, 0, &_graphics_queue);
    if (!external_compute_queue)
        vkGetDeviceQueue(device, _vk_device->queue_family_indices.compute, 0, &_compute_queue);
    if (!external_copy_queue)
        vkGetDeviceQueue(device, _vk_device->queue_family_indices.transfer, 0, &_copy_queue);
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
    if (bindless_enabled) {
        {
            buffer_heap_pool.full_size = 262144;
            VkDescriptorPoolSize pool_size;
            pool_size.descriptorCount = buffer_heap_pool.full_size;
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            VkDescriptorPoolCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                .maxSets = 1,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size};
            VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &create_info, alloc_callbacks(), &_bdls_buffer_desc_pool));
            VkDescriptorSetLayoutBinding binding{
                0,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                buffer_heap_pool.full_size,
                VK_SHADER_STAGE_ALL,
                nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_layout{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &bindless_binding_flags,
                .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                .bindingCount = 1,
                .pBindings = &binding};
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptor_layout, alloc_callbacks(), &_bdls_buffer_set_layout));
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
            VkDescriptorPoolCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                .maxSets = 1,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size};
            VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &create_info, alloc_callbacks(), &_bdls_tex2d_desc_pool));
            VkDescriptorSetLayoutBinding binding{
                0,
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                tex2d_heap_pool.full_size,
                VK_SHADER_STAGE_ALL,
                nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_layout{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &bindless_binding_flags,
                .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                .bindingCount = 1,
                .pBindings = &binding};
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptor_layout, alloc_callbacks(), &_bdls_tex2d_set_layout));
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
            VkDescriptorPoolCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
                .maxSets = 1,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size};
            VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &create_info, alloc_callbacks(), &_bdls_tex3d_desc_pool));
            VkDescriptorSetLayoutBinding binding{
                0,
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                tex3d_heap_pool.full_size,
                VK_SHADER_STAGE_ALL,
                nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_layout{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &bindless_binding_flags,
                .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                .bindingCount = 1,
                .pBindings = &binding};
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptor_layout, alloc_callbacks(), &_bdls_tex3d_set_layout));
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = _bdls_tex3d_desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &_bdls_tex3d_set_layout};
            VK_CHECK_RESULT(vkAllocateDescriptorSets(logic_device(), &alloc_info, &_bdls_tex3d_set));
            tex3d_bindless_imgview.resize(tex3d_heap_pool.full_size);
        }
    }
    // sampler desc_pool
    {
        VkDescriptorPoolSize pool_size;
        pool_size.descriptorCount = 16;
        pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        VkDescriptorPoolCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size};
        VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &create_info, alloc_callbacks(), &_sampler_pool));
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
        VkDescriptorSetLayoutCreateInfo descriptor_layout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &descriptor_layout, alloc_callbacks(), &_sampler_set_layout));
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
    if (g_dxc_compiler) {
        std::lock_guard lck(g_dxc_mutex);
        if (--g_dxc_ref_count == 0) {
            g_dxc_compiler.destroy();
        }
    }
#endif
    if (external_device) {
        _vk_device->logical_device = nullptr;
        _vk_device->physical_device = nullptr;
    }
}
void *Device::native_handle() const noexcept { return _vk_device->logical_device; }
BufferCreationInfo Device::create_buffer(const luisa::compute::Type *element, size_t elem_count, void *external_ptr) noexcept {
    if (element && element->is_custom()) [[unlikely]] {
        LUISA_ERROR("Indirect buffer not supported.");
    }
    BufferCreationInfo info{};
    info.element_stride = (element == Type::of<void>()) ? 1 : element->size();
    DefaultBuffer *ptr;
    if (external_ptr)
        ptr = new DefaultBuffer(this, static_cast<VkBuffer>(external_ptr), nullptr, info.element_stride * elem_count);
    else
        ptr = new DefaultBuffer(this, info.element_stride * elem_count, true);
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
    reinterpret_cast<Stream *>(stream_handle)->dispatch(list.commands(), list.steal_callbacks(), list.presents(), _inqueue_limit);
}

// swap chain
SwapchainCreationInfo Device::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
    auto ptr = new Swapchain(this);
    ptr->create_swapchain(
        option.display,
        option.window,
        option.size.x,
        option.size.y,
        option.back_buffer_count,
        false,
        option.wants_hdr,
        option.wants_vsync, option.wants_transparent);
    SwapchainCreationInfo r{};
    r.handle = reinterpret_cast<uint64_t>(ptr);
    r.storage = ptr->is_hdr() ? PixelStorage::HALF4 : PixelStorage::BYTE4;
    r.native_handle = ptr->swapchain();
    return r;
}
void Device::destroy_swapchain(uint64_t handle) noexcept {
    delete reinterpret_cast<Swapchain *>(handle);
}
void Device::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->present(reinterpret_cast<Texture const *>(image_handle), 0, reinterpret_cast<Swapchain *>(swapchain_handle), _inqueue_limit);
}

static const bool kComputePrintCode = ([] {
    // read env LUISA_DUMP_SOURCE
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    if (env == nullptr) {
        return false;
    }
    return std::string_view{env} == "1";
})();
bool Device::print_code() {
    return kComputePrintCode;
}
// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    LUISA_ASSERT(Device::compiler(), "Shader compiler not loaded.");
    ShaderCreationInfo info;
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= 1;
    }
    if (option.enable_debug_info) {
        mask |= 2;
    }

    // Check if this shader uses motion blur trace operations
    bool requires_motion_blur = kernel.propagated_builtin_callables().uses_raytracing_motion_blur();

    if (requires_motion_blur && !motion_blur_enabled) {
        LUISA_WARNING("Shader uses motion blur but device does not support it. "
                      "Falling back to non-motion compute shader.");
        requires_motion_blur = false;
    }

    if (requires_motion_blur) {
        if (option.compile_only) {
            LUISA_ERROR("compile_only is not yet supported for motion blur shaders.");
        }
        // Use ray tracing pipeline for motion blur shaders
        auto code = hlsl::CodegenUtility{}.RayTracingCodegen(kernel, option.native_include, mask, true);
        vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});

        vstd::string_view file_name;
        vstd::string str_cache;
        SerdeType serde_type = SerdeType::kCache;
        if (option.enable_cache) {
            if (option.name.empty()) {
                str_cache << check_md5.to_string(false) << "_rt.spv"sv;
                file_name = str_cache;
            } else {
                file_name = option.name;
                serde_type = SerdeType::kByteCode;
            }
        }
        auto shader = RayTracingShader::compile(
            _binary_io,
            this,
            ShaderSerializer::serialize_saved_args(kernel),
            [&]() { return std::move(code); },
            check_md5,
            hlsl::binding_to_arg(kernel.bound_arguments()),
            kernel.block_size(),
            file_name,
            serde_type,
            kShaderModel,
            option.enable_fast_math,
            option.enable_debug_info);
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pipeline();
    } else {
#ifdef LUISA_XIR_TO_SPIRV
        static constexpr uint32_t VK_VENDOR_ID_AMD = 0x1002u;
        static constexpr uint32_t VK_VENDOR_ID_NVIDIA = 0x10deu;
        static constexpr uint32_t VK_VENDOR_ID_INTEL = 0x8086u;
        bool use_native_float_atomics = _vk_device->properties.vendorID == VK_VENDOR_ID_NVIDIA;
        auto spv_result = lc::spirv::SpirvCodegenEntry::compile_spirv(kernel, option, use_native_float_atomics);
        for (size_t i = 0; i < spv_result.properties.size(); ++i) {
            auto &p = spv_result.properties[i];
            LUISA_VERBOSE("  prop[{}]: type={}, space={}, reg={}, array_size={}", i, (int)p.type, p.space_index, p.register_index, p.array_size);
        }
        if (print_code()) [[unlikely]] {
            {
                auto filename = luisa::format("spirv_output_{}.spvasm", kernel.name().empty() ? "unnamed" : kernel.name());
                std::ofstream file(filename.c_str(), std::ios::app);
                if (file) {
                    file << "\n; === KERNEL: " << kernel.name() << " hash=" << kernel.hash() << " ===\n";
                    spv::Disassemble(file, spv_result.spv_bin);
                }
                LUISA_VERBOSE("SPIRV printed to {}.", filename);
            }
            // Test HLSL
            auto code = hlsl::CodegenUtility{}.Codegen(kernel, option.native_include, mask, true);
            auto f = fopen("hlsl_output.hlsl", "ab");
            if (f) {
                fwrite(code.result.view().data(), code.result.view().size(), 1, f);
                fclose(f);
            }
            auto comp_result = Device::compiler()->compile_compute(
                code.result.view(),
                !option.enable_debug_info,
                kernel.use_cooperative_operations() ? kTensorShaderModel : (kernel.allowed_warp_size().has_value() ? kHighShaderModel : kShaderModel),
                option.enable_fast_math,
                true,
                option.enable_debug_info);
            if (comp_result.is_type_of<vstd::string>()) [[unlikely]] {
                LUISA_ERROR("HLSL test compilation error: {}", *comp_result.try_get<vstd::string>());
            } else {
                LUISA_VERBOSE("HLSL test compilation successful.");
                {
                    auto filename = luisa::format("spirv_output_hlsl_{}.spvasm", kernel.name().empty() ? "unnamed" : kernel.name());
                    std::ofstream file(filename.c_str(), std::ios::app);
                    if (file) {
                        file << "\n; === KERNEL: " << kernel.name() << " hash=" << kernel.hash() << " ===\n";
                        auto &&blob = comp_result.force_get<lc::hlsl::ComUniquePtr<IDxcBlob>>();
                        std::vector<uint32_t> vec((blob->GetBufferSize() + sizeof(uint) - 1) / sizeof(uint));
                        std::memcpy(vec.data(), blob->GetBufferPointer(), blob->GetBufferSize());
                        spv::Disassemble(file, vec);
                    }
                    LUISA_VERBOSE("SPIRV-HLSL printed to {}.", filename);
                }
            }
        }
        if (option.compile_only) {
            assert(!option.name.empty());
            info.invalidate();
            ShaderSerializer::serialize_bytecode(
                spv_result.properties,
                ShaderSerializer::serialize_saved_args(kernel),
                vstd::MD5{vstd::span<const uint8_t>(reinterpret_cast<const uint8_t *>(spv_result.spv_bin.data()), spv_result.spv_bin.size() * sizeof(uint32_t))},
                hlsl::CodegenUtility::GetTypeMD5(kernel),
                kernel.block_size(),
                option.name,
                spv_result.spv_bin,
                SerdeType::kByteCode,
                _binary_io,
                spv_result.useTex2DBindless,
                spv_result.useTex3DBindless,
                spv_result.useBufferBindless,
                spv_result.printers);
        } else {
            auto shader = new ComputeShader(
                this,
                kernel.block_size(),
                spv_result.properties,
                ShaderSerializer::serialize_saved_args(kernel),
                {reinterpret_cast<const uint *>(spv_result.spv_bin.data()), spv_result.spv_bin.size()},
                hlsl::binding_to_arg(kernel.bound_arguments()),
                {},
                spv_result.useTex2DBindless,
                spv_result.useTex3DBindless,
                spv_result.useBufferBindless,
                std::move(spv_result.printers),
                {spv_result.constant_ubo_data.data(), spv_result.constant_ubo_data.size()});
            LUISA_VERBOSE("ComputeShader created successfully, pipeline: {}", reinterpret_cast<void *>(shader->pipeline()));
            info.handle = reinterpret_cast<uint64_t>(shader);
            info.native_handle = shader->pipeline();
        }

#else
        auto code = hlsl::CodegenUtility{}.Codegen(kernel, option.native_include, mask, true);
        vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
        if (option.compile_only) {
            assert(!option.name.empty());
            info.invalidate();
            if (print_code()) {
                auto f = fopen("hlsl_output.hlsl", "ab");
                fwrite(code.result.view().data(), code.result.view().size(), 1, f);
                fclose(f);
            }
            auto comp_result = Device::compiler()->compile_compute(
                code.result.view(),
                !option.enable_debug_info,
                kernel.use_cooperative_operations() ? kTensorShaderModel : (kernel.allowed_warp_size().has_value() ? kHighShaderModel : kShaderModel),
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
                        SerdeType::kByteCode,
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
                    serde_type = SerdeType::kCache;
                } else {
                    file_name = option.name;
                    serde_type = SerdeType::kByteCode;
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
                kernel.use_cooperative_operations() ? kTensorShaderModel : (kernel.allowed_warp_size().has_value() ? kHighShaderModel : kShaderModel),
                option.enable_fast_math);
            info.handle = reinterpret_cast<uint64_t>(shader);
            info.native_handle = shader->pipeline();
        }
#endif
    }// end else (non-motion-blur path)
    return info;
}
ShaderCreationInfo Device::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept { return ShaderCreationInfo::make_invalid(); }
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const luisa::compute::Type *const> arg_types) noexcept {
    ShaderCreationInfo info;
    auto deser_result = ShaderSerializer::try_deser_compute(this, {}, {}, name, SerdeType::kByteCode, _binary_io);
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
    return shader->saved_arguments()[index].var_usage;
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

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &r) {
    bool destroy_inst = false;
    {
        std::lock_guard lck{detail::instance_mtx};
        if (!detail::vk_instance) {
            destroy_inst = true;
#ifdef NDEBUG
            constexpr bool enable_validation = false;
#else
            constexpr bool enable_validation = true;
#endif
            bool enable_surface{false};
            detail::create_instance(enable_validation, enable_surface, detail::vk_instance, {}, {}, {});
        }
    }
    vstd::vector<VkPhysicalDevice> physical_devices;
    uint32_t gpu_count = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(detail::vk_instance, &gpu_count, nullptr));
    if (gpu_count == 0) {
        return;
    }
    // Enumerate devices
    luisa::enlarge_by(physical_devices, gpu_count);
    auto err = vkEnumeratePhysicalDevices(detail::vk_instance, &gpu_count, physical_devices.data());
    if (err) {
        LUISA_ERROR("Could not enumerate physical devices : {}", (int)err);
        return;
    }
    r.reserve(physical_devices.size());
    VkPhysicalDeviceProperties device_properties;
    for (auto &&i : physical_devices) {
        vkGetPhysicalDeviceProperties(i, &device_properties);
        r.emplace_back(device_properties.deviceName);
    }
    if (destroy_inst) {
        vkDestroyInstance(detail::vk_instance, Device::alloc_callbacks());
        vks::VulkanDevice::force_free_volk();
    }
}

hlsl::ShaderCompiler *Device::compiler() {
    return g_dxc_compiler ? g_dxc_compiler.ptr() : nullptr;
}

VkInstance Device::instance() {
    return detail::vk_instance;
}
// HACK: for some app need external instance without device
LUISA_EXPORT_API VkInstance init_vk_instance(bool enable_validation, bool &enable_surface, const luisa::string *extra_instance_exts, size_t extra_instance_ext_count, const char *custom_vk_lib_path, const char *custom_vk_lib_name) {
    std::lock_guard lck{detail::instance_mtx};
    if (!detail::vk_instance) {
#ifdef NDEBUG
        constexpr bool enable_validation = false;
#else
        constexpr bool enable_validation = true;
#endif
        (void)enable_validation;
        detail::create_instance(enable_validation, enable_surface, detail::vk_instance, custom_vk_lib_path ? luisa::filesystem::path{custom_vk_lib_path} : luisa::filesystem::path{}, custom_vk_lib_name ? luisa::string_view{custom_vk_lib_name} : luisa::string_view{}, luisa::span{extra_instance_exts, extra_instance_ext_count});
    }
    return detail::vk_instance;
}

LUISA_EXPORT_API DeviceInterface *create(Context &&c, DeviceConfig const *settings) {
    return new Device(std::move(c), settings);
}

LUISA_EXPORT_API void destroy(DeviceInterface *device) {
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
Device::LazyLoadShader::LazyLoadShader(LoadFunc load_func) : _load_func(load_func) {}
Device::LazyLoadShader::~LazyLoadShader() = default;
ComputeShader *Device::LazyLoadShader::get(Device *self) {
    if (!_shader) {
        _shader = vstd::create_unique(_load_func(self));
    }
    return _shader.get();
}
bool Device::LazyLoadShader::check(Device *self) {
    if (_shader) return true;
    _shader = vstd::create_unique(_load_func(self));
    if (_shader) {
        auto afterExit = vstd::scope_exit([&] { _shader = nullptr; });
        return true;
    }
    return false;
}
ResourceCreationInfo Device::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    VkMemoryRequirements req{
        .size = byte_size,
        .alignment = kSparseBufferSize,
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
SparseBufferCreationInfo Device::create_sparse_buffer(const luisa::compute::Type *element, size_t elem_count) noexcept {
    if (element->is_custom()) [[unlikely]] {
        LUISA_ERROR("Indirect buffer not supported.");
    }
    SparseBufferCreationInfo info{};
    auto ptr = new SparseBuffer(this, element->size() * elem_count, true);
    info.element_stride = (element == Type::of<void>()) ? 1 : element->size();
    info.handle = reinterpret_cast<uint64_t>(ptr);
    info.native_handle = ptr->vk_buffer();
    info.total_size_bytes = ptr->byte_size();
    info.tile_size_bytes = kSparseBufferSize;
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
    r.tile_size_bytes = kSparseBufferSize;
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
    auto ite = _exts.find(name);
    if (ite == _exts.end()) return nullptr;
    auto &v = ite->second;
    {
        std::lock_guard lck{_ext_mtx};
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

#include "../common/export_version.inl.h"
