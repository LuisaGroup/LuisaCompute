#include "device.h"
#include "device_feature_plan.h"
#include "float_atomic_policy.h"
#include "sampler_anisotropy.h"
#include "user_compute_codegen_route.h"
#include "../common/env_flag.h"
#include <luisa/ast/op.h>
#include <luisa/core/clock.h>
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
#include "../common/backend_print_code.h"
#include "../common/spirv/spirv_codegen/target_feature_mask.h"
#include "builtin_kernel.h"
#include "shader_serializer.h"
#include "default_buffer.h"
#include "upload_buffer.h"
#include "indirect_buffer.h"
#include "stream.h"
#include "event.h"
#include "texture.h"
#include "resource_barrier_contract.h"
#include "queue_family_contract.h"
#include "bindless_array.h"
#include "blas.h"
#include "tlas.h"
#include "motion_instance.h"
#include "rt_shader.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include "sparse_heap.h"
#include "pinned_memory_ext.h"
#include "vk_raster_ext.h"
#include "vk_native_res_ext.h"
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/runtime/dispatch_buffer.h>
#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
#include "vk_cuda_interop_ext.h"
#endif
#if defined(LUISA_XIR_TO_SPIRV) && defined(LUISA_AST_LLVM_TO_SPIRV)
#error "Vulkan compute SPIR-V codegen paths are mutually exclusive."
#endif

#ifdef LUISA_XIR_TO_SPIRV
#include <spirv_codegen/entry.h>
#include <spirv_codegen/utils.h>
#include <SPIRV/disassemble.h>
#include <fstream>
#elif defined(LUISA_AST_LLVM_TO_SPIRV)
#include <spirv_llvm/spirv_llvm.h>
#include <SPIRV/disassemble.h>
#include <fstream>
#endif
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>

namespace lc::vk {
using namespace std::string_literals;
namespace {

[[nodiscard]] bool require_native_xir_spirv() noexcept {
    return lc::detail::env_flag(
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV");
}

[[nodiscard]] uint32_t query_bindless_heap_capacity(
    VkPhysicalDevice physical_device) noexcept {
    VkPhysicalDeviceVulkan12Features features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features};
    vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    if (features.descriptorIndexing != VK_TRUE ||
        features.shaderSampledImageArrayNonUniformIndexing != VK_TRUE ||
        features.shaderStorageBufferArrayNonUniformIndexing != VK_TRUE ||
        features.descriptorBindingSampledImageUpdateAfterBind != VK_TRUE ||
        features.descriptorBindingStorageBufferUpdateAfterBind != VK_TRUE ||
        features.descriptorBindingPartiallyBound != VK_TRUE ||
        features.runtimeDescriptorArray != VK_TRUE) {
        return 0u;
    }
    VkPhysicalDeviceMaintenance3Properties maintenance3_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceDescriptorIndexingProperties descriptor_indexing_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
        .pNext = &maintenance3_properties};
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &descriptor_indexing_properties};
    vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    return detail::plan_bindless_heap_capacity(
        {.max_per_set_descriptors =
             maintenance3_properties.maxPerSetDescriptors,
         .max_per_stage_update_after_bind_samplers =
             descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindSamplers,
         .max_descriptor_set_update_after_bind_samplers =
             descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSamplers,
         .max_per_stage_update_after_bind_storage_buffers =
             descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindStorageBuffers,
         .max_descriptor_set_update_after_bind_storage_buffers =
             descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindStorageBuffers,
         .max_per_stage_update_after_bind_sampled_images =
             descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindSampledImages,
         .max_descriptor_set_update_after_bind_sampled_images =
             descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSampledImages,
         .max_per_stage_update_after_bind_resources =
             descriptor_indexing_properties.maxPerStageUpdateAfterBindResources,
         .max_update_after_bind_descriptors_in_all_pools =
             descriptor_indexing_properties.maxUpdateAfterBindDescriptorsInAllPools});
}

#ifdef LUISA_XIR_TO_SPIRV
[[nodiscard]] luisa::string describe_hlsl_fallback_reasons(
    detail::UserComputeCodegenRoute route) noexcept {
    constexpr std::array reasons{
        detail::UserComputeHlslFallbackReason::NATIVE_INCLUDE,
        detail::UserComputeHlslFallbackReason::PRINTING,
        detail::UserComputeHlslFallbackReason::COOPERATIVE_OPERATIONS,
        detail::UserComputeHlslFallbackReason::ASYNC_COPY,
        detail::UserComputeHlslFallbackReason::MOTION_BLUR};
    luisa::string description;
    for (auto reason : reasons) {
        if (!route.contains(reason)) { continue; }
        if (!description.empty()) { description.append(", "); }
        description.append(
            detail::user_compute_hlsl_fallback_reason_name(reason));
    }
    return description;
}
#endif

[[nodiscard]] bool validate_sampler_anisotropy_requirement(
    Function function, luisa::string_view native_include,
    bool anisotropy_enabled) noexcept {
    auto usage = detail::analyze_sampler_usage(function);
    if (usage.has_invalid_filter) [[unlikely]] {
        LUISA_ERROR(
            "Vulkan shader '{}' contains an explicit texture sampler filter "
            "outside the valid [0, 4) selector range.",
            function.name());
    }
    // Native source can reference the fixed sampler heap directly. Without a
    // typed source-level contract, conservatively record that it may select
    // an anisotropic entry.
    auto unrestricted_native_sampler_access = !native_include.empty();
    auto requires_anisotropy =
        usage.requires_anisotropy || unrestricted_native_sampler_access;
    if (!detail::sampler_requirement_is_supported(
            requires_anisotropy, anisotropy_enabled)) [[unlikely]] {
        if (unrestricted_native_sampler_access) {
            LUISA_ERROR(
                "Vulkan shader '{}' includes unrestricted native HLSL, which "
                "may access anisotropic sampler-heap entries, but "
                "samplerAnisotropy is not enabled on this logical device.",
                function.name());
        }
        if (usage.has_dynamic_filter) {
            LUISA_ERROR(
                "Vulkan shader '{}' dynamically selects a texture sampler "
                "filter, which may select ANISOTROPIC, but samplerAnisotropy "
                "is not enabled on this logical device.",
                function.name());
        }
        LUISA_ERROR(
            "Vulkan shader '{}' selects ANISOTROPIC texture filtering, but "
            "samplerAnisotropy is not enabled on this logical device.",
            function.name());
    }
    return requires_anisotropy;
}

[[nodiscard]] SpirvArtifactFeatureRequirements
validated_spirv_artifact_requirements(
    const Device *device,
    lc::spirv::SpirvTargetFeatureMask required) noexcept {
    auto check = lc::spirv::check_spirv_target_feature_requirements(
        required, device->enabled_spirv_artifact_features());
    LUISA_ASSERT(
        static_cast<bool>(check),
        "Vulkan attempted to create or persist a SPIR-V artifact with "
        "unknown requirements 0x{:016x} or unavailable requirements "
        "0x{:016x}.",
        check.unknown_required_bits, check.missing_required_bits);
    return SpirvArtifactFeatureRequirements{required};
}

[[nodiscard]] SpirvArtifactFeatureRequirements
conservative_spirv_artifact_requirements(
    const Device *device, bool requires_sampler_anisotropy) noexcept {
    auto required = device->enabled_spirv_artifact_features();
    if (requires_sampler_anisotropy) {
        required |= lc::spirv::target_feature::sampler_anisotropy;
    }
    return validated_spirv_artifact_requirements(device, required);
}

}// namespace
#ifndef LC_NO_HLSL_BUILTIN
static luisa::spin_mutex g_dxc_mutex;
static vstd::StackObject<hlsl::ShaderCompiler, false> g_dxc_compiler;
static luisa::filesystem::path g_dxc_runtime_directory;
static int32 g_dxc_ref_count = 0;
static bool g_dxc_compiler_initialized = false;
#endif

namespace detail {

[[nodiscard]] bool validation_enabled_by_default() noexcept {
#ifdef NDEBUG
    auto enabled = false;
#else
    auto enabled = true;
#endif
    if (lc::detail::env_flag("LUISA_VULKAN_VALIDATION")) { enabled = true; }
    return enabled;
}

struct Settings {
    bool validation{false};
    bool fullscreen{false};
    bool vsync{false};
    bool overlay{true};
};

static VkInstance vk_instance{nullptr};
static std::mutex instance_mtx;
static std::mutex dispatch_lifetime_mtx;
static uint32_t live_device_count{};
static bool vk_instance_surface_enabled{};
static bool vk_instance_validation_enabled{};
static vstd::unordered_set<luisa::string> vk_instance_extra_exts;
static Settings settings{};
static PFN_vkCreateDebugUtilsMessengerEXT vk_create_debug_utils_messenger_ext;
static PFN_vkDestroyDebugUtilsMessengerEXT vk_destroy_debug_utils_messenger_ext;
static VkDebugUtilsMessengerEXT debug_utils_messenger;
static VkInstance debug_utils_messenger_instance;
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
    if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
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

    if (debug_utils_messenger != VK_NULL_HANDLE) {
        LUISA_ASSERT(
            debug_utils_messenger_instance == instance,
            "Vulkan validation messenger is already attached to a different "
            "process instance.");
        return;
    }

    vk_create_debug_utils_messenger_ext = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    vk_destroy_debug_utils_messenger_ext = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    LUISA_ASSERT(
        vk_create_debug_utils_messenger_ext != nullptr &&
            vk_destroy_debug_utils_messenger_ext != nullptr,
        "VK_EXT_debug_utils was enabled, but its messenger entry points are unavailable.");

    VkDebugUtilsMessengerCreateInfoEXT debug_utils_messenger_ci{};
    debug_utils_messenger_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_utils_messenger_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_utils_messenger_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debug_utils_messenger_ci.pfnUserCallback = debug_utils_messenger_callback;
    VK_CHECK_RESULT(vk_create_debug_utils_messenger_ext(
        instance, &debug_utils_messenger_ci,
        Device::alloc_callbacks(), &debug_utils_messenger));
    debug_utils_messenger_instance = instance;
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

#if defined(LUISA_PLATFORM_APPLE) && defined(VK_KHR_portability_enumeration)
        // MoltenVK requires portability enumeration to expose all conformant physical devices.
        if (supported_instance_exts.find(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != supported_instance_exts.end()) {
            emplace_instance_ext(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instance_create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif
        if (settings.validation) {
            emplace_instance_ext(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);// SRS - Dependency when VK_EXT_DEBUG_MARKER is enabled
            if (!emplace_instance_ext(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                LUISA_WARNING(
                    "VK_EXT_debug_utils is unavailable; Vulkan validation "
                    "messenger output is disabled.");
                settings.validation = false;
            }
        }
        instance_create_info.enabledExtensionCount = (uint32_t)instance_exts.size();
        instance_create_info.ppEnabledExtensionNames = instance_exts.size() > 0 ? instance_exts.data() : nullptr;
        VK_CHECK_RESULT(vkCreateInstance(&instance_create_info, Device::alloc_callbacks(), &instance));
    }
    volkLoadInstance(instance);
}

void load_or_create_process_instance(
    bool enable_validation, bool &enable_surface,
    luisa::filesystem::path const &custom_path,
    luisa::string_view lib_name,
    luisa::span<luisa::string const> extra_exts) {
    auto creating = vk_instance == VK_NULL_HANDLE;
    if (!creating) {
        enable_surface &= vk_instance_surface_enabled;
        settings.validation = vk_instance_validation_enabled;
        for (auto &&extension : extra_exts) {
            LUISA_ASSERT(
                vk_instance_extra_exts.contains(extension),
                "The process Vulkan instance is already live without "
                "requested extension '{}'. Instance extensions cannot be "
                "added after creation.",
                extension);
        }
    }
    create_instance(
        enable_validation, enable_surface, vk_instance,
        custom_path, lib_name, extra_exts);
    if (creating) {
        vk_instance_surface_enabled = enable_surface;
        vk_instance_validation_enabled = settings.validation;
        vk_instance_extra_exts.clear();
        for (auto &&extension : extra_exts) {
            vk_instance_extra_exts.emplace(extension);
        }
    }
}

}// namespace detail

Device::GlobalDispatchLease::GlobalDispatchLease() noexcept {
    std::lock_guard lock{detail::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::live_device_count == 0u,
        "The Vulkan backend currently supports only one live Device per "
        "process because Volk uses process-global instance/device dispatch "
        "tables. Destroy the existing Vulkan Device before creating another.");
    ++detail::live_device_count;
}

Device::GlobalDispatchLease::~GlobalDispatchLease() noexcept {
    std::lock_guard lock{detail::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::live_device_count == 1u,
        "Vulkan global dispatch lifetime accounting is unbalanced.");
    --detail::live_device_count;
}

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
    VkPhysicalDeviceSubgroupProperties subgroup_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_properties};
    vkGetPhysicalDeviceProperties2(physical_device(), &properties2);
    return subgroup_properties.subgroupSize;
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
      set_accel_kernel(BuiltinKernel::load_accel_set_kernel),
      prepare_indirect_kernel(BuiltinKernel::load_indirect_prepare_kernel) {
    bool headless = false;
    bool use_lmdb = false;
    auto require_native_spirv = require_native_xir_spirv();
#ifdef LC_NO_HLSL_BUILTIN
    constexpr auto dxc_compatibility_compiled = false;
#else
    constexpr auto dxc_compatibility_compiled = true;
#endif
    bool load_dxc_for_config_readback = false;
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
    luisa::string lib_name;

    if (_config_ext) {
        auto external_device = _config_ext->create_external_device();
        auto has_queue_metadata =
            external_device.graphics_queue != VK_NULL_HANDLE ||
            external_device.compute_queue != VK_NULL_HANDLE ||
            external_device.copy_queue != VK_NULL_HANDLE ||
            external_device.graphics_queue_family_index !=
                VK_QUEUE_FAMILY_IGNORED ||
            external_device.compute_queue_family_index !=
                VK_QUEUE_FAMILY_IGNORED ||
            external_device.copy_queue_family_index !=
                VK_QUEUE_FAMILY_IGNORED;
        auto ancestry = detail::validate_external_device_ancestry(
            external_device.instance != VK_NULL_HANDLE,
            external_device.physical_device != VK_NULL_HANDLE,
            external_device.device != VK_NULL_HANDLE,
            has_queue_metadata);
        LUISA_ASSERT(
            static_cast<bool>(ancestry),
            "Invalid imported Vulkan device ancestry: {}.",
            detail::external_device_ancestry_status_name(ancestry.status));
        auto instance_api = detail::validate_external_instance_api(
            external_device.instance != VK_NULL_HANDLE,
            external_device.api_version);
        LUISA_ASSERT(
            static_cast<bool>(instance_api),
            "Invalid imported Vulkan instance contract: {} (reported API "
            "version {}.{}.{}).",
            detail::external_instance_api_status_name(instance_api.status),
            VK_API_VERSION_MAJOR(external_device.api_version),
            VK_API_VERSION_MINOR(external_device.api_version),
            VK_API_VERSION_PATCH(external_device.api_version));
        auto queue_handles = detail::validate_external_queue_handles(
            external_device.device != VK_NULL_HANDLE,
            {external_device.graphics_queue != VK_NULL_HANDLE,
             external_device.compute_queue != VK_NULL_HANDLE,
             external_device.copy_queue != VK_NULL_HANDLE});
        LUISA_ASSERT(
            static_cast<bool>(queue_handles),
            "Imported Vulkan device has an invalid {} queue contract: {}. "
            "Supply the actual queue handle; the backend cannot infer which "
            "queues were created on an existing logical device.",
            detail::queue_family_role_name(queue_handles.role),
            detail::external_queue_handle_status_name(queue_handles.status));
        auto required_features =
            detail::validate_external_required_features(
                external_device.device != VK_NULL_HANDLE,
                external_device.required_features.timeline_semaphore,
                external_device.required_features.synchronization2);
        LUISA_ASSERT(
            static_cast<bool>(required_features),
            "Imported Vulkan device is missing a mandatory enabled-feature "
            "contract: {}.",
            detail::external_required_feature_status_name(
                required_features.status));
        ext_phy_device = external_device.physical_device;
        ext_device = external_device.device;
        _instance = external_device.instance;
        // A config extension may request DXC for legacy integrations, but a
        // strict native route and a build that omitted DXC are both
        // authoritative.
        load_dxc_for_config_readback =
            _config_ext->load_dxc() && dxc_compatibility_compiled &&
            !require_native_spirv;
        _graphics_queue = external_device.graphics_queue;
        auto ext_path = _config_ext->external_vulkan_lib_path();
        custom_path = std::move(ext_path.lib_path);
        lib_name = std::move(ext_path.lib_name);
        _compute_queue = external_device.compute_queue;
        _copy_queue = external_device.copy_queue;
        external_instance = external_device.instance;
        this->external_device = external_device.device;
        external_graphics_queue_family_index =
            external_device.graphics_queue_family_index;
        external_compute_queue_family_index =
            external_device.compute_queue_family_index;
        external_copy_queue_family_index =
            external_device.copy_queue_family_index;
        bindless_enabled = _config_ext->enable_bindless_feature();
        raytracing_enabled = _config_ext->enable_raytracing_feature();
        interop_enabled = _config_ext->enable_interop_feature();
        device_address_enabled = _config_ext->enable_device_address_feature();
        surface_enabled = _config_ext->enable_surface_feature();
        if (external_instance) {
            // The enabled instance-extension list is not queryable. A
            // borrowed instance therefore remains compute-only until the
            // import API grows an explicit surface-extension attestation.
            surface_enabled = false;
        }
        if (ext_device != VK_NULL_HANDLE) {
            // Optional logical-device features and extensions cannot be
            // queried after VkDevice creation. Until the import API carries
            // explicit attestations for them, keep every optional backend
            // path fail-closed.
            bindless_enabled = false;
            raytracing_enabled = false;
            interop_enabled = false;
            device_address_enabled = false;
            surface_enabled = false;
        }
    }
    device_address_enabled |= raytracing_enabled;

#ifndef LC_NO_HLSL_BUILTIN
    Context ctx{this->_ctx_impl};
    {
        std::lock_guard lck(g_dxc_mutex);
        if (g_dxc_ref_count == 0) {
            g_dxc_runtime_directory = ctx.runtime_directory();
        }
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
    if (external_instance) {
        // Initialize dispatch entry points against the borrowed instance, but
        // never install caller-owned lifetime into the backend's global owned
        // instance slot.
        auto capabilities = detail::plan_instance_runtime_capabilities(
            true, surface_enabled,
            detail::validation_enabled_by_default());
        detail::settings.validation = capabilities.debug_utils;
        auto enable_validation = capabilities.debug_utils;
        auto enable_surface = capabilities.surface;
        detail::create_instance(
            enable_validation, enable_surface, _instance,
            custom_path, lib_name, {});
        surface_enabled = enable_surface;
    } else {
        std::lock_guard lck{detail::instance_mtx};
        auto enable_validation = detail::validation_enabled_by_default();
        luisa::vector<luisa::string> extra_exts = [&]() {
            if (_config_ext) {
                return _config_ext->extra_instance_exts();
            } else {
                return luisa::vector<luisa::string>{};
            }
        }();
        bool enable_surface = surface_enabled;
        // Reload Volk's global instance table even when reusing the
        // process-owned instance: the previous Device may have borrowed a
        // different instance and replaced those entry points.
        detail::load_or_create_process_instance(
            enable_validation, enable_surface,
            custom_path, lib_name, extra_exts);
        surface_enabled = enable_surface;
        _instance = detail::vk_instance;
    }
#ifndef LUISA_VULKAN_ENABLE_CUDA_INTEROP
    interop_enabled = false;
#endif
    _init_device(ext_phy_device, ext_device, device_idx);

    {
        constexpr auto dummy_word_count =
            IndirectDispatchLayout::header_word_count +
            IndirectDispatchLayout::record_word_count;
        std::array<uint32_t, dummy_word_count> zero_words{};
        _indirect_dispatch_dummy = luisa::make_unique<UploadBuffer>(
            this, sizeof(zero_words));
        _indirect_dispatch_dummy->copy_from(
            zero_words.data(), 0u, sizeof(zero_words));
        // This host write precedes every possible queue submission. Flushing
        // here makes the persistent descriptor valid on non-coherent heaps;
        // it never changes again and needs no per-command resource tracking.
        static_cast<void>(_indirect_dispatch_dummy->flush_host());
    }

    if (_config_ext) {
        _config_ext->init_volk(vkGetInstanceProcAddr);
        hlsl::ShaderCompiler *dxc = nullptr;
        if (load_dxc_for_config_readback) {
            dxc = Device::compiler();
        }
        _config_ext->readback_vulkan_device(
            instance(), physical_device(), logic_device(), alloc_callbacks(),
            _pso_header, _graphics_queue, _compute_queue, _copy_queue,
            graphics_queue_index(), compute_queue_index(), copy_queue_index(),
            dxc == nullptr ? nullptr : dxc->compiler(),
            dxc == nullptr ? nullptr : dxc->library(),
            dxc == nullptr ? nullptr : dxc->utils());
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
            detail::setup_debugging(instance());
        }

        // Physical device
        uint32_t gpu_count = 0;
        // Get number of available physical devices
        VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance(), &gpu_count, nullptr));
        if (gpu_count == 0) {
            LUISA_ERROR("No device with Vulkan support found");
            return;
        }
        vstd::vector<VkPhysicalDevice> physical_devices;
        // Enumerate devices
        luisa::enlarge_by(physical_devices, gpu_count);
        err = vkEnumeratePhysicalDevices(instance(), &gpu_count, physical_devices.data());
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
#if defined(LUISA_PLATFORM_APPLE)
            selected_device = 0;
            detail::DefaultDeviceCandidate selected_candidate{};
            for (uint32_t i = 0u; i < gpu_count; i++) {
                uint32_t queue_family_count = 0u;
                vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_count, nullptr);
                vstd::vector<VkQueueFamilyProperties> queue_families;
                luisa::enlarge_by(queue_families, queue_family_count);
                vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &queue_family_count, queue_families.data());
                auto supports_graphics_compute = std::any_of(
                    queue_families.begin(), queue_families.end(),
                    [](auto &&queue_family) noexcept {
                        constexpr auto required = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
                        return queue_family.queueCount > 0u &&
                               (queue_family.queueFlags & required) == required;
                    });
                auto candidate = detail::DefaultDeviceCandidate{
                    .supports_graphics_compute = supports_graphics_compute,
                    .bindless_heap_capacity =
                        bindless_enabled ?
                            query_bindless_heap_capacity(physical_devices[i]) :
                            0u};
                if (detail::prefer_default_device_candidate(
                        candidate, selected_candidate, bindless_enabled)) {
                    selected_device = i;
                    selected_candidate = candidate;
                }
            }
            vkGetPhysicalDeviceProperties(
                physical_devices[selected_device], &device_properties);
            LUISA_INFO(
                "Select device: {} (device ID: {:#010x}, bindless capacity: {})",
                device_properties.deviceName, device_properties.deviceID,
                selected_candidate.bindless_heap_capacity);
#else
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
#endif
        }
        physical_device = physical_devices[std::min<uint32_t>(selected_device, physical_devices.size() - 1)];
    }

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    auto supported_ext = detail::supported_exts(physical_device);
    VkPhysicalDeviceFeatures device_features{};
    vkGetPhysicalDeviceFeatures(physical_device, &device_features);
    auto storage_image_format_features =
        detail::plan_storage_image_format_features({
            .read_without_format =
                device_features.shaderStorageImageReadWithoutFormat ==
                VK_TRUE,
            .write_without_format =
                device_features.shaderStorageImageWriteWithoutFormat ==
                VK_TRUE,
            .imported_device = external_device != VK_NULL_HANDLE});
    device_features.shaderStorageImageReadWithoutFormat =
        storage_image_format_features.read_without_format ?
            VK_TRUE :
            VK_FALSE;
    device_features.shaderStorageImageWriteWithoutFormat =
        storage_image_format_features.write_without_format ?
            VK_TRUE :
            VK_FALSE;
    // Derived examples can override this to set actual features (based on above readings) to enable for logical device creation

    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    _vk_device.create(physical_device);
    _vk_device->logical_device = external_device;
    auto external_queue_contract =
        detail::validate_external_queue_family_contract(
            external_device != VK_NULL_HANDLE,
            std::array{
                external_graphics_queue_family_index,
                external_compute_queue_family_index,
                external_copy_queue_family_index},
            luisa::span<const VkQueueFamilyProperties>{
                _vk_device->queue_family_properties.data(),
                _vk_device->queue_family_properties.size()});
    LUISA_ASSERT(
        static_cast<bool>(external_queue_contract),
        "Imported Vulkan device has an invalid {} queue-family contract: {} "
        "(index {}, required flags 0x{:x}, available flags 0x{:x}).",
        detail::queue_family_role_name(external_queue_contract.role),
        detail::queue_family_contract_status_name(
            external_queue_contract.status),
        external_queue_contract.family_index,
        external_queue_contract.required_flags,
        external_queue_contract.available_flags);
    auto sampler_anisotropy_plan = detail::plan_sampler_anisotropy({.physical_device_feature =
                                                                        device_features.samplerAnisotropy == VK_TRUE,
                                                                    .imported_device = external_device != VK_NULL_HANDLE,
                                                                    .max_sampler_anisotropy =
                                                                        _vk_device->properties.limits.maxSamplerAnisotropy});
    sampler_anisotropy_enabled = sampler_anisotropy_plan.enabled;
    _max_sampler_anisotropy = sampler_anisotropy_plan.max_anisotropy;
    // Keep VulkanDevice::enabled_features honest for both owned and imported
    // logical devices. For imports the enabled feature chain is unknowable,
    // so physical-device support must not be advertised as enabled support.
    device_features.samplerAnisotropy =
        sampler_anisotropy_enabled ? VK_TRUE : VK_FALSE;
    if (external_device != VK_NULL_HANDLE && bindless_enabled) {
        // Vulkan exposes physical support but cannot report which descriptor
        // indexing features were enabled when an imported logical device was
        // created. Without an explicit import-side feature contract, creating
        // update-after-bind layouts would be unsound.
        LUISA_INFO(
            "Vulkan bindless descriptors disabled for imported logical device "
            "because enabled descriptor-indexing features are unknown.");
        bindless_enabled = false;
        _bindless_heap_capacity = 0u;
    }
    auto supports_device_extension = [&supported_ext](char const *name) noexcept {
        return supported_ext.find(name) != supported_ext.end();
    };
    auto enable_device_extension = [this](char const *name) noexcept {
        if (std::find(_enable_device_exts.begin(),
                      _enable_device_exts.end(), name) ==
            _enable_device_exts.end()) {
            _enable_device_exts.emplace_back(name);
        }
    };
    auto api_version = _vk_device->properties.apiVersion;
    auto has_core_timeline_semaphore = api_version >= VK_API_VERSION_1_2;
    auto has_core_synchronization2 = api_version >= VK_API_VERSION_1_3;
    auto has_physical_device_api_1_3 =
        api_version >= VK_API_VERSION_1_3;
    VkPhysicalDeviceSynchronization2Features supported_synchronization2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .pNext = nullptr};
    if (has_core_synchronization2 ||
        supports_device_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_synchronization2};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    }
    auto required_device_features = detail::plan_required_device_features({.timeline_semaphore_core = has_core_timeline_semaphore,
                                                                           .timeline_semaphore_extension =
                                                                               supports_device_extension(
                                                                                   VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME),
                                                                           .timeline_semaphore_feature =
                                                                               _vk_device->features_12.timelineSemaphore == VK_TRUE,
                                                                           .synchronization2_core = has_core_synchronization2,
                                                                           .synchronization2_extension =
                                                                               supports_device_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),
                                                                           .synchronization2_feature =
                                                                               supported_synchronization2.synchronization2 == VK_TRUE,
                                                                           .physical_device_api_1_3 = has_physical_device_api_1_3});
    if (!required_device_features.supported) [[unlikely]] {
        LUISA_ERROR(
            "The Vulkan backend requires a Vulkan 1.3 physical device "
            "because it uses the core copy_commands2 entry points, as well "
            "as timelineSemaphore and synchronization2; "
            "physical device API {}.{}.{} reports timelineSemaphore={} "
            "(core={}, extension={}) and synchronization2={} "
            "(core={}, extension={}) and core copy_commands2={}.",
            VK_API_VERSION_MAJOR(api_version),
            VK_API_VERSION_MINOR(api_version),
            VK_API_VERSION_PATCH(api_version),
            _vk_device->features_12.timelineSemaphore == VK_TRUE,
            has_core_timeline_semaphore,
            supports_device_extension(
                VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME),
            supported_synchronization2.synchronization2 == VK_TRUE,
            has_core_synchronization2,
            supports_device_extension(
                VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME),
            has_physical_device_api_1_3);
    }
    {
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &_timeline_semaphore_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
        LUISA_ASSERT(
            _timeline_semaphore_properties
                    .maxTimelineSemaphoreValueDifference != 0u,
            "Vulkan 1.3 timeline-semaphore support reported a zero "
            "maxTimelineSemaphoreValueDifference.");
    }
    if (required_device_features.enable_timeline_semaphore_extension) {
        enable_device_extension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }
    if (required_device_features.enable_synchronization2_extension) {
        enable_device_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    detail::NarrowNumericFeaturePlan narrow_numeric_features{};
    bool enable_float8 = false;
    bool enable_buffer_int64_atomics = false;
    bool enable_shared_int64_atomics = false;
    bool enable_barycentric = false;
    bool enable_motion_blur = false;
    bool enable_buffer_float32_atomics = false;
    bool enable_buffer_float32_atomic_add = false;
    bool enable_buffer_float32_atomic_min_max = false;
    bool enable_shared_float32_atomics = false;
    bool enable_shared_float32_atomic_add = false;
    bool enable_shared_float32_atomic_min_max = false;
    bool enable_subgroup_size_control = false;
    bool enable_subgroup_extended_types = false;
    bool enable_shader_device_clock = false;
    {
        VkPhysicalDeviceVulkan12Features vk12_atomic_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vk12_atomic_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        enable_buffer_int64_atomics =
            vk12_atomic_features.shaderBufferInt64Atomics == VK_TRUE;
        enable_shared_int64_atomics =
            vk12_atomic_features.shaderSharedInt64Atomics == VK_TRUE;
        if (enable_buffer_int64_atomics || enable_shared_int64_atomics) {
            if (supported_ext.find(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) !=
                supported_ext.end()) {
                enable_device_extension(
                    VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
            }
        }
    }
    {
        VkPhysicalDeviceVulkan11Features vk11_narrow_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = nullptr};
        VkPhysicalDeviceVulkan12Features vk12_narrow_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &vk11_narrow_features};
        VkPhysicalDeviceFeatures2 features2_narrow{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vk12_narrow_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2_narrow);
        narrow_numeric_features = detail::plan_narrow_numeric_features({.shader_float16 =
                                                                            vk12_narrow_features.shaderFloat16 == VK_TRUE,
                                                                        .shader_int8 = vk12_narrow_features.shaderInt8 == VK_TRUE,
                                                                        .storage_buffer_8bit_access =
                                                                            vk12_narrow_features.storageBuffer8BitAccess == VK_TRUE,
                                                                        .uniform_storage_buffer_8bit_access =
                                                                            vk12_narrow_features.uniformAndStorageBuffer8BitAccess ==
                                                                            VK_TRUE,
                                                                        .storage_buffer_16bit_access =
                                                                            vk11_narrow_features.storageBuffer16BitAccess == VK_TRUE,
                                                                        .uniform_storage_buffer_16bit_access =
                                                                            vk11_narrow_features.uniformAndStorageBuffer16BitAccess ==
                                                                            VK_TRUE});
    }
    if (supported_ext.find(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) !=
        supported_ext.end()) {
        VkPhysicalDeviceShaderFloat8FeaturesEXT supported_float8{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_float8};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (supported_float8.shaderFloat8 == VK_TRUE) {
            enable_device_extension(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
            enable_float8 = true;
        }
    }
#ifndef NDEBUG
    if (supported_ext.find(VK_KHR_SHADER_CLOCK_EXTENSION_NAME) !=
        supported_ext.end()) {
        VkPhysicalDeviceShaderClockFeaturesKHR supported_shader_clock{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_shader_clock};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (supported_shader_clock.shaderDeviceClock == VK_TRUE) {
            enable_device_extension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
            enable_shader_device_clock = true;
        }
    }
#endif
    {
        VkPhysicalDeviceVulkan12Features vk12_subgroup_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vk12_subgroup_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (vk12_subgroup_features.shaderSubgroupExtendedTypes == VK_TRUE) {
            enable_subgroup_extended_types = true;
        }
    }
    if (supported_ext.find(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_float_atomics{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_float_atomics};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        enable_buffer_float32_atomics =
            supported_float_atomics.shaderBufferFloat32Atomics == VK_TRUE;
        enable_buffer_float32_atomic_add =
            supported_float_atomics.shaderBufferFloat32AtomicAdd == VK_TRUE;
        enable_shared_float32_atomics =
            supported_float_atomics.shaderSharedFloat32Atomics == VK_TRUE;
        enable_shared_float32_atomic_add =
            supported_float_atomics.shaderSharedFloat32AtomicAdd == VK_TRUE;
    }
    if (supported_ext.find(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) != supported_ext.end() &&
        supported_ext.find(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT supported_float_atomics2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_float_atomics2};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        enable_buffer_float32_atomic_min_max =
            supported_float_atomics2.shaderBufferFloat32AtomicMinMax == VK_TRUE;
        enable_shared_float32_atomic_min_max =
            supported_float_atomics2.shaderSharedFloat32AtomicMinMax == VK_TRUE;
    }
    auto enable_float_atomic_ext =
        enable_buffer_float32_atomics ||
        enable_buffer_float32_atomic_add ||
        enable_shared_float32_atomics ||
        enable_shared_float32_atomic_add ||
        enable_buffer_float32_atomic_min_max ||
        enable_shared_float32_atomic_min_max;
    if (enable_float_atomic_ext) {
        enable_device_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    if (enable_buffer_float32_atomic_min_max ||
        enable_shared_float32_atomic_min_max) {
        enable_device_extension(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    }
    // Try enabling cooperative-matrix extensions in order of preference:
    // VK_KHR_cooperative_matrix -> VK_NV_cooperative_matrix2 (requires KHR) -> VK_NV_cooperative_matrix.
    enum class CooperativeMatrixExt : uint8_t { None,
                                                KHR,
                                                KHRAndNV2,
                                                NV };
    auto enabled_cooperative_matrix_ext = CooperativeMatrixExt::None;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features_khr{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = nullptr,
        .cooperativeMatrix = VK_FALSE};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV cooperative_matrix_2_features_nv{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
        .pNext = nullptr,
        .cooperativeMatrixWorkgroupScope = VK_FALSE};
    VkPhysicalDeviceCooperativeMatrixFeaturesNV cooperative_matrix_features_nv{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
        .pNext = nullptr,
        .cooperativeMatrix = VK_FALSE};
    if (supported_ext.find(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &cooperative_matrix_features_khr};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperative_matrix_properties_khr{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
            .pNext = nullptr};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &cooperative_matrix_properties_khr};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
        if (cooperative_matrix_features_khr.cooperativeMatrix == VK_TRUE &&
            (cooperative_matrix_properties_khr.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0u) {
            enabled_cooperative_matrix_ext = CooperativeMatrixExt::KHR;
            enable_device_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }
    }
    if (enabled_cooperative_matrix_ext == CooperativeMatrixExt::KHR &&
        supported_ext.find(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &cooperative_matrix_2_features_nv};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (cooperative_matrix_2_features_nv.cooperativeMatrixWorkgroupScope == VK_TRUE) {
            enabled_cooperative_matrix_ext = CooperativeMatrixExt::KHRAndNV2;
            enable_device_extension(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
        }
    }
    if (enabled_cooperative_matrix_ext == CooperativeMatrixExt::None &&
        supported_ext.find(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &cooperative_matrix_features_nv};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        VkPhysicalDeviceCooperativeMatrixPropertiesNV cooperative_matrix_properties_nv{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV,
            .pNext = nullptr};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &cooperative_matrix_properties_nv};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
        if (cooperative_matrix_features_nv.cooperativeMatrix == VK_TRUE &&
            (cooperative_matrix_properties_nv.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0u) {
            enabled_cooperative_matrix_ext = CooperativeMatrixExt::NV;
            enable_device_extension(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }
    }
    // Probe for cooperative-vector support (VK_NV_cooperative_vector).
    VkPhysicalDeviceCooperativeVectorFeaturesNV cooperative_vector_features_nv{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
        .pNext = nullptr,
        .cooperativeVector = VK_FALSE};
    if (supported_ext.find(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &cooperative_vector_features_nv};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        VkPhysicalDeviceCooperativeVectorPropertiesNV cooperative_vector_properties_nv{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV,
            .pNext = nullptr};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &cooperative_vector_properties_nv};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
        if (cooperative_vector_features_nv.cooperativeVector == VK_TRUE &&
            (cooperative_vector_properties_nv.cooperativeVectorSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0u) {
            // Query the list of supported cooperative-vector configurations.
            uint32_t prop_count = 0u;
            VkResult cv_result = vkGetPhysicalDeviceCooperativeVectorPropertiesNV(
                physical_device, &prop_count, nullptr);
            if (cv_result == VK_SUCCESS && prop_count > 0u) {
                // Allocate and zero-initialize the properties array dynamically
                // so we can handle any number of configs the driver returns.
                vstd::vector<VkCooperativeVectorPropertiesNV> cv_props(prop_count);
                uint32_t query_count = prop_count;
                for (uint32_t i = 0; i < query_count; ++i) {
                    cv_props[i] = VkCooperativeVectorPropertiesNV{};
                    cv_props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;
                }
                cv_result = vkGetPhysicalDeviceCooperativeVectorPropertiesNV(
                    physical_device, &query_count, cv_props.data());
                if (cv_result == VK_INCOMPLETE || cv_result == VK_SUCCESS) {
                    // query_count is now the actual number written by the driver.
                    // Iterate only over valid entries and skip any with obviously
                    // invalid enum values (driver corruption guard).
                    // The VkComponentTypeKHR enum includes packed/quantized types
                    // with large values (e.g. 0x10001450001, 1000491000). Accept
                    // any value that is not an obviously corrupted pointer.
                    constexpr uint32_t max_reasonable = 0x20000000u;
                    auto actual_count = query_count;
                    for (uint32_t i = 0; i < actual_count; ++i) {
                        auto const &p = cv_props[i];
                        if (static_cast<uint32_t>(p.inputType) >= max_reasonable ||
                            static_cast<uint32_t>(p.inputInterpretation) >= max_reasonable ||
                            static_cast<uint32_t>(p.matrixInterpretation) >= max_reasonable ||
                            static_cast<uint32_t>(p.biasInterpretation) >= max_reasonable ||
                            static_cast<uint32_t>(p.resultType) >= max_reasonable) {
                            LUISA_VERBOSE(
                                "  CooperativeVector config[{}]: SKIPPED (corrupted entry)", i);
                            continue;
                        }
                        LUISA_INFO("  CooperativeVector config[{}]: inputType={}, inputInterp={}, matrixInterp={}, biasInterp={}, resultType={}, transpose={}",
                                   i,
                                   static_cast<int>(p.inputType),
                                   static_cast<int>(p.inputInterpretation),
                                   static_cast<int>(p.matrixInterpretation),
                                   static_cast<int>(p.biasInterpretation),
                                   static_cast<int>(p.resultType),
                                   static_cast<int>(p.transpose));
                    }
                    // Check if the required FP32 all-float configuration is supported.
                    bool has_fp32_float_config = false;
                    for (uint32_t i = 0; i < actual_count; ++i) {
                        auto const &p = cv_props[i];
                        if (static_cast<uint32_t>(p.inputType) >= max_reasonable) { continue; }
                        if (p.inputType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            p.inputInterpretation == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            p.matrixInterpretation == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            p.resultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                            p.transpose == VK_FALSE) {
                            has_fp32_float_config = true;
                            break;
                        }
                    }
                    if (has_fp32_float_config) {
                        // Pure-FP32 cooperative-vector path available.
                        cooperative_vector_fp32_enabled = true;
                        LUISA_INFO("VK_NV_cooperative_vector: FP32 float config supported.");
                    } else {
                        LUISA_INFO("VK_NV_cooperative_vector: FP32 float config NOT supported (quantized configs only).");
                    }
                }
            }
            // Enable the extension if the hardware supports it.
            cooperative_vector_enabled = true;
            enable_device_extension(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
            LUISA_INFO("VK_NV_cooperative_vector extension enabled on device.");
        }
    }
    {
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &subgroup_size_control_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &subgroup_size_control_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
        if (subgroup_size_control_features.subgroupSizeControl == VK_TRUE &&
            (subgroup_size_control_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0u) {
            enable_subgroup_size_control = true;
            subgroup_size_control_enabled = true;
            _subgroup_size_control_properties = subgroup_size_control_properties;
            if (supported_ext.find(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) != supported_ext.end()) {
                enable_device_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
            }
        }
    }
#if ENABLE_HIDDEN_FEATURES
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup_memory_explicit_layout_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
        .pNext = nullptr,
        .workgroupMemoryExplicitLayout = VK_FALSE,
        .workgroupMemoryExplicitLayoutScalarBlockLayout = VK_FALSE,
        .workgroupMemoryExplicitLayout8BitAccess = VK_FALSE,
        .workgroupMemoryExplicitLayout16BitAccess = VK_FALSE};
    VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
        .pNext = nullptr,
        .maintenance5 = VK_FALSE};
    if (supported_ext.find(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME) != supported_ext.end() &&
        supported_ext.find(VK_KHR_MAINTENANCE_5_EXTENSION_NAME) != supported_ext.end()) {
        workgroup_memory_explicit_layout_features.pNext = &maintenance5_features;
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &workgroup_memory_explicit_layout_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout == VK_TRUE &&
            maintenance5_features.maintenance5 == VK_TRUE) {
            async_copy_enabled = true;
            enable_device_extension(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
            enable_device_extension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
        }
    }
#endif // ENABLE_HIDDEN_FEATURES
    if (bindless_enabled) {
        // Descriptor indexing is core in Vulkan 1.2. Query the promoted feature
        // structure directly: chaining it together with the EXT structure is
        // invalid, because both structures describe the same promoted state.
        VkPhysicalDeviceVulkan12Features descriptor_indexing_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &descriptor_indexing_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        auto supported =
            descriptor_indexing_features.descriptorIndexing == VK_TRUE &&
            descriptor_indexing_features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
            descriptor_indexing_features.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE &&
            descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE &&
            descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind == VK_TRUE &&
            descriptor_indexing_features.descriptorBindingPartiallyBound == VK_TRUE &&
            descriptor_indexing_features.runtimeDescriptorArray == VK_TRUE;
        if (supported) {
            _descriptor_indexing_properties = VkPhysicalDeviceDescriptorIndexingProperties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
                .pNext = nullptr};
            VkPhysicalDeviceProperties2 properties2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &_descriptor_indexing_properties};
            VkPhysicalDeviceMaintenance3Properties maintenance3_properties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
            _descriptor_indexing_properties.pNext = &maintenance3_properties;
            vkGetPhysicalDeviceProperties2(physical_device, &properties2);
            _descriptor_indexing_properties.pNext = nullptr;
            _bindless_heap_capacity =
                detail::plan_bindless_heap_capacity({.max_per_set_descriptors =
                                                         maintenance3_properties.maxPerSetDescriptors,
                                                     .max_per_stage_update_after_bind_samplers =
                                                         _descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindSamplers,
                                                     .max_descriptor_set_update_after_bind_samplers =
                                                         _descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSamplers,
                                                     .max_per_stage_update_after_bind_storage_buffers =
                                                         _descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindStorageBuffers,
                                                     .max_descriptor_set_update_after_bind_storage_buffers =
                                                         _descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindStorageBuffers,
                                                     .max_per_stage_update_after_bind_sampled_images =
                                                         _descriptor_indexing_properties.maxPerStageDescriptorUpdateAfterBindSampledImages,
                                                     .max_descriptor_set_update_after_bind_sampled_images =
                                                         _descriptor_indexing_properties.maxDescriptorSetUpdateAfterBindSampledImages,
                                                     .max_per_stage_update_after_bind_resources =
                                                         _descriptor_indexing_properties.maxPerStageUpdateAfterBindResources,
                                                     .max_update_after_bind_descriptors_in_all_pools =
                                                         _descriptor_indexing_properties.maxUpdateAfterBindDescriptorsInAllPools});
            supported = _bindless_heap_capacity != 0u;
            if (_bindless_heap_capacity <
                detail::requested_bindless_heap_capacity) {
                LUISA_INFO(
                    "Vulkan bindless heap capacity clamped from {} to {} by "
                    "descriptor-indexing limits.",
                    detail::requested_bindless_heap_capacity,
                    _bindless_heap_capacity);
            }
            if (supported_ext.find(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) !=
                supported_ext.end()) {
                enable_device_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            }
        }
        if (!supported) {
            _bindless_heap_capacity = 0u;
            bindless_enabled = false;
        }
    }
    auto robust_buffer_access_enabled =
        detail::plan_robust_buffer_access({.physical_device_feature =
                                               device_features.robustBufferAccess == VK_TRUE,
                                           .storage_buffer_update_after_bind = bindless_enabled,
                                           .robust_buffer_access_update_after_bind =
                                               _descriptor_indexing_properties
                                                   .robustBufferAccessUpdateAfterBind == VK_TRUE});
    if (device_features.robustBufferAccess == VK_TRUE &&
        !robust_buffer_access_enabled) {
        LUISA_INFO(
            "Vulkan robustBufferAccess disabled because bindless storage "
            "buffers use update-after-bind and the device does not support "
            "robustBufferAccessUpdateAfterBind.");
    }
    device_features.robustBufferAccess =
        robust_buffer_access_enabled ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR supported_barycentric{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
        .pNext = nullptr};
    if (supports_device_extension(
            VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_barycentric};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    }
    VkPhysicalDeviceRayQueryFeaturesKHR supported_ray_query{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = nullptr};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR supported_acceleration_structure{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &supported_ray_query};
    auto has_ray_query_extensions =
        supports_device_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
        supports_device_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        supports_device_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    if (supports_device_extension(
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
        _acceleration_structure_properties =
            VkPhysicalDeviceAccelerationStructurePropertiesKHR{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
                .pNext = nullptr};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &_acceleration_structure_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    }
    if (has_ray_query_extensions) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_acceleration_structure};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    }
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR supported_ray_tracing_pipeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = nullptr};
    VkPhysicalDeviceRayTracingMotionBlurFeaturesNV supported_motion_blur{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
        .pNext = &supported_ray_tracing_pipeline};
    auto has_motion_blur_extensions =
        has_ray_query_extensions &&
        supports_device_extension(
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
        supports_device_extension(
            VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
    if (has_motion_blur_extensions) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_motion_blur};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    }
    auto raytracing_requested = raytracing_enabled;
    auto motion_blur_requested =
        raytracing_requested &&
        (!_config_ext || _config_ext->enable_motion_blur());
    auto optional_device_features = detail::plan_optional_device_features(
        {
            .fragment_shader_barycentric_extension =
                supports_device_extension(
                    VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME),
            .fragment_shader_barycentric_feature =
                supported_barycentric.fragmentShaderBarycentric == VK_TRUE,
            .ray_query_extension =
                supports_device_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME),
            .ray_query_feature = supported_ray_query.rayQuery == VK_TRUE,
            .acceleration_structure_extension =
                supports_device_extension(
                    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME),
            .acceleration_structure_feature =
                supported_acceleration_structure.accelerationStructure == VK_TRUE,
            .deferred_host_operations_extension =
                supports_device_extension(
                    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME),
            .buffer_device_address =
                (api_version >= VK_API_VERSION_1_2 ||
                 supports_device_extension(
                     VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) &&
                _vk_device->features_12.bufferDeviceAddress == VK_TRUE,
            .ray_tracing_pipeline_extension =
                supports_device_extension(
                    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME),
            .ray_tracing_pipeline_feature =
                supported_ray_tracing_pipeline.rayTracingPipeline == VK_TRUE,
            .ray_traversal_primitive_culling_feature =
                supported_ray_tracing_pipeline.rayTraversalPrimitiveCulling ==
                VK_TRUE,
            .ray_tracing_motion_blur_extension =
                supports_device_extension(
                    VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME),
            .ray_tracing_motion_blur_feature =
                supported_motion_blur.rayTracingMotionBlur == VK_TRUE,
        },
        {.ray_query = raytracing_requested,
         .ray_tracing_motion_blur = motion_blur_requested});
    enable_barycentric =
        optional_device_features.fragment_shader_barycentric;
    raytracing_enabled = optional_device_features.ray_query;
    enable_motion_blur =
        optional_device_features.ray_tracing_motion_blur;
    motion_blur_enabled = enable_motion_blur;
    if (raytracing_requested && !raytracing_enabled) {
        LUISA_WARNING(
            "Vulkan ray query disabled: rayQuery={} accelerationStructure={} "
            "bufferDeviceAddress={} and required extensions={}",
            supported_ray_query.rayQuery == VK_TRUE,
            supported_acceleration_structure.accelerationStructure == VK_TRUE,
            _vk_device->features_12.bufferDeviceAddress == VK_TRUE,
            has_ray_query_extensions);
    }
    if (motion_blur_requested && raytracing_enabled &&
        !motion_blur_enabled) {
        LUISA_INFO(
            "Vulkan motion blur disabled: rayTracingPipeline={} "
            "rayTraversalPrimitiveCulling={} rayTracingMotionBlur={} and "
            "required extensions={}",
            supported_ray_tracing_pipeline.rayTracingPipeline == VK_TRUE,
            supported_ray_tracing_pipeline.rayTraversalPrimitiveCulling ==
                VK_TRUE,
            supported_motion_blur.rayTracingMotionBlur == VK_TRUE,
            has_motion_blur_extensions);
    }
    if (external_device != VK_NULL_HANDLE) {
        // Vulkan exposes physical-device support, not the feature bits that
        // were enabled when an imported VkDevice was created. Reset every
        // optional feature discovered above; physical support alone must
        // neither enter a feature chain nor become a runtime capability.
        device_features = {};
        narrow_numeric_features = {};
        enable_float8 = false;
        enable_buffer_int64_atomics = false;
        enable_shared_int64_atomics = false;
        raytracing_enabled = false;
        enable_barycentric = false;
        enable_motion_blur = false;
        motion_blur_enabled = false;
        enable_buffer_float32_atomics = false;
        enable_buffer_float32_atomic_add = false;
        enable_buffer_float32_atomic_min_max = false;
        enable_shared_float32_atomics = false;
        enable_shared_float32_atomic_add = false;
        enable_shared_float32_atomic_min_max = false;
        enable_subgroup_size_control = false;
        subgroup_size_control_enabled = false;
        _subgroup_size_control_properties = {};
        enable_subgroup_extended_types = false;
#ifndef NDEBUG
        enable_shader_device_clock = false;
#endif
        enabled_cooperative_matrix_ext = CooperativeMatrixExt::None;
        cooperative_vector_enabled = false;
        cooperative_vector_fp32_enabled = false;
#if ENABLE_HIDDEN_FEATURES
        async_copy_enabled = false;
#endif
        interop_enabled = false;
        device_address_enabled = false;
        surface_enabled = false;
        _enable_device_exts.clear();
    }
    if (enable_barycentric) {
        enable_device_extension(
            VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    }
    if (raytracing_enabled) {
        enable_device_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        enable_device_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        enable_device_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }
    if (enable_motion_blur) {
        enable_device_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        enable_device_extension(VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME);
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
    // bufferDeviceAddress is core in Vulkan 1.2. The extension name is only
    // needed for an older physical-device API, but the feature bit is required
    // in either case.
    if (device_address_enabled) {
        auto has_core_buffer_device_address =
            api_version >= VK_API_VERSION_1_2;
        auto has_buffer_device_address_extension =
            supports_device_extension(
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        auto has_buffer_device_address_feature =
            _vk_device->features_12.bufferDeviceAddress == VK_TRUE;
        if ((!has_core_buffer_device_address &&
             !has_buffer_device_address_extension) ||
            !has_buffer_device_address_feature) {
            LUISA_WARNING(
                "bufferDeviceAddress is unavailable (core={}, extension={}, "
                "feature={}); disabling device address.",
                has_core_buffer_device_address,
                has_buffer_device_address_extension,
                has_buffer_device_address_feature);
            device_address_enabled = false;
        } else if (!has_core_buffer_device_address) {
            enable_device_extension(
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }
    }
    luisa::vector<luisa::string> extra_exts = [&]() {
        if (_config_ext && external_device == VK_NULL_HANDLE) {
            return _config_ext->extra_device_exts();
        } else {
            return luisa::vector<luisa::string>{};
        }
    }();
    for (auto &i : extra_exts) {
        if (supported_ext.find(i) != supported_ext.end())
            enable_device_extension(i.c_str());
    }
    void *feature_next{nullptr};
    if (_config_ext && external_device == VK_NULL_HANDLE) {
        feature_next = _config_ext->device_feature_settings();
    }
    auto custom_feature_chain_validation =
        detail::validate_device_feature_settings_chain(
            static_cast<const VkBaseInStructure *>(feature_next));
    if (!custom_feature_chain_validation) [[unlikely]] {
        if (custom_feature_chain_validation.error ==
            detail::DeviceFeatureChainValidationError::CYCLE) {
            LUISA_ERROR(
                "Vulkan device_feature_settings() returned a cyclic pNext "
                "chain.");
        } else if (custom_feature_chain_validation.error ==
                   detail::DeviceFeatureChainValidationError::DUPLICATE_STRUCTURE) {
            LUISA_ERROR(
                "Vulkan device_feature_settings() pNext nodes {} and {} "
                "repeat VkStructureType {}.",
                custom_feature_chain_validation.related_node_index,
                custom_feature_chain_validation.node_index,
                static_cast<int32_t>(
                    custom_feature_chain_validation.structure_type));
        } else {
            LUISA_ERROR(
                "Vulkan device_feature_settings() pNext node {} uses reserved "
                "VkStructureType {}; this feature structure is owned by the "
                "Vulkan backend or conflicts with one of its promoted feature "
                "structures.",
                custom_feature_chain_validation.node_index,
                static_cast<int32_t>(
                    custom_feature_chain_validation.structure_type));
        }
    }
    VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR raster_bary{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
        .pNext = feature_next,
        .fragmentShaderBarycentric =
            enable_barycentric ? VK_TRUE : VK_FALSE};
    if (enable_barycentric) {
        feature_next = &raster_bary;
    }

    // 16-bit storage features are set in VkPhysicalDeviceVulkan11Features below
    VkPhysicalDeviceRayQueryFeaturesKHR enable_rayquery_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .pNext = feature_next,
        .rayQuery = raytracing_enabled ? VK_TRUE : VK_FALSE};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enabled_acceleration_structure_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &enable_rayquery_features,
        .accelerationStructure = raytracing_enabled ? VK_TRUE : VK_FALSE};
    VkPhysicalDeviceRayTracingMotionBlurFeaturesNV motion_blur_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
        .pNext = &enabled_acceleration_structure_features,
        .rayTracingMotionBlur = enable_motion_blur ? VK_TRUE : VK_FALSE,
        .rayTracingMotionBlurPipelineTraceRaysIndirect = VK_FALSE};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = &motion_blur_features,
        .rayTracingPipeline = enable_motion_blur ? VK_TRUE : VK_FALSE,
        .rayTraversalPrimitiveCulling =
            enable_motion_blur ? VK_TRUE : VK_FALSE};
    if (raytracing_enabled) {
        if (enable_motion_blur) {
            feature_next = &rt_pipeline_features;
        } else {
            feature_next = &enabled_acceleration_structure_features;
        }
    }
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT float_atomic_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = feature_next,
        .shaderBufferFloat32Atomics =
            enable_buffer_float32_atomics ? VK_TRUE : VK_FALSE,
        .shaderBufferFloat32AtomicAdd =
            enable_buffer_float32_atomic_add ? VK_TRUE : VK_FALSE,
        .shaderSharedFloat32Atomics =
            enable_shared_float32_atomics ? VK_TRUE : VK_FALSE,
        .shaderSharedFloat32AtomicAdd =
            enable_shared_float32_atomic_add ? VK_TRUE : VK_FALSE};
    if (enable_buffer_float32_atomics ||
        enable_buffer_float32_atomic_add ||
        enable_shared_float32_atomics ||
        enable_shared_float32_atomic_add) {
        feature_next = &float_atomic_features;
    }
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT float_atomic2_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
        .pNext = feature_next,
        .shaderBufferFloat32AtomicMinMax =
            enable_buffer_float32_atomic_min_max ? VK_TRUE : VK_FALSE,
        .shaderSharedFloat32AtomicMinMax =
            enable_shared_float32_atomic_min_max ? VK_TRUE : VK_FALSE};
    if (enable_buffer_float32_atomic_min_max ||
        enable_shared_float32_atomic_min_max) {
        feature_next = &float_atomic2_features;
    }
    VkPhysicalDeviceShaderFloat8FeaturesEXT float8_features{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
        .pNext = feature_next,
        .shaderFloat8 = enable_float8 ? VK_TRUE : VK_FALSE,
        .shaderFloat8CooperativeMatrix = VK_FALSE};
    if (enable_float8) {
        feature_next = &float8_features;
    }
#ifndef NDEBUG
    VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_features{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
        .pNext = feature_next,
        .shaderSubgroupClock = VK_FALSE,
        .shaderDeviceClock =
            enable_shader_device_clock ? VK_TRUE : VK_FALSE};
    if (enable_shader_device_clock) {
        feature_next = &shader_clock_features;
    }
#endif
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .pNext = feature_next,
        .subgroupSizeControl = VK_TRUE,
        .computeFullSubgroups = VK_FALSE};
    if (enable_subgroup_size_control) {
        feature_next = &subgroup_size_control_feature;
    }
    if (enabled_cooperative_matrix_ext != CooperativeMatrixExt::None) {
        switch (enabled_cooperative_matrix_ext) {
            case CooperativeMatrixExt::KHRAndNV2:
                cooperative_matrix_2_features_nv.pNext = feature_next;
                feature_next = &cooperative_matrix_2_features_nv;
                cooperative_matrix_features_khr.pNext = feature_next;
                feature_next = &cooperative_matrix_features_khr;
                break;
            case CooperativeMatrixExt::KHR:
                cooperative_matrix_features_khr.pNext = feature_next;
                feature_next = &cooperative_matrix_features_khr;
                break;
            case CooperativeMatrixExt::NV:
                cooperative_matrix_features_nv.pNext = feature_next;
                feature_next = &cooperative_matrix_features_nv;
                break;
            default:
                break;
        }
    }
    if (cooperative_vector_enabled) {
        cooperative_vector_features_nv.pNext = feature_next;
        feature_next = &cooperative_vector_features_nv;
    }
#if ENABLE_HIDDEN_FEATURES
    if (async_copy_enabled) {
        maintenance5_features.pNext = feature_next;
        feature_next = &maintenance5_features;
        workgroup_memory_explicit_layout_features.pNext = feature_next;
        feature_next = &workgroup_memory_explicit_layout_features;
    }
#endif
    VkPhysicalDeviceSynchronization2Features barrier_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .pNext = feature_next,
        .synchronization2 =
            required_device_features.supported ? VK_TRUE : VK_FALSE};
    // Variable pointers to storage buffers are required by cooperative
    // vector/matrix SPIR-V operations (which take buffer pointers via
    // [[vk::ext_reference]]). Only enable the device feature when a
    // cooperative-matrix/vector extension is active.
    bool enable_variable_pointers_storage_buffer = false;
    if (enabled_cooperative_matrix_ext != CooperativeMatrixExt::None || cooperative_vector_enabled) {
        VkPhysicalDeviceVulkan11Features vk11_vp_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vk11_vp_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (vk11_vp_features.variablePointersStorageBuffer == VK_TRUE &&
            vk11_vp_features.variablePointers == VK_TRUE) {
            enable_variable_pointers_storage_buffer = true;
        }
    }
    VkPhysicalDeviceVulkan11Features vk11_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &barrier_feature,
        .storageBuffer16BitAccess =
            narrow_numeric_features.storage_buffer_16bit_access ?
                VK_TRUE :
                VK_FALSE,
        .uniformAndStorageBuffer16BitAccess =
            narrow_numeric_features.uniform_storage_buffer_16bit_access ?
                VK_TRUE :
                VK_FALSE,
        .variablePointersStorageBuffer = enable_variable_pointers_storage_buffer ? VK_TRUE : VK_FALSE,
        .variablePointers = enable_variable_pointers_storage_buffer ? VK_TRUE : VK_FALSE};
    auto vk_bindless_enabled = bindless_enabled ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan12Features vk12_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vk11_feature,
        .storageBuffer8BitAccess =
            narrow_numeric_features.storage_buffer_8bit_access ?
                VK_TRUE :
                VK_FALSE,
        .uniformAndStorageBuffer8BitAccess =
            narrow_numeric_features.uniform_storage_buffer_8bit_access ?
                VK_TRUE :
                VK_FALSE,
        .shaderBufferInt64Atomics =
            enable_buffer_int64_atomics ? VK_TRUE : VK_FALSE,
        .shaderSharedInt64Atomics =
            enable_shared_int64_atomics ? VK_TRUE : VK_FALSE,
        .shaderFloat16 = narrow_numeric_features.shader_float16 ?
                             VK_TRUE :
                             VK_FALSE,
        .shaderInt8 = narrow_numeric_features.shader_int8 ?
                          VK_TRUE :
                          VK_FALSE,
        .descriptorIndexing = vk_bindless_enabled,
        .shaderSampledImageArrayNonUniformIndexing = vk_bindless_enabled,
        .shaderStorageBufferArrayNonUniformIndexing = vk_bindless_enabled,
        .descriptorBindingSampledImageUpdateAfterBind = vk_bindless_enabled,
        .descriptorBindingStorageBufferUpdateAfterBind = vk_bindless_enabled,
        .descriptorBindingPartiallyBound = vk_bindless_enabled,
        .runtimeDescriptorArray = vk_bindless_enabled,

        .shaderSubgroupExtendedTypes = enable_subgroup_extended_types ? VK_TRUE : VK_FALSE,

        .timelineSemaphore =
            required_device_features.supported ? VK_TRUE : VK_FALSE,
        .bufferDeviceAddress = device_address_enabled ? VK_TRUE : VK_FALSE};
    VK_CHECK_RESULT(_vk_device->create_logical_device(device_features, _enable_device_exts, &vk12_feature, surface_enabled));
    if (external_device != VK_NULL_HANDLE) {
        _vk_device->queue_family_indices.graphics =
            external_graphics_queue_family_index;
        _vk_device->queue_family_indices.compute =
            external_compute_queue_family_index;
        _vk_device->queue_family_indices.transfer =
            external_copy_queue_family_index;
        // Sparse capabilities are intentionally unavailable on imported
        // logical devices until the import contract can attest both feature
        // enablement and an actual sparse-capable queue.
        _vk_device->queue_family_indices.sparse =
            VK_QUEUE_FAMILY_IGNORED;
    }
    // Vulkan cannot query which optional features were enabled on an imported
    // logical device. Be conservative there: only advertise the features that
    // this backend itself requested while creating the device.
    subgroup_extended_types_enabled =
        external_device == VK_NULL_HANDLE && enable_subgroup_extended_types;
#ifndef NDEBUG
    _shader_device_clock_enabled =
        external_device == VK_NULL_HANDLE && enable_shader_device_clock;
#endif
    if (external_device == VK_NULL_HANDLE) {
        _numeric_features = {
            .shader_float8 = enable_float8,
            .shader_float16 = narrow_numeric_features.shader_float16,
            .shader_float64 = device_features.shaderFloat64 == VK_TRUE,
            .shader_int8 = narrow_numeric_features.shader_int8,
            .shader_int16 = device_features.shaderInt16 == VK_TRUE,
            .shader_int64 = device_features.shaderInt64 == VK_TRUE,
            .storage_buffer_8bit_access =
                narrow_numeric_features.storage_buffer_8bit_access,
            .uniform_storage_buffer_8bit_access =
                narrow_numeric_features.uniform_storage_buffer_8bit_access,
            .storage_buffer_16bit_access =
                narrow_numeric_features.storage_buffer_16bit_access,
            .uniform_storage_buffer_16bit_access =
                narrow_numeric_features.uniform_storage_buffer_16bit_access};
        _float_atomic_features = {
            .shader_buffer_float32_atomics =
                enable_buffer_float32_atomics,
            .shader_buffer_float32_atomic_add =
                enable_buffer_float32_atomic_add,
            .shader_buffer_float32_atomic_min_max =
                enable_buffer_float32_atomic_min_max,
            .shader_shared_float32_atomics =
                enable_shared_float32_atomics,
            .shader_shared_float32_atomic_add =
                enable_shared_float32_atomic_add,
            .shader_shared_float32_atomic_min_max =
                enable_shared_float32_atomic_min_max};
        _int64_atomic_features = {
            .shader_buffer_int64_atomics =
                enable_buffer_int64_atomics,
            .shader_shared_int64_atomics =
                enable_shared_int64_atomics};
    } else {
        _numeric_features = {};
        _float_atomic_features = {};
        _int64_atomic_features = {};
    }
    auto device = _vk_device->logical_device;
    volkLoadDevice(device);

    if (external_device == VK_NULL_HANDLE) {
        vkGetDeviceQueue(device, _vk_device->queue_family_indices.graphics, 0, &_graphics_queue);
        vkGetDeviceQueue(device, _vk_device->queue_family_indices.compute, 0, &_compute_queue);
        vkGetDeviceQueue(device, _vk_device->queue_family_indices.transfer, 0, &_copy_queue);
        if (_vk_device->enabled_features.sparseBinding == VK_TRUE) {
            vkGetDeviceQueue(
                device, _vk_device->queue_family_indices.sparse,
                0, &_sparse_queue);
        }
    }
    LUISA_ASSERT(
        _graphics_queue != VK_NULL_HANDLE &&
            _compute_queue != VK_NULL_HANDLE &&
            _copy_queue != VK_NULL_HANDLE,
        "Vulkan queue acquisition returned a null graphics, compute, or copy queue.");
    LUISA_ASSERT(
        _vk_device->enabled_features.sparseBinding != VK_TRUE ||
            _sparse_queue != VK_NULL_HANDLE,
        "Vulkan sparseBinding is enabled, but sparse queue acquisition "
        "returned a null handle.");
    auto queue_lock_plan = detail::plan_queue_locks(
        std::array{
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(_graphics_queue)),
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(_compute_queue)),
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(_copy_queue)),
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(_sparse_queue))});
    std::array queue_locks{
        &_graphics_queue_mtx,
        &_compute_queue_mtx,
        &_copy_queue_mtx,
        &_sparse_queue_mtx};
    _graphics_queue_lock =
        queue_locks[queue_lock_plan.lock_indices[0u]];
    _compute_queue_lock =
        queue_locks[queue_lock_plan.lock_indices[1u]];
    _copy_queue_lock =
        queue_locks[queue_lock_plan.lock_indices[2u]];
    _sparse_queue_lock =
        queue_locks[queue_lock_plan.lock_indices[3u]];
    _pso_header.headerSize = sizeof(VkPipelineCacheHeaderVersionOne);
    _pso_header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    _pso_header.vendorID = _vk_device->properties.vendorID;
    _pso_header.deviceID = _vk_device->properties.deviceID;
    memcpy(_pso_header.pipelineCacheUUID, _vk_device->properties.pipelineCacheUUID, VK_UUID_SIZE);
    _allocator.create(*this);
    // bind desc_pool
    VkDescriptorBindingFlags desc_binding_flag =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindless_binding_flags{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 1,
        .pBindingFlags = &desc_binding_flag};
    std::array<VkDescriptorSetLayoutBinding, 3u>
        global_bindless_bindings{};
    std::array<VkDescriptorSetLayoutCreateInfo, 3u>
        global_bindless_layouts{};
    if (bindless_enabled) {
        constexpr std::array descriptor_types{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
        for (auto i = 0u; i < descriptor_types.size(); ++i) {
            global_bindless_bindings[i] = VkDescriptorSetLayoutBinding{
                .binding = 0u,
                .descriptorType = descriptor_types[i],
                .descriptorCount = _bindless_heap_capacity,
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr};
            global_bindless_layouts[i] = VkDescriptorSetLayoutCreateInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = &bindless_binding_flags,
                .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                .bindingCount = 1u,
                .pBindings = &global_bindless_bindings[i]};
        }
        // Some implementations apply additional per-layout restrictions not
        // reflected by descriptor-indexing properties. Negotiate one capacity
        // accepted by all three persistent global layouts before creating any
        // descriptor pool or layout.
        auto planned_capacity = _bindless_heap_capacity;
        _bindless_heap_capacity = detail::negotiate_bindless_heap_capacity(
            planned_capacity, [&](uint32_t capacity) noexcept {
                for (auto i = 0u; i < global_bindless_layouts.size(); ++i) {
                    global_bindless_bindings[i].descriptorCount = capacity;
                    VkDescriptorSetLayoutSupport support{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
                    vkGetDescriptorSetLayoutSupport(
                        logic_device(), &global_bindless_layouts[i], &support);
                    if (support.supported != VK_TRUE) { return false; }
                }
                return true;
            });
        LUISA_ASSERT(
            _bindless_heap_capacity != 0u,
            "Vulkan device does not support any usable global bindless "
            "descriptor layout capacity (planned upper bound: {}).",
            planned_capacity);
        for (auto &binding : global_bindless_bindings) {
            binding.descriptorCount = _bindless_heap_capacity;
        }
        if (_bindless_heap_capacity < planned_capacity) {
            LUISA_INFO(
                "Vulkan bindless heap capacity negotiated from {} to {} by "
                "descriptor-set layout support.",
                planned_capacity, _bindless_heap_capacity);
        }
    }
    // bindless buffer desc_pool
    if (bindless_enabled) {
        {
            buffer_heap_pool.full_size = _bindless_heap_capacity;
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
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &global_bindless_layouts[0u], alloc_callbacks(), &_bdls_buffer_set_layout));
            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = _bdls_buffer_desc_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &_bdls_buffer_set_layout};
            VK_CHECK_RESULT(vkAllocateDescriptorSets(logic_device(), &alloc_info, &_bdls_buffer_set));
        }
        // bindless tex2d desc_pool
        {
            tex2d_heap_pool.full_size = _bindless_heap_capacity;
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
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &global_bindless_layouts[1u], alloc_callbacks(), &_bdls_tex2d_set_layout));
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
            tex3d_heap_pool.full_size = _bindless_heap_capacity;
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
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(logic_device(), &global_bindless_layouts[2u], alloc_callbacks(), &_bdls_tex3d_set_layout));
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
        pool_size.descriptorCount = detail::sampler_heap_size;
        pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        VkDescriptorPoolCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = 0,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size};
        VK_CHECK_RESULT(vkCreateDescriptorPool(logic_device(), &create_info, alloc_callbacks(), &_sampler_pool));
        _samplers.resize(detail::sampler_heap_size);
        for (auto address_index :
             vstd::range(detail::sampler_address_count))
            for (auto filter_index :
                 vstd::range(detail::sampler_filter_count)) {
                auto idx = detail::sampler_heap_index(
                    filter_index, address_index);
                VkSamplerCreateInfo info{
                    VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    nullptr,
                    0};
                info.maxAnisotropy = 1.0f;

                switch (static_cast<Sampler::Filter>(filter_index)) {
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
                        // Preserve the fixed 16-entry descriptor ABI even on
                        // devices without samplerAnisotropy. The anisotropic
                        // slots contain valid linear placeholders there, but
                        // shader/runtime validation makes them unreachable.
                        info.anisotropyEnable =
                            sampler_anisotropy_enabled ? VK_TRUE : VK_FALSE;
                        info.maxAnisotropy = _max_sampler_anisotropy;
                        break;
                    default: LUISA_ASSUME(false); break;
                }

                VkSamplerAddressMode address = [&] {
                    switch (static_cast<Sampler::Address>(address_index)) {
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
            detail::sampler_heap_size,
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
    {
        std::lock_guard lck(g_dxc_mutex);
        if (g_dxc_ref_count > 0 && --g_dxc_ref_count == 0) {
            if (g_dxc_compiler_initialized) {
                g_dxc_compiler.destroy();
                g_dxc_compiler_initialized = false;
            }
            g_dxc_runtime_directory.clear();
        }
    }
#endif
    if (external_device) {
        _vk_device->logical_device = nullptr;
        _vk_device->physical_device = nullptr;
    }
}
void *Device::native_handle() const noexcept { return _vk_device->logical_device; }

luisa::shared_ptr<NativeImageState> Device::acquire_native_image_state(
    VkImage image, VkFormat format, uint dimension, uint3 size,
    uint mip_levels, bool simultaneous_access) {
    auto identity = detail::native_image_identity(image);
    std::lock_guard lock{_native_image_state_mtx};
    auto sweep_period = std::max<size_t>(
        64u, _native_image_states.size() / 4u);
    auto expired_hint =
        _native_image_state_expiration_counter->load(
            std::memory_order_relaxed);
    if (expired_hint >= 64u ||
        ++_native_image_state_acquisitions_since_sweep >= sweep_period) {
        _native_image_state_acquisitions_since_sweep = 0u;
        _native_image_state_expiration_counter->exchange(
            0u, std::memory_order_relaxed);
        for (auto iter = _native_image_states.begin();
             iter != _native_image_states.end();) {
            if (iter->second.expired()) {
                iter = _native_image_states.erase(iter);
            } else {
                ++iter;
            }
        }
    }
    if (auto iter = _native_image_states.find(identity);
        iter != _native_image_states.end()) {
        if (auto state = iter->second.lock()) {
            LUISA_ASSERT(
                state->image == image && state->format == format &&
                    state->dimension == dimension &&
                    state->size.x == size.x && state->size.y == size.y &&
                    state->size.z == size.z &&
                    state->mip_levels == mip_levels &&
                    state->simultaneous_access == simultaneous_access,
                "Vulkan wrappers aliasing image 0x{:016x} disagree on its "
                "native metadata (format, dimension, extent, mip count, or "
                "simultaneous-access policy). Mutable-format aliases are not "
                "supported.",
                identity);
            return state;
        }
        _native_image_states.erase(iter);
    }
    auto state = luisa::make_shared<NativeImageState>(
        image, format, dimension, size, mip_levels, simultaneous_access,
        _native_image_state_expiration_counter);
    _native_image_states.emplace(identity, state);
    return state;
}

BufferCreationInfo Device::create_buffer(const luisa::compute::Type *element, size_t elem_count, void *external_ptr) noexcept {
    if (element && element->is_custom()) [[unlikely]] {
        LUISA_ASSERT(
            element == Type::of<IndirectKernelDispatch>(),
            "Unsupported Vulkan custom buffer element type '{}'.",
            element->description());
        LUISA_ASSERT(
            external_ptr == nullptr,
            "Vulkan indirect-dispatch buffers cannot wrap external memory: "
            "their header/record ABI is backend-owned.");
        size_t indirect_size = 0u;
        LUISA_ASSERT(
            IndirectDispatchLayout::try_total_size(
                elem_count, indirect_size) &&
                indirect_size <=
                    properties().limits.maxStorageBufferRange,
            "Vulkan indirect-dispatch buffer for {} records exceeds the "
            "device storage-buffer range limit {}.",
            elem_count, properties().limits.maxStorageBufferRange);
        auto ptr = new IndirectBuffer(this, elem_count);
        BufferCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(ptr);
        info.native_handle = ptr->vk_buffer();
        info.element_stride = IndirectDispatchLayout::record_size;
        info.total_size_bytes = ptr->byte_size();
        return info;
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
void Device::destroy_buffer(uint64_t handle) noexcept {
    auto *buffer = reinterpret_cast<Buffer *>(handle);
    // Keep stream membership and lifetime stable until every per-stream
    // barrier cache has dropped the wrapper. destroy_stream takes this same
    // lock before deleting a Stream and cannot invalidate an entry midway.
    std::lock_guard lock{_stream_mtx};
    for (auto *stream : _streams) {
        stream->remove_resource_state(buffer);
    }
    delete buffer;
}

// texture
ResourceCreationInfo Device::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth, uint mipmap_levels,
    void *external_native_handle, bool simultaneous_access,
    bool allow_raster_target) noexcept {

    auto size = uint3(width, height, depth);
    auto ptr = external_native_handle == nullptr ?
                   new Texture(
                       this, dimension, format, size, mipmap_levels,
                       simultaneous_access, allow_raster_target) :
                   new Texture(
                       this, static_cast<VkImage>(external_native_handle),
                       dimension, format, size,
                       mipmap_levels, simultaneous_access);
    ResourceCreationInfo r{
        .handle = reinterpret_cast<uint64_t>(ptr),
        .native_handle = ptr->vk_image()};
    return r;
}
void Device::destroy_texture(uint64_t handle) noexcept {
    auto *texture = reinterpret_cast<Texture *>(handle);
    std::lock_guard lock{_stream_mtx};
    for (auto *stream : _streams) {
        stream->remove_resource_state(texture);
    }
    delete texture;
}
luisa::FirstFit::Node *Device::HeapAlloc::sub_alloc(uint32_t size) {
    std::lock_guard lck{mtx};
    LUISA_ASSERT(size > 0u && size <= full_size,
                 "Invalid Vulkan bindless contiguous allocation size {} for "
                 "a heap of {} descriptors.",
                 size, full_size);
    auto ptr = sub_allocator.allocate_best_fit(size);
    LUISA_ASSERT(ptr != nullptr,
                 "Vulkan bindless contiguous descriptor allocator exhausted.");
    auto index = full_size - (ptr->offset() + ptr->size());
    if (index < count) [[unlikely]] {
        sub_allocator.free(ptr);
        LUISA_ERROR(
            "Vulkan bindless descriptor heap exhausted: contiguous allocation "
            "of {} slots overlaps {} individually allocated slots in a heap "
            "of {} descriptors.",
            size, count, full_size);
    }
    sub_allocations.emplace_back(ptr);
    return ptr;
}
void Device::HeapAlloc::free(luisa::FirstFit::Node *ptr) {
    std::lock_guard lck{mtx};
    auto iter = std::find(sub_allocations.begin(), sub_allocations.end(), ptr);
    LUISA_ASSERT(iter != sub_allocations.end(),
                 "Attempted to free an unknown Vulkan bindless allocation.");
    sub_allocations.erase(iter);
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
    {
        std::lock_guard lck{_stream_mtx};
        _streams.emplace(ptr);
    }
    ResourceCreationInfo info{
        .handle = reinterpret_cast<uint64_t>(ptr),
        .native_handle = ptr->queue()};
    return info;
}
void Device::destroy_stream(uint64_t handle) noexcept {
    auto *stream = reinterpret_cast<Stream *>(handle);
    {
        std::lock_guard lck{_stream_mtx};
        _streams.erase(stream);
    }
    delete stream;
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

bool Device::print_code() {
    return luisa::compute::backend_print_code_enabled();
}

luisa::string Device::query(luisa::string_view property) noexcept {
    if (property == "shader_device_clock") {
        return _shader_device_clock_enabled ? "true" : "false";
    }
    if (property == "buffer_device_address") {
        return device_address_enabled ? "true" : "false";
    }
    if (property == "shader_int64") {
        return _numeric_features.shader_int64 ? "true" : "false";
    }
    return DeviceInterface::query(property);
}

uint64_t Device::enabled_spirv_artifact_features() const noexcept {
    using namespace lc::spirv;
    SpirvTargetFeatureMask mask{};
    auto enable = [&mask](bool enabled,
                          SpirvTargetFeatureMask feature) noexcept {
        if (enabled) { mask |= feature; }
    };
    auto owned_logical_device = !external_device;
    enable(owned_logical_device &&
               _vk_device->enabled_features
                       .shaderSampledImageArrayDynamicIndexing == VK_TRUE,
           target_feature::sampled_image_array_dynamic_indexing);
    enable(owned_logical_device && bindless_enabled,
           target_feature::sampled_image_array_non_uniform_indexing);
    enable(owned_logical_device &&
               _vk_device->enabled_features.shaderResourceMinLod == VK_TRUE,
           target_feature::shader_resource_min_lod);
    enable(_float_atomic_features.shader_buffer_float32_atomics,
           target_feature::shader_buffer_float32_atomics);
    enable(_float_atomic_features.shader_buffer_float32_atomic_add,
           target_feature::shader_buffer_float32_atomic_add);
    enable(_float_atomic_features.shader_buffer_float32_atomic_min_max,
           target_feature::shader_buffer_float32_atomic_min_max);
    enable(_float_atomic_features.shader_shared_float32_atomics,
           target_feature::shader_shared_float32_atomics);
    enable(_float_atomic_features.shader_shared_float32_atomic_add,
           target_feature::shader_shared_float32_atomic_add);
    enable(_float_atomic_features.shader_shared_float32_atomic_min_max,
           target_feature::shader_shared_float32_atomic_min_max);

    VkPhysicalDeviceSubgroupProperties subgroup_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_properties};
    vkGetPhysicalDeviceProperties2(physical_device(), &properties2);
    auto subgroup_compute_supported =
        (subgroup_properties.supportedStages &
         VK_SHADER_STAGE_COMPUTE_BIT) != 0u;
    auto subgroup_operation_supported =
        [&](VkSubgroupFeatureFlagBits feature) noexcept {
            return subgroup_compute_supported &&
                   (subgroup_properties.supportedOperations & feature) != 0u;
        };
    enable(subgroup_operation_supported(VK_SUBGROUP_FEATURE_BASIC_BIT),
           target_feature::subgroup_basic);
    enable(subgroup_operation_supported(VK_SUBGROUP_FEATURE_VOTE_BIT),
           target_feature::subgroup_vote);
    enable(subgroup_operation_supported(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT),
           target_feature::subgroup_arithmetic);
    enable(subgroup_operation_supported(VK_SUBGROUP_FEATURE_BALLOT_BIT),
           target_feature::subgroup_ballot);
    enable(subgroup_operation_supported(VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
           target_feature::subgroup_shuffle);
    enable(subgroup_extended_types_enabled,
           target_feature::subgroup_extended_types);

    enable(owned_logical_device &&
               _vk_device->enabled_features
                       .shaderStorageImageReadWithoutFormat == VK_TRUE,
           target_feature::storage_image_read_without_format);
    enable(owned_logical_device &&
               _vk_device->enabled_features
                       .shaderStorageImageWriteWithoutFormat == VK_TRUE,
           target_feature::storage_image_write_without_format);
    enable(_numeric_features.shader_float8,
           target_feature::shader_float8);
    enable(_numeric_features.shader_float16,
           target_feature::shader_float16);
    enable(_numeric_features.shader_float64,
           target_feature::shader_float64);
    enable(_numeric_features.shader_int8, target_feature::shader_int8);
    enable(_numeric_features.shader_int16, target_feature::shader_int16);
    enable(_numeric_features.shader_int64, target_feature::shader_int64);
    enable(_numeric_features.storage_buffer_8bit_access,
           target_feature::storage_buffer_8bit_access);
    enable(_numeric_features.uniform_storage_buffer_8bit_access,
           target_feature::uniform_storage_buffer_8bit_access);
    enable(_numeric_features.storage_buffer_16bit_access,
           target_feature::storage_buffer_16bit_access);
    enable(_numeric_features.uniform_storage_buffer_16bit_access,
           target_feature::uniform_storage_buffer_16bit_access);
    enable(owned_logical_device && raytracing_enabled,
           target_feature::ray_query);
    enable(sampler_anisotropy_enabled,
           target_feature::sampler_anisotropy);
    enable(_int64_atomic_features.shader_buffer_int64_atomics,
           target_feature::shader_buffer_int64_atomics);
    enable(_int64_atomic_features.shader_shared_int64_atomics,
           target_feature::shader_shared_int64_atomics);

    auto descriptor_indexing_enabled =
        owned_logical_device && bindless_enabled;
    enable(descriptor_indexing_enabled,
           target_feature::descriptor_indexing);
    enable(descriptor_indexing_enabled,
           target_feature::runtime_descriptor_array);
    enable(descriptor_indexing_enabled,
           target_feature::descriptor_binding_partially_bound);
    enable(descriptor_indexing_enabled,
           target_feature::storage_buffer_array_non_uniform_indexing);
    enable(descriptor_indexing_enabled,
           target_feature::descriptor_binding_sampled_image_update_after_bind);
    enable(descriptor_indexing_enabled,
           target_feature::descriptor_binding_storage_buffer_update_after_bind);
    enable(owned_logical_device &&
               _vk_device->enabled_features
                       .shaderStorageBufferArrayDynamicIndexing == VK_TRUE,
           target_feature::storage_buffer_array_dynamic_indexing);
    enable(owned_logical_device && _shader_device_clock_enabled,
           target_feature::shader_device_clock);
    enable(owned_logical_device && device_address_enabled,
           target_feature::buffer_device_address);
    return mask;
}

#ifdef LUISA_XIR_TO_SPIRV
[[nodiscard]] static uint64_t xir_spirv_environment_hash() noexcept {
    auto hash_env = [](const char *name) noexcept {
        auto *value = std::getenv(name);
        return value == nullptr ? 0ull :
                                  luisa::hash_value(luisa::string_view{value});
    };
    return luisa::hash_combine({
        hash_env("LUISA_XIR_DISABLE_OPTIMIZATION"),
        hash_env("LUISA_XIR_ENABLE_SCALARIZER"),
        hash_env("LUISA_SPIRV_OPT_LEVEL"),
        hash_env("LUISA_SPIRV_OPT_PASSES"),
    });
}

[[nodiscard]] static vstd::MD5 compute_shader_cache_md5(
    Function kernel, const ShaderOption &option,
    lc::spirv::SpirvTargetFeatures target_features) noexcept {
    using namespace std::string_view_literals;
    auto option_flags =
        (static_cast<uint64_t>(option.enable_fast_math) << 0u) |
        (static_cast<uint64_t>(option.enable_debug_info) << 1u) |
        (static_cast<uint64_t>(option.enable_extended_accel_limits) << 2u) |
        (static_cast<uint64_t>(option.enable_scalarizer) << 3u);
    auto block_size = kernel.block_size();
    uint64_t data[] = {
        luisa::hash_value("luisa-vk-xir-spv-cache-v14"sv),
        kernel.hash(),
        kernel.body()->hash(),
        luisa::hash_value(block_size),
        static_cast<uint64_t>(block_size.x) |
            (static_cast<uint64_t>(block_size.y) << 21u) |
            (static_cast<uint64_t>(block_size.z) << 42u),
        option_flags,
        static_cast<uint64_t>(option.max_registers),
        luisa::hash_value(option.native_include),
        option.native_include.size(),
        target_features.enabled_mask(),
        static_cast<uint64_t>(
            target_features.buffer_float32_atomic_rmw_policy),
        xir_spirv_environment_hash(),
        static_cast<uint64_t>(kernel.allowed_warp_size().value_or(0u)),
    };
    return vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(data), sizeof(data)}};
}
#endif

ShaderCreationInfo Device::_create_shader_hlsl(
    const ShaderOption &option, Function kernel,
    bool requires_sampler_anisotropy) noexcept {
#ifdef LC_NO_HLSL_BUILTIN
    LUISA_ERROR(
        "This Vulkan shader requires the HLSL-to-SPIR-V fallback, but the "
        "backend was built without the bundled HLSL compiler.");
#else
    if (kernel.requires_raytracing() && !raytracing_enabled) {
        LUISA_ERROR(
            "Vulkan shader '{}' requires ray tracing, but ray-query support "
            "is not enabled on this device.",
            kernel.name());
    }
    ShaderCreationInfo info;
    uint mask = 0u;
    if (option.enable_fast_math) { mask |= 1u; }
    if (option.enable_debug_info) { mask |= 2u; }

    auto code = hlsl::CodegenUtility{}.Codegen(
        kernel, option.native_include, mask, true, false,
        option.enable_debug_info, option.enable_fast_math);
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(
                             code.result.data() + code.immutableHeaderSize),
                         code.result.size() - code.immutableHeaderSize});
    auto requires_async_copy =
        kernel.propagated_builtin_callables().test(CallOp::ASYNC_COPY) ||
        kernel.propagated_builtin_callables().test(CallOp::PIPELINE_COMMIT) ||
        kernel.propagated_builtin_callables().test(CallOp::PIPELINE_WAIT_PRIOR);
    if (requires_async_copy && !async_copy_enabled) {
        LUISA_WARNING(
            "ASYNC_COPY is used but VK_KHR_workgroup_memory_explicit_layout "
            "is unavailable. Async copy will use per-thread copy + barrier "
            "instead of hardware-accelerated OpGroupAsyncCopy.");
    }
    auto shader_model = [&]() noexcept -> uint {
        if (kernel.use_cooperative_operations()) { return kTensorShaderModel; }
        if (kernel.allowed_warp_size().has_value() || requires_async_copy) {
            return kHighShaderModel;
        }
        return kShaderModel;
    }();

    if (option.compile_only) {
        LUISA_ASSERT(!option.name.empty(),
                     "Vulkan compile-only shader requires a non-empty shader name.");
        info.invalidate();
        if (print_code()) {
            if (auto file = fopen("hlsl_output.hlsl", "ab")) {
                fwrite(code.result.view().data(), code.result.view().size(), 1u,
                       file);
                fclose(file);
            }
        }
        auto cache_result = ShaderSerializer::try_deser_compute(
            this,
            {.shader_md5 = check_md5,
             .type_md5 = code.typeMD5,
             .codegen_dialect =
                 detail::ShaderCodegenDialect::HLSL_SPIRV},
            hlsl::binding_to_arg(kernel.bound_arguments()),
            option.name,
            SerdeType::kByteCode,
            _binary_io,
            32u,
            option.enable_driver_optimization);
        if (cache_result.shader) {
            delete static_cast<ComputeShader *>(cache_result.shader);
            LUISA_VERBOSE("ComputeShader (HLSL compile-only) loaded from cache.");
            info.block_size = kernel.block_size();
            return info;
        }
        auto comp_result = Device::compiler()->compile_compute(
            code.result.view(),
            !option.enable_debug_info,
            shader_model,
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
                    {reinterpret_cast<const uint *>(buffer->GetBufferPointer()),
                     buffer->GetBufferSize() / sizeof(uint)},
                    SerdeType::kByteCode,
                    _binary_io,
                    code.useTex2DBindless,
                    code.useTex3DBindless,
                    code.useBufferBindless,
                    code.printers,
                    code.validation_count,
                    {},
                    kernel.allowed_warp_size(),
                    conservative_spirv_artifact_requirements(
                        this, requires_sampler_anisotropy));
            },
            [](auto &&error) {
                LUISA_ERROR("Compile Error: {}", error);
                return nullptr;
            });
    } else {
        vstd::string_view file_name;
        vstd::string cache_name;
        SerdeType serde_type{};
        if (option.enable_cache) {
            if (option.name.empty()) {
                cache_name << check_md5.to_string(false) << ".spv"sv;
                file_name = cache_name;
                serde_type = SerdeType::kCache;
            } else {
                file_name = option.name;
                serde_type = SerdeType::kByteCode;
            }
        }
        auto validation_count = code.validation_count;
        auto shader = ComputeShader::compile(
            _binary_io,
            this,
            ShaderSerializer::serialize_saved_args(kernel),
            [&]() { return std::move(code); },
            check_md5,
            code.typeMD5,
            hlsl::binding_to_arg(kernel.bound_arguments()),
            kernel.block_size(),
            file_name,
            serde_type,
            shader_model,
            option.enable_fast_math,
            validation_count,
            kernel.allowed_warp_size(),
            requires_sampler_anisotropy,
            32u,
            detail::ShaderCodegenDialect::HLSL_SPIRV,
            option.enable_driver_optimization);
        LUISA_VERBOSE(
            "ComputeShader (HLSL) created, pipeline: {}",
            reinterpret_cast<void *>(shader->pipeline()));
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pipeline();
    }
    info.block_size = kernel.block_size();
    return info;
#endif
}

// kernel
ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    ShaderCreationInfo info;

#ifdef LUISA_XIR_TO_SPIRV
    constexpr auto native_xir_spirv_compiled = true;
#else
    constexpr auto native_xir_spirv_compiled = false;
#endif
    auto require_native = require_native_xir_spirv();
    auto builtin_calls = kernel.propagated_builtin_callables();
    auto requires_motion_blur =
        builtin_calls.uses_raytracing_motion_blur();
    detail::UserComputeCodegenRequirements codegen_requirements{
        .native_include = !option.native_include.empty(),
        .printing = kernel.requires_printing(),
        .cooperative_operations = kernel.use_cooperative_operations(),
        .async_copy = builtin_calls.test(CallOp::ASYNC_COPY) ||
                      builtin_calls.test(CallOp::PIPELINE_COMMIT) ||
                      builtin_calls.test(CallOp::PIPELINE_WAIT_PRIOR),
        .motion_blur = requires_motion_blur};
    auto codegen_route =
        detail::plan_user_compute_codegen_route(codegen_requirements);
    auto native_requirement = detail::plan_required_native_xir_spirv(
        require_native, native_xir_spirv_compiled, codegen_route);
    LUISA_ASSERT(
        native_requirement.status !=
            detail::RequiredNativeXirSpirvStatus::NATIVE_CODEGEN_UNAVAILABLE,
        "Vulkan shader '{}' requires native XIR-to-SPIR-V codegen because "
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV is enabled, but this Vulkan "
        "backend was built without LUISA_XIR_TO_SPIRV.",
        kernel.name());

#ifdef LUISA_XIR_TO_SPIRV
    luisa::string fallback_reasons;
    if (codegen_route.requires_hlsl_fallback()) {
        fallback_reasons = describe_hlsl_fallback_reasons(codegen_route);
        LUISA_ASSERT(
            native_requirement.status != detail::RequiredNativeXirSpirvStatus::
                                             HLSL_FALLBACK_REQUIRED,
            "Vulkan shader '{}' cannot use the required native "
            "XIR-to-SPIR-V path because it requires the HLSL fallback for: "
            "{}. LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV is enabled.",
            kernel.name(), fallback_reasons);
    }
#endif

    // Capability errors are meaningful only after the requested codegen
    // route has been accepted. In strict mode the route-contract diagnostic
    // must take precedence over unrelated device limitations.
    auto requires_sampler_anisotropy =
        validate_sampler_anisotropy_requirement(
            kernel, option.native_include,
            sampler_anisotropy_enabled);
    if (requires_motion_blur && !motion_blur_enabled) {
        LUISA_ERROR("Vulkan device does not support VK_NV_ray_tracing_motion_blur; "
                    "motion-time compute tracing cannot be compiled.");
    }
#ifdef LUISA_XIR_TO_SPIRV
    if (codegen_route.requires_hlsl_fallback()) {
        LUISA_VERBOSE(
            "Vulkan shader '{}' requires the HLSL-to-SPIR-V fallback for: "
            "{}.",
            kernel.name(), fallback_reasons);
        return _create_shader_hlsl(
            option, kernel, requires_sampler_anisotropy);
    }
    auto enabled_spirv_features = enabled_spirv_artifact_features();
    auto target_features =
        lc::spirv::SpirvTargetFeatures::from_enabled_mask(
            enabled_spirv_features);
    auto float_atomic_policy = detail::plan_vulkan_float_atomic_codegen(
        _vk_device->properties.vendorID);
    if (float_atomic_policy
            .native_xir_spirv_prefers_software_buffer_float32_rmw) {
        target_features.buffer_float32_atomic_rmw_policy =
            lc::spirv::SpirvBufferFloat32AtomicRmwPolicy::PREFER_WORD_CAS;
    }
    LUISA_ASSERT(target_features.enabled_mask() == enabled_spirv_features,
                 "Vulkan SPIR-V target-feature mask did not round-trip.");
    vstd::optional<lc::spirv::SpirvResult> spv_result;
    auto profile =
        lc::detail::env_flag("LUISA_VULKAN_PROFILE_COMPILATION");
    auto shader_md5 = compute_shader_cache_md5(
        kernel, option, target_features);
    auto require_print_code = print_code();
    auto type_md5 = hlsl::CodegenUtility::GetTypeMD5(kernel);
    auto uses_user_path = !option.name.empty();
    auto serde_type = (option.compile_only || uses_user_path) ? SerdeType::kByteCode : SerdeType::kCache;
    auto use_binary_io = option.compile_only || uses_user_path || option.enable_cache;
    if (option.compile_only && option.name.empty()) [[unlikely]] {
        LUISA_ERROR("Vulkan compile-only shader requires a non-empty shader name.");
    }
    luisa::string shader_name = uses_user_path ? option.name : luisa::format("{}.spv", shader_md5.to_string(false));
    auto compile_native_spirv = [&] {
        Clock codegen_clock;
        auto result =
            lc::spirv::SpirvCodegenEntry::compile_spirv(
                kernel, option, target_features);
        if (profile) {
            LUISA_INFO(
                "Vulkan native AST-to-SPIR-V total: {:.3f} ms",
                codegen_clock.toc());
        }
        return result;
    };

    if (require_print_code || !use_binary_io ||
        ShaderSerializer::require_recompile(
            shader_name,
            {.shader_md5 = shader_md5,
             .type_md5 = type_md5,
             .codegen_dialect =
                 detail::ShaderCodegenDialect::XIR_SPIRV},
            serde_type, _binary_io)) {
        spv_result = compile_native_spirv();
    }

    if (!spv_result && !option.compile_only && use_binary_io) {
        auto deser = ShaderSerializer::try_deser_compute(
            this,
            {.shader_md5 = shader_md5,
             .type_md5 = type_md5,
             .codegen_dialect =
                 detail::ShaderCodegenDialect::XIR_SPIRV},
            hlsl::binding_to_arg(kernel.bound_arguments()),
            shader_name,
            serde_type,
            _binary_io,
            32u,
            option.enable_driver_optimization);
        if (deser.shader) {
            auto shader = static_cast<ComputeShader *>(deser.shader);
            LUISA_VERBOSE("ComputeShader loaded successfully, pipeline: {}", reinterpret_cast<void *>(shader->pipeline()));
            info.handle = reinterpret_cast<uint64_t>(shader);
            info.native_handle = shader->pipeline();
            info.block_size = shader->block_size();
            return info;
        }
        spv_result = compile_native_spirv();
    }

    if (spv_result) {
        auto artifact_requirements =
            validated_spirv_artifact_requirements(
                this, spv_result->required_target_features);
        for (size_t i = 0; i < spv_result->properties.size(); ++i) {
            auto &p = spv_result->properties[i];
            LUISA_VERBOSE("  prop[{}]: type={}, space={}, reg={}, array_size={}", i, (int)p.type, p.space_index, p.register_index, p.array_size);
        }
        if (require_print_code) [[unlikely]] {
            auto dump_name = [&]() -> luisa::string {
                if (!shader_name.empty()) return shader_name;
                if (!kernel.name().empty()) return luisa::string{kernel.name()};
                return luisa::format("{:x}", kernel.hash());
            }();
            auto filename = luisa::format("spv_code_{}.spvasm", dump_name);
            std::ofstream file(filename.c_str());
            if (file) {
                file << "; === KERNEL: " << kernel.name() << " hash=" << kernel.hash() << " ===\n";
                spv::Disassemble(file, spv_result->spv_bin);
            }
            LUISA_VERBOSE("SPIRV printed to {}.", filename);
        }
        if (use_binary_io) {
            Clock serialization_clock;
            ShaderSerializer::serialize_bytecode(
                spv_result->properties,
                ShaderSerializer::serialize_saved_args(
                    luisa::span{spv_result->argument_usages}, true,
                    luisa::span{spv_result->argument_roles}),
                shader_md5,
                type_md5,
                kernel.block_size(),
                shader_name,
                spv_result->spv_bin,
                serde_type,
                _binary_io,
                spv_result->useTex2DBindless,
                spv_result->useTex3DBindless,
                spv_result->useBufferBindless,
                spv_result->printers,
                0,
                spv_result->constant_ubo_data,
                kernel.allowed_warp_size(),
                artifact_requirements,
                detail::ShaderCodegenDialect::XIR_SPIRV);
            if (profile) {
                LUISA_INFO(
                    "Vulkan native shader-artifact serialization: {:.3f} ms",
                    serialization_clock.toc());
            }
        }
    }
    if (option.compile_only) {
        assert(!option.name.empty());
        info.invalidate();
    } else {
        LUISA_ASSERT(spv_result, "Vulkan SPIR-V cache load failed without recompilation.");
        Clock pipeline_clock;
        auto shader = new ComputeShader(
            this,
            kernel.block_size(),
            spv_result->properties,
            ShaderSerializer::serialize_saved_args(
                luisa::span{spv_result->argument_usages}, true,
                luisa::span{spv_result->argument_roles}),
            {reinterpret_cast<const uint *>(spv_result->spv_bin.data()), spv_result->spv_bin.size()},
            hlsl::binding_to_arg(kernel.bound_arguments()),
            {},
            spv_result->useTex2DBindless,
            spv_result->useTex3DBindless,
            spv_result->useBufferBindless,
            std::move(spv_result->printers),
            {spv_result->constant_ubo_data.data(), spv_result->constant_ubo_data.size()},
            0,
            kernel.allowed_warp_size(),
            32u,
            detail::ShaderCodegenDialect::XIR_SPIRV,
            option.enable_driver_optimization);
        if (profile) {
            LUISA_INFO(
                "Vulkan native ComputeShader construction: {:.3f} ms",
                pipeline_clock.toc());
        }
        if (use_binary_io) {
            Clock pso_serialization_clock;
            ShaderSerializer::serialize_pso(
                this, shader, shader_md5, _binary_io);
            if (profile) {
                LUISA_INFO(
                    "Vulkan native pipeline-cache serialization: {:.3f} ms",
                    pso_serialization_clock.toc());
            }
        }
        LUISA_VERBOSE("ComputeShader created successfully, pipeline: {}", reinterpret_cast<void *>(shader->pipeline()));
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pipeline();
    }

#elif defined(LUISA_AST_LLVM_TO_SPIRV)
    // === AST LLVM to SPIR-V codegen path ===
    if (kernel.requires_raytracing() && !raytracing_enabled) {
        LUISA_ERROR(
            "Vulkan shader '{}' requires ray tracing, but ray-query support "
            "is not enabled on this device.",
            kernel.name());
    }
    auto llvm_result = lc::llvm_codegen::compile_spirv(kernel, option);
    for (size_t i = 0; i < llvm_result.properties.size(); ++i) {
        auto &p = llvm_result.properties[i];
        LUISA_VERBOSE("  LLVM prop[{}]: type={}, space={}, reg={}, array_size={}",
                      i, (int)p.type, p.space_index, p.register_index, p.array_size);
    }
    if (print_code()) [[unlikely]] {
        auto dump_name = [&]() -> luisa::string {
            if (!option.name.empty()) return option.name;
            if (!kernel.name().empty()) return luisa::string{kernel.name()};
            return luisa::format("{:x}", kernel.hash());
        }();
        auto filename = luisa::format("spv_code_llvm_{}.spvasm", dump_name);
        std::ofstream file(filename.c_str());
        if (file) {
            file << "; === LLVM KERNEL: " << kernel.name()
                 << " hash=" << kernel.hash() << " ===\n";
            spv::Disassemble(file, std::vector<uint32_t>{llvm_result.spv_bin.begin(), llvm_result.spv_bin.end()});
        }
        LUISA_VERBOSE("SPIRV-LLVM printed to {}.", filename);
    }
    if (option.compile_only) {
        assert(!option.name.empty());
        info.invalidate();
        ShaderSerializer::serialize_bytecode(
            llvm_result.properties,
            ShaderSerializer::serialize_saved_args(kernel),
            vstd::MD5{vstd::span<const uint8_t>(
                reinterpret_cast<const uint8_t *>(llvm_result.spv_bin.data()),
                llvm_result.spv_bin.size() * sizeof(uint32_t))},
            hlsl::CodegenUtility::GetTypeMD5(kernel),
            kernel.block_size(),
            option.name,
            llvm_result.spv_bin,
            SerdeType::kByteCode,
            _binary_io,
            llvm_result.useTex2DBindless,
            llvm_result.useTex3DBindless,
            llvm_result.useBufferBindless,
            llvm_result.printers,
            0,
            {},// LLVM constants are embedded in the module; no constant UBO.
            kernel.allowed_warp_size(),
            conservative_spirv_artifact_requirements(
                this, requires_sampler_anisotropy),
            detail::ShaderCodegenDialect::LLVM_SPIRV);
    } else {
        auto shader = new ComputeShader(
            this,
            kernel.block_size(),
            llvm_result.properties,
            ShaderSerializer::serialize_saved_args(kernel),
            {reinterpret_cast<const uint *>(llvm_result.spv_bin.data()),
             llvm_result.spv_bin.size()},
            hlsl::binding_to_arg(kernel.bound_arguments()),
            {},
            llvm_result.useTex2DBindless,
            llvm_result.useTex3DBindless,
            llvm_result.useBufferBindless,
            std::move(llvm_result.printers),
            {},// LLVM constants are embedded in the module; no constant UBO.
            0,
            kernel.allowed_warp_size(),
            32u,
            detail::ShaderCodegenDialect::LLVM_SPIRV,
            option.enable_driver_optimization);
        LUISA_VERBOSE("ComputeShader (LLVM) created, pipeline: {}",
                      reinterpret_cast<void *>(shader->pipeline()));
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pipeline();
    }
#else
    return _create_shader_hlsl(
        option, kernel, requires_sampler_anisotropy);
#endif
    info.block_size = kernel.block_size();
    return info;
}
ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const luisa::compute::Type *const> arg_types) noexcept {
    ShaderCreationInfo info;
    luisa::optional<detail::ShaderCodegenDialect> required_dialect;
    if (require_native_xir_spirv()) {
        required_dialect = detail::ShaderCodegenDialect::XIR_SPIRV;
    }
    auto type_md5 = hlsl::CodegenUtility::GetTypeMD5(arg_types);
    auto deser_result = ShaderSerializer::try_deser_compute(
        this,
        {.type_md5 = type_md5,
         .codegen_dialect = required_dialect},
        {}, name,
        SerdeType::kByteCode, _binary_io);
    if (!deser_result.shader) {
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
    std::lock_guard dispatch_lock{detail::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::live_device_count == 0u,
        "Cannot enumerate Vulkan devices while a Vulkan Device is live; "
        "Volk uses process-global dispatch tables.");
    std::lock_guard lck{detail::instance_mtx};
    auto enable_validation = detail::validation_enabled_by_default();
    // Create the reusable enumeration instance with the supported surface
    // extensions as a superset, so a later graphics Device can reuse it.
    bool enable_surface{true};
    // Keep this process-owned instance alive for subsequent enumeration or
    // device creation, and always reload the instance dispatch table.
    detail::load_or_create_process_instance(
        enable_validation, enable_surface,
        {}, {}, {});
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
}

hlsl::ShaderCompiler *Device::compiler() {
#ifndef LC_NO_HLSL_BUILTIN
    std::lock_guard lck(g_dxc_mutex);
    if (!g_dxc_compiler_initialized) {
        if (g_dxc_runtime_directory.empty()) [[unlikely]] {
            LUISA_ERROR("Vulkan internal HLSL compiler requested before device initialization.");
        }
        g_dxc_compiler.create(g_dxc_runtime_directory, true);
        g_dxc_compiler_initialized = true;
    }
    return g_dxc_compiler.ptr();
#else
    LUISA_ERROR(
        "Vulkan DXC compatibility was disabled at build time. This path "
        "requires the legacy HLSL-to-SPIR-V compiler.");
    return nullptr;
#endif
}

VkInstance Device::instance() const noexcept {
    return _instance;
}
// HACK: for some app need external instance without device
LUISA_EXPORT_API VkInstance init_vk_instance(bool enable_validation, bool &enable_surface, const luisa::string *extra_instance_exts, size_t extra_instance_ext_count, const char *custom_vk_lib_path, const char *custom_vk_lib_name) {
    std::lock_guard dispatch_lock{detail::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::live_device_count == 0u,
        "Cannot initialize the process Vulkan instance while a Vulkan Device "
        "is live; Volk uses process-global dispatch tables.");
    std::lock_guard lck{detail::instance_mtx};
    enable_validation |= detail::validation_enabled_by_default();
    detail::load_or_create_process_instance(enable_validation, enable_surface, custom_vk_lib_path ? luisa::filesystem::path{custom_vk_lib_path} : luisa::filesystem::path{}, custom_vk_lib_name ? luisa::string_view{custom_vk_lib_name} : luisa::string_view{}, luisa::span{extra_instance_exts, extra_instance_ext_count});
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
        auto upper_bound = full_size;
        for (auto ptr : sub_allocations) {
            upper_bound = std::min(upper_bound, get_index(ptr));
        }
        LUISA_ASSERT(count < upper_bound,
                     "Vulkan bindless descriptor heap exhausted at {} slots.",
                     full_size);
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
    LUISA_ASSERT(
        _vk_device->enabled_features.sparseBinding == VK_TRUE,
        "Vulkan sparse heap allocation requires sparseBinding to be enabled "
        "on the logical device.");
    auto heap = vengine_new<VulkanSparseHeap>(this, byte_size);
    auto handle = reinterpret_cast<uint64_t>(heap);
    auto registration = _sparse_residency_registry.register_heap(handle);
    LUISA_ASSERT(
        static_cast<bool>(registration),
        "Failed to register Vulkan sparse heap 0x{:016x}: {}.",
        handle,
        detail::sparse_residency_registry_status_name(
            registration.status));
    return ResourceCreationInfo{
        .handle = handle,
        // Allocation is lazy and ResourceCreationInfo cannot update its
        // native handle later. Native sparse-heap interop is therefore
        // intentionally unsupported on Vulkan.
        .native_handle = nullptr};
}
void Device::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    auto unregistration =
        _sparse_residency_registry.unregister_heap(handle);
    LUISA_ASSERT(
        static_cast<bool>(unregistration),
        "Cannot destroy Vulkan sparse heap 0x{:016x}: {}. Explicitly unmap "
        "every resident range before destroying its heap.",
        handle,
        detail::sparse_residency_registry_status_name(
            unregistration.status));
    vengine_delete(reinterpret_cast<VulkanSparseHeap *>(handle));
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
    info.element_stride = (element == Type::of<void>()) ? 1 : element->size();
    LUISA_ASSERT(
        elem_count != 0u &&
            info.element_stride <=
                std::numeric_limits<size_t>::max() / elem_count,
        "Vulkan sparse buffer element count {} and stride {} produce an "
        "empty or overflowing byte size.",
        elem_count, info.element_stride);
    auto ptr = new SparseBuffer(
        this, info.element_stride * elem_count, true);
    info.handle = reinterpret_cast<uint64_t>(ptr);
    auto registration = _sparse_residency_registry.register_resource(
        info.handle,
        detail::SparseResidencyResourceKind::BUFFER);
    LUISA_ASSERT(
        static_cast<bool>(registration),
        "Failed to register Vulkan sparse buffer 0x{:016x}: {}.",
        info.handle,
        detail::sparse_residency_registry_status_name(
            registration.status));
    info.native_handle = ptr->vk_buffer();
    info.total_size_bytes = ptr->byte_size();
    info.tile_size_bytes = ptr->sparse_block_size();
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
    auto registration = _sparse_residency_registry.register_resource(
        r.handle,
        detail::SparseResidencyResourceKind::IMAGE);
    LUISA_ASSERT(
        static_cast<bool>(registration),
        "Failed to register Vulkan sparse image 0x{:016x}: {}.",
        r.handle,
        detail::sparse_residency_registry_status_name(
            registration.status));
    r.native_handle = ptr->vk_image();
    r.tile_size_bytes = ptr->sparse_block_size();
    r.tile_size = ptr->tile_size();
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
    auto unregistration =
        _sparse_residency_registry.unregister_resource(handle);
    LUISA_ASSERT(
        static_cast<bool>(unregistration),
        "Cannot destroy Vulkan sparse image 0x{:016x}: {}. Explicitly unmap "
        "every resident range before destroying the image.",
        handle,
        detail::sparse_residency_registry_status_name(
            unregistration.status));
    destroy_texture(handle);
}
void Device::destroy_sparse_buffer(uint64_t handle) noexcept {
    auto unregistration =
        _sparse_residency_registry.unregister_resource(handle);
    LUISA_ASSERT(
        static_cast<bool>(unregistration),
        "Cannot destroy Vulkan sparse buffer 0x{:016x}: {}. Explicitly unmap "
        "every resident range before destroying the buffer.",
        handle,
        detail::sparse_residency_registry_status_name(
            unregistration.status));
    destroy_buffer(handle);
}
void Device::set_stream_log_callback(uint64_t stream_handle,
                                     const StreamLogCallback &callback) noexcept {
    reinterpret_cast<Stream *>(stream_handle)->logger = callback;
}
}// namespace lc::vk

#include "../common/export_version.inl.h"
