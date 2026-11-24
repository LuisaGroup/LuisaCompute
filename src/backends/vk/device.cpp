#include "device.h"
#include "device_feature_plan.h"
#include "float_atomic_policy.h"
#include "sampler_anisotropy.h"
#include "user_compute_codegen_route.h"
#include "bfloat16_requirement.h"
#include "fallback_rtx_hlsl_codegen.h"
#include "../common/env_flag.h"
#include <luisa/ast/op.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include "log.h"
#include <luisa/vstl/config.h>
#include <luisa/core/binary_file_stream.h>
#include "compute_shader.h"
#include "hlsl_spirv_opt.h"
#include "native_shader_ext.h"
#include "vk_accel_archive_ext.h"
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
#include "submit_profile.h"
#include "texture.h"
#include "resource_barrier_contract.h"
#include "queue_family_contract.h"
#include "bindless_array.h"
#include "blas.h"
#include "opacity_micromap.h"
#include "cluster_blas.h"
#include "tlas.h"
#include "motion_instance.h"
#include "rt_shader.h"
#include "swapchain.h"
#include "sparse_buffer.h"
#include "sparse_heap.h"
#include "pinned_memory_ext.h"
#include "vk_raster_ext.h"
#include "vk_native_res_ext.h"
#include "dstorage_ext.h"
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
// `marl::Scheduler::get()` is how the shader compile pool captures its executor on
// the device-constructing thread (plan P5.1).
#include <marl/scheduler.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>

namespace lc::vk {
using namespace std::string_literals;
namespace {
namespace vk_device_detail {

[[nodiscard]] bool require_native_xir_spirv() noexcept {
    return luisa::compute::detail::env_flag(
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
        detail::UserComputeHlslFallbackReason::FALLBACK_RTX,
        detail::UserComputeHlslFallbackReason::ASYNC_COPY,
        detail::UserComputeHlslFallbackReason::MOTION_BLUR,
        detail::UserComputeHlslFallbackReason::ACCEL_SIZE,
        detail::UserComputeHlslFallbackReason::BFLOAT16};
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

}  // namespace vk_device_detail
}// namespace
#ifndef LC_NO_HLSL_BUILTIN
namespace vk_device_detail_2 {
static luisa::spin_mutex g_dxc_mutex;
static vstd::StackObject<hlsl::ShaderCompiler, false> g_dxc_compiler;
static luisa::filesystem::path g_dxc_runtime_directory;
static int32 g_dxc_ref_count = 0;
static bool g_dxc_compiler_initialized = false;
}  // namespace vk_device_detail_2
#endif

namespace detail {

[[nodiscard]] bool validation_enabled_by_default() noexcept {
#ifdef NDEBUG
    auto enabled = false;
#else
    auto enabled = true;
#endif
    if (luisa::compute::detail::env_flag(
            "LUISA_VULKAN_VALIDATION")) {
        enabled = true;
    }
    return enabled;
}

struct Settings {
    bool validation{false};
    bool fullscreen{false};
    bool vsync{false};
    bool overlay{true};
};

namespace vk_device_detail_3 {
static VkInstance vk_instance{nullptr};
static std::mutex instance_mtx;
static std::mutex dispatch_lifetime_mtx;
static uint32_t live_device_count{};
static bool vk_instance_surface_enabled{};
static bool vk_instance_validation_enabled{};
static vstd::unordered_set<luisa::string> vk_instance_extra_exts;
// Instance extensions the backend enabled itself (surface/platform +
// VK_KHR_get_surface_capabilities2 + VK_KHR_surface_maintenance1). Vulkan
// cannot report them back, so the optional-extension registry needs them handed
// over explicitly. Empty for a borrowed instance.
static vstd::vector<luisa::string> vk_instance_enabled_extensions;
static Settings settings{};
static PFN_vkCreateDebugUtilsMessengerEXT vk_create_debug_utils_messenger_ext;

}  // namespace vk_device_detail_3
// VK_KHR_shader_untyped_pointers was released in Vulkan 1.4.325. Validation
// layers built against older headers do not know the extension's device
// feature structure type and reject a vkCreateDevice pNext chain that
// contains it (VUID-VkDeviceCreateInfo-pNext-pNext); the backend's debug
// messenger escalates that into a fatal error and device creation fails.
// Only enable the extension when the active Khronos validation layer is new
// enough to recognize the structure type, or when validation is disabled.
[[nodiscard]] bool validation_layer_supports_shader_untyped_pointers() noexcept {
    if (!vk_device_detail_3::settings.validation) { return true; }
    // settings.validation is only true when VK_LAYER_KHRONOS_validation was
    // enabled at instance creation (see create_instance); query the API
    // version the layer was built against.
    uint32_t layer_count = 0u;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    vstd::vector<VkLayerProperties> layers;
    luisa::enlarge_by(layers, layer_count);
    if (layer_count != 0u) {
        vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    }
    constexpr auto untyped_pointers_validation_version =
        VK_MAKE_API_VERSION(0, 1, 4, 325);
    for (auto const &layer : layers) {
        if (luisa::string_view{layer.layerName} ==
            "VK_LAYER_KHRONOS_validation") {
            return layer.specVersion >= untyped_pointers_validation_version;
        }
    }
    // Fail closed: validation claims to be active but the Khronos layer's
    // version is unknown, so do not assume it understands the structure type.
    return false;
}
namespace vk_device_detail_3 {
static PFN_vkDestroyDebugUtilsMessengerEXT vk_destroy_debug_utils_messenger_ext;
static VkDebugUtilsMessengerEXT debug_utils_messenger;
static VkInstance debug_utils_messenger_instance;
}  // namespace vk_device_detail_3
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
namespace vk_device_detail_3 {
static AllocCallbacks alloc;
}  // namespace vk_device_detail_3
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

    if (vk_device_detail_3::debug_utils_messenger != VK_NULL_HANDLE) {
        LUISA_ASSERT(
            vk_device_detail_3::debug_utils_messenger_instance == instance,
            "Vulkan validation messenger is already attached to a different "
            "process instance.");
        return;
    }

    vk_device_detail_3::vk_create_debug_utils_messenger_ext = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    vk_device_detail_3::vk_destroy_debug_utils_messenger_ext = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    LUISA_ASSERT(
        vk_device_detail_3::vk_create_debug_utils_messenger_ext != nullptr &&
            vk_device_detail_3::vk_destroy_debug_utils_messenger_ext != nullptr,
        "VK_EXT_debug_utils was enabled, but its messenger entry points are unavailable.");

    VkDebugUtilsMessengerCreateInfoEXT debug_utils_messenger_ci{};
    debug_utils_messenger_ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_utils_messenger_ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_utils_messenger_ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debug_utils_messenger_ci.pfnUserCallback = debug_utils_messenger_callback;
    VK_CHECK_RESULT(vk_device_detail_3::vk_create_debug_utils_messenger_ext(
        instance, &debug_utils_messenger_ci,
        Device::alloc_callbacks(), &vk_device_detail_3::debug_utils_messenger));
    vk_device_detail_3::debug_utils_messenger_instance = instance;
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
        vk_device_detail_3::settings.validation = enable_validation;

        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "luisa_compute";
        app_info.pEngineName = app_info.pApplicationName;
        // The instance apiVersion advertises the highest core version the backend may
        // use (Vulkan 1.3 entry points: vkCmdPipelineBarrier2/vkCmdCopyBuffer2/...).
        // It must NOT be lowered to the device floor: loaders and validation layers
        // refuse to hand out core entry points above the instance's declared version,
        // which would leave the 1.3 function pointers null even on 1.3+ devices.
        // Vulkan 1.2 devices remain supported: the loader clamps the requested
        // version, and Device::sync2_capable()/copy2_capable() select the legacy
        // barrier/copy paths when the device lacks the 1.3 core entry points.
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
            // Optional surface capability queries used by
            // VK_KHR_surface_maintenance1 / VK_EXT_present_timing
            // (vkGetPhysicalDeviceSurfaceCapabilities2EXT,
            // vkGetPhysicalDeviceSurfacePresentModes2EXT). Harmless when unused.
            emplace_instance_ext(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            // Harmless to enable, and required for the
            // VK_KHR_surface_maintenance1 / VK_EXT_present_timing capability
            // queries (surface present modes / scaling).
            emplace_instance_ext(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
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
        if (vk_device_detail_3::settings.validation) {
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
                vk_device_detail_3::settings.validation = false;
            }
        }

#if defined(LUISA_PLATFORM_APPLE) && defined(VK_KHR_portability_enumeration)
        // MoltenVK requires portability enumeration to expose all conformant physical devices.
        if (supported_instance_exts.find(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != supported_instance_exts.end()) {
            emplace_instance_ext(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            instance_create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif
        if (vk_device_detail_3::settings.validation) {
            emplace_instance_ext(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);// SRS - Dependency when VK_EXT_DEBUG_MARKER is enabled
            if (!emplace_instance_ext(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                LUISA_WARNING(
                    "VK_EXT_debug_utils is unavailable; Vulkan validation "
                    "messenger output is disabled.");
                vk_device_detail_3::settings.validation = false;
            }
        }
        {
            auto &recorded = vk_device_detail_3::vk_instance_enabled_extensions;
            recorded.clear();
            for (auto name : instance_exts) {
                recorded.emplace_back(luisa::string{name});
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
    auto creating = vk_device_detail_3::vk_instance == VK_NULL_HANDLE;
    if (!creating) {
        enable_surface &= vk_device_detail_3::vk_instance_surface_enabled;
        vk_device_detail_3::settings.validation = vk_device_detail_3::vk_instance_validation_enabled;
        for (auto &&extension : extra_exts) {
            LUISA_ASSERT(
                vk_device_detail_3::vk_instance_extra_exts.contains(extension),
                "The process Vulkan instance is already live without "
                "requested extension '{}'. Instance extensions cannot be "
                "added after creation.",
                extension);
        }
    }
    create_instance(
        enable_validation, enable_surface, vk_device_detail_3::vk_instance,
        custom_path, lib_name, extra_exts);
    if (creating) {
        vk_device_detail_3::vk_instance_surface_enabled = enable_surface;
        vk_device_detail_3::vk_instance_validation_enabled = vk_device_detail_3::settings.validation;
        vk_device_detail_3::vk_instance_extra_exts.clear();
        for (auto &&extension : extra_exts) {
            vk_device_detail_3::vk_instance_extra_exts.emplace(extension);
        }
    }
}

}// namespace detail

Device::GlobalDispatchLease::GlobalDispatchLease() noexcept {
    std::lock_guard lock{detail::vk_device_detail_3::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::vk_device_detail_3::live_device_count == 0u,
        "The Vulkan backend currently supports only one live Device per "
        "process because Volk uses process-global instance/device dispatch "
        "tables. Destroy the existing Vulkan Device before creating another.");
    ++detail::vk_device_detail_3::live_device_count;
}

Device::GlobalDispatchLease::~GlobalDispatchLease() noexcept {
    std::lock_guard lock{detail::vk_device_detail_3::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::vk_device_detail_3::live_device_count == 1u,
        "Vulkan global dispatch lifetime accounting is unbalanced.");
    --detail::vk_device_detail_3::live_device_count;
}

VkAllocationCallbacks *Device::alloc_callbacks() {
    return &detail::vk_device_detail_3::alloc.callbacks;
}
//////////////// Not implemented area
lc::fallback_rtx::FallbackRtxDevice *Device::fallback_rtx() noexcept {
    if (!use_fallback_rtx_bit) { return nullptr; }
    std::lock_guard lock{_fallback_rtx_mutex};
    if (_fallback_rtx == nullptr) {
        _fallback_rtx = luisa::make_unique<lc::fallback_rtx::FallbackRtxDevice>(
            static_cast<DeviceInterface *>(this));
    }
    return _fallback_rtx.get();
}

ResourceCreationInfo Device::create_mesh(
    const AccelOption &option) noexcept {
    // The fallback hands out the handle of its own bottom-level structure, so
    // the handle of a fallback `Mesh` *is* the handle of its fallback BLAS and
    // `owns_fallback_blas()` later tells the build command where to go.
    if (auto *fallback = use_fallback_rtx_bit ? fallback_rtx() : nullptr) {
        auto blas = fallback->create_blas(option);
        return ResourceCreationInfo{
            .handle = blas,
            .native_handle = nullptr};
    }
    auto mesh = new Blas(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(mesh),
        .native_handle = nullptr};
}
void Device::destroy_mesh(uint64_t handle) noexcept {
    if (auto *fallback = use_fallback_rtx_bit ? fallback_rtx() : nullptr) {
        fallback->destroy_blas(handle);
        return;
    }
    delete reinterpret_cast<Blas *>(handle);
}

ResourceCreationInfo Device::create_opacity_micromap(
    const OpacityMicromapOption &option) noexcept {
    // Fail closed, with a message that names the switch to flip. Nothing is created
    // that a later build would silently ignore (D6).
    if (!OpacityMicromap::available(this)) {
        LUISA_ERROR(
            "Opacity micromaps are not available on this device: they need "
            "VK_KHR_opacity_micromap (or the VK_EXT_opacity_micromap alias), its "
            "`micromap` feature, buffer device addresses and "
            "enable_opacity_micromap() in the Vulkan config extension.");
    }
    if (use_fallback_rtx_bit) {
        LUISA_ERROR(
            "The software ray-tracing fallback has no opacity micromap path, so an "
            "opacity micromap cannot be built on this device.");
    }
    // A micromap is not a texture and cannot be a shader argument, so an attach is
    // only legal for one of the two spells: the EXT alias (no execution mode
    // needed), or the KHR spelling with `OpExecutionModeId OpacityMicromapIdKHR`
    // emitted by the SPIR-V codegen (D8).
    if (!OpacityMicromap::ext_alias_negotiated(this) &&
        !opacity_micromap_execution_mode_available()) {
        LUISA_ERROR(
            "VK_KHR_opacity_micromap requires every ray query to declare "
            "`OpExecutionModeId OpacityMicromapIdKHR`, and this device does not "
            "compile its ray-query modules with that execution mode, so a micromap "
            "would be ignored by the traversal.");
    }
    if (option.index_format == 1u) {
        LUISA_ERROR(
            "A one-byte per-triangle micromap index needs VK_KHR_index_type_uint8, "
            "which this backend does not enable; use 2 or 4 bytes, or no index "
            "buffer at all.");
    }
    if (option.data_buffer == 0u || option.data_size == 0u) {
        LUISA_ERROR(
            "An opacity micromap needs a non-empty packed bit buffer (got handle "
            "0x{:x}, {} bytes).",
            option.data_buffer, option.data_size);
    }
    auto *source = reinterpret_cast<Buffer const *>(option.data_buffer);
    if (!source->device_address_capable()) {
        LUISA_ERROR(
            "The packed bit buffer of an opacity micromap must be an ordinary "
            "device buffer of this device.");
    }
    auto format = option.format == 1u ?
                      detail::OpacityMicromapFormat::FOUR_STATE :
                      detail::OpacityMicromapFormat::TWO_STATE;
    auto request = OpacityMicromap::make_build_request(
        this, option.triangle_count, format,
        static_cast<uint8_t>(option.subdivision_level),
        /* public_api_request */ true);
    auto plan = detail::plan_opacity_micromap_build(request);
    if (!plan.build) {
        LUISA_ERROR(
            "An opacity micromap of {} triangles at subdivision level {} cannot be "
            "built on this device: {}.",
            option.triangle_count, option.subdivision_level,
            detail::opacity_micromap_build_status_name(plan.status));
    }
    auto micromap = OpacityMicromap::create_from_buffers(
        this, request, source->vk_buffer(), option.data_offset, option.data_size);
    if (micromap == nullptr) {
        LUISA_ERROR(
            "Failed to create an opacity micromap of {} triangles at subdivision "
            "level {}.",
            option.triangle_count, option.subdivision_level);
    }
    mark_optional_extension_active(detail::OptionalExtension::OPACITY_MICROMAP);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(micromap),
        .native_handle = micromap->handle()};
}

void Device::destroy_opacity_micromap(uint64_t handle) noexcept {
    // Unlike a texture the micromap holds no per-stream state, so there is no
    // `remove_resource_state` to mirror here: it is just the object.
    //
    // The object is created with `new_with_allocator` (see
    // `OpacityMicromap::create_sized`), so it has to be released with the
    // matching allocator. A plain `delete` would hand a block that luisa's
    // allocator owns to the CRT heap, which corrupts the heap and aborts the
    // process in the middle of a test.
    luisa::delete_with_allocator(reinterpret_cast<OpacityMicromap *>(handle));
}

uint32_t Device::opacity_micromap_max_subdivision_level(bool four_state) noexcept {
    if (!OpacityMicromap::available(this)) { return 0u; }
    auto request = OpacityMicromap::make_build_request(
        this, 1u,
        four_state ? detail::OpacityMicromapFormat::FOUR_STATE :
                     detail::OpacityMicromapFormat::TWO_STATE,
        0u, /* public_api_request */ true);
    return four_state ? request.max_four_state_subdivision_level :
                        request.max_two_state_subdivision_level;
}

ResourceCreationInfo Device::create_mega_geometry(
    const AccelOption &option) noexcept {
    // Fail loudly rather than hand out a handle whose every build would have to
    // be rejected: a cluster BLAS can only exist on a device that enabled
    // VK_NV_cluster_acceleration_structure.
    if (!cluster_acceleration_structure_enabled) [[unlikely]] {
        LUISA_ERROR(
            "Cluster (mega-geometry) acceleration structures are not enabled on "
            "this device: VK_NV_cluster_acceleration_structure, its "
            "clusterAccelerationStructure feature and buffer device addresses "
            "are all required, and the software ray-tracing fallback has no "
            "cluster path.");
    }
    auto blas = new ClusterBlas(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(blas),
        .native_handle = nullptr};
}
void Device::destroy_mega_geometry(uint64_t handle) noexcept {
    delete reinterpret_cast<ClusterBlas *>(handle);
}
uint64_t Device::mega_geometry_device_address(uint64_t handle) noexcept {
    if (handle == 0u) { return 0u; }
    return reinterpret_cast<ClusterBlas *>(handle)->cluster_blas_address();
}
uint64_t Device::buffer_device_address(uint64_t buffer_handle) noexcept {
    // 0 means "unavailable", never a fatal check: a device without buffer
    // device addresses cannot serve cluster build records at all, and the
    // public API documents 0 as the portable answer.
    if (!device_address_enabled || buffer_handle == 0u) { return 0u; }
    auto *buffer = reinterpret_cast<Buffer *>(buffer_handle);
    if (!buffer->device_address_capable()) { return 0u; }
    return buffer->get_device_address();
}

ResourceCreationInfo Device::create_procedural_primitive(
    const AccelOption &option) noexcept {
    if (use_fallback_rtx_bit) [[unlikely]] {
        LUISA_ERROR(
            "Procedural primitives are not supported by the Vulkan software "
            "ray-tracing fallback "
            "(VulkanDeviceConfigExt::use_fallback_rtx()).");
    }
    return create_mesh(option);
}

uint Device::compute_warp_size() const noexcept {
    if (prefers_default_wave32()) {
        return 32u;
    }
    VkPhysicalDeviceSubgroupProperties subgroup_properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_properties};
    vkGetPhysicalDeviceProperties2(physical_device(), &properties2);
    return subgroup_properties.subgroupSize;
}
size_t Device::compute_max_shared_memory_size() const noexcept {
    auto limit = static_cast<size_t>(properties().limits.maxComputeSharedMemorySize);
    if (optional_extension_enabled(detail::OptionalExtension::SHADER_SPLIT_BARRIER)) {
        // VK_EXT_shader_split_barrier reserves part of the workgroup shared
        // memory for its barriers; shaders may only use the remainder. This
        // infrequently-called query does a direct properties2 read instead of
        // caching another physical-device property block. The public query has
        // no per-shader usage input, so this is deliberately conservative for
        // kernels that do not use split barriers.
        VkPhysicalDeviceShaderSplitBarrierPropertiesEXT split_barrier_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SPLIT_BARRIER_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &split_barrier_properties};
        vkGetPhysicalDeviceProperties2(physical_device(), &properties2);
        auto reserved = static_cast<size_t>(
            split_barrier_properties.splitBarrierReservedSharedMemory);
        // Saturating subtraction: a driver reporting more reserved memory than
        // the total limit leaves no usable shared memory rather than wrapping.
        return reserved >= limit ? 0u : limit - reserved;
    }
    return limit;
}
uint64_t Device::memory_granularity() const noexcept {
    return kSparseBufferSize;
}

void Device::destroy_procedural_primitive(uint64_t handle) noexcept {
    destroy_mesh(handle);
}

ResourceCreationInfo Device::create_accel(const AccelOption &option) noexcept {
    if (auto *fallback = use_fallback_rtx_bit ? fallback_rtx() : nullptr) {
        auto accel = fallback->create_accel(option);
        return ResourceCreationInfo{
            .handle = accel,
            .native_handle = nullptr};
    }
    auto accel = new Tlas(this, option);
    return ResourceCreationInfo{
        .handle = reinterpret_cast<uint64_t>(accel),
        .native_handle = nullptr};
}
void Device::destroy_accel(uint64_t handle) noexcept {
    if (auto *fallback = use_fallback_rtx_bit ? fallback_rtx() : nullptr) {
        fallback->destroy_accel(handle);
        return;
    }
    delete reinterpret_cast<Tlas *>(handle);
}

ResourceCreationInfo Device::create_motion_instance(const AccelMotionOption &option) noexcept {
    if (use_fallback_rtx_bit) [[unlikely]] {
        LUISA_ERROR(
            "Motion instances are not supported by the Vulkan software "
            "ray-tracing fallback "
            "(VulkanDeviceConfigExt::use_fallback_rtx()).");
    }
    if (!motion_blur_enabled) [[unlikely]] {
        LUISA_ERROR("Motion instances require VK_NV_ray_tracing_motion_blur, "
                    "which is not enabled on this device.");
    }
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
    auto require_native_spirv = vk_device_detail::require_native_xir_spirv();
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
    {
        // In-flight depth (plan P1.5), latched here, once: every stream reads it
        // back through `max_frames_in_flight()`, so the bound cannot change under
        // a stream that already has batches in flight. Zero is rejected instead of
        // being read as "unlimited" - `DeviceConfig::inqueue_buffer_limit=false`
        // is the switch that disables host throttling.
        auto requested_frames_in_flight = _config_ext ? _config_ext->max_frames_in_flight() : detail::kDefaultFramesInFlight;
        auto frames_in_flight_plan =
            detail::plan_frames_in_flight(requested_frames_in_flight);
        if (!frames_in_flight_plan) [[unlikely]] {
            LUISA_ERROR(
                "VulkanDeviceConfigExt::max_frames_in_flight() returned {} ({}); "
                "zero is not 'unlimited'. Leave the getter at its default ({}), "
                "pick a depth of at least 1, or disable host throttling with "
                "DeviceConfig::inqueue_buffer_limit = false.",
                requested_frames_in_flight,
                detail::frames_in_flight_status_name(frames_in_flight_plan.status),
                detail::kDefaultFramesInFlight);
        }
        _max_frames_in_flight = frames_in_flight_plan.depth;
    }
    // SM-count reporting contract (plan P6.2), latched once, next to the
    // in-flight depth and for the same reason: it is a device attribute the
    // caller supplies, not a per-dispatch decision. Vulkan has no physical
    // SM-count query, so this backend never invents one - the value is either
    // the caller's externally calibrated estimate (reported together with
    // `sm_count_source = calibration_estimate`) or unknown (empty `sm_count`,
    // `sm_count_source = unknown`). An implausible input is rejected here
    // instead of being published as a device attribute.
    {
        auto requested_sm_count = _config_ext ? _config_ext->calibrated_sm_count_estimate() : 0u;
        auto sm_count_plan = detail::plan_calibrated_sm_count(requested_sm_count);
        if (sm_count_plan.status == detail::CalibratedSmCountStatus::REJECTED) [[unlikely]] {
            LUISA_WARNING(
                "VulkanDeviceConfigExt::calibrated_sm_count_estimate() returned "
                "{}, which is not a plausible SM count for one device (bound "
                "{}); reporting sm_count as unknown instead. This getter is a "
                "calibration input, not a device attribute.",
                requested_sm_count,
                detail::kMaxCalibratedSmCount);
        }
        _capability_status.sm_count = sm_count_plan.value;
        _capability_status.sm_count_source =
            sm_count_plan ?
                VulkanSmCountSource::CALIBRATION_ESTIMATE :
                VulkanSmCountSource::UNKNOWN;
    }
    // The config extension seeds the runtime command-reorder switch. It can
    // still be flipped later through CommandReorderExt, and
    // LUISA_DISABLE_COMMAND_REORDER=1 wins over both.
    _command_reorder.seed(_config_ext ? _config_ext->enable_command_reorder() : true);
    // Honor a compile-time minimum API version (xmake option
    // lc_vk_min_api_version) and an explicit config-extension override. The
    // runtime override wins over the compile-time define.
#ifdef LC_VK_MIN_API_VERSION
    _min_api_version = LC_VK_MIN_API_VERSION;
#endif
    if (_config_ext) {
        auto configured_min = _config_ext->min_api_version();
        if (configured_min != 0u) { _min_api_version = configured_min; }
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
        std::lock_guard lck(vk_device_detail_2::g_dxc_mutex);
        if (vk_device_detail_2::g_dxc_ref_count == 0) {
            vk_device_detail_2::g_dxc_runtime_directory = ctx.runtime_directory();
        }
        vk_device_detail_2::g_dxc_ref_count++;
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
        detail::vk_device_detail_3::settings.validation = capabilities.debug_utils;
        auto enable_validation = capabilities.debug_utils;
        auto enable_surface = capabilities.surface;
        detail::create_instance(
            enable_validation, enable_surface, _instance,
            custom_path, lib_name, {});
        surface_enabled = enable_surface;
    } else {
        std::lock_guard lck{detail::vk_device_detail_3::instance_mtx};
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
        _instance = detail::vk_device_detail_3::vk_instance;
    }
#ifndef LUISA_VULKAN_ENABLE_CUDA_INTEROP
    interop_enabled = false;
#endif
    _init_device(ext_phy_device, ext_device, device_idx);
    // One device-scoped VkPipelineCache for every DSL shader (plan P1.3). It is
    // created here, after `_init_device` produced the logical device and the
    // VkPipelineCacheHeaderVersionOne identity, and it is the only cache those
    // shaders use. Seeding reads at most one versioned artifact and fails closed
    // to an empty cache, so a bad blob can never cost the SPIR-V cache or abort
    // device creation.
    _pipeline_cache.create(this);

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
    _exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{NativeShaderExt::name},
#else
        NativeShaderExt::name,
#endif
        [](Device *device) -> DeviceExtension * {
            return new VkNativeShaderExt(device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<VkNativeShaderExt *>(ext);
        });
    _exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{AccelArchiveExt::name},
#else
        AccelArchiveExt::name,
#endif
        [](Device *device) -> DeviceExtension * {
            return new VkAccelArchiveExt(device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<VkAccelArchiveExt *>(ext);
        });
    // Direct storage: the Vulkan backend provides a host-staged fallback
    // (see src/backends/vk/dstorage_ext.{h,cpp}).
    _exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{DStorageExt::name},
#else
        DStorageExt::name,
#endif
        [](Device *device) -> DeviceExtension * {
            return new VkDStorageExt(device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<VkDStorageExt *>(ext);
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

    // Ordered batch compilation (plan P5.1): the *executor* of the shader compile pool
    // is captured here, once, on this thread, because `marl::Scheduler::get()` answers
    // "which scheduler is bound to *my* thread". Selecting it later - at the first
    // batch, or at destruction - would answer that question on a different thread and
    // silently pick the wrong executor. The pool object itself is lightweight (it owns
    // no thread, no lane and no Vulkan object until a batch needs them), so an eager
    // construction keeps the capture honest without changing what
    // `stats.pool_created` means: that still says "a batch actually used lanes".
    //
    // A captured scheduler is *external*: it must outlive this device and every batch,
    // because the pool enqueues each batch's lane tasks on it directly.
    _shader_compile_pool =
        luisa::make_unique<detail::ShaderCompilePool>(this, marl::Scheduler::get());
}

void Device::_init_device(VkPhysicalDevice external_physical_device, VkDevice external_device, uint32_t selected_device) {
    VkPhysicalDevice physical_device = external_physical_device;
    if (!physical_device) {
        VkResult err;

        // If requested, we enable the default validation layers for debugging
        if (detail::vk_device_detail_3::settings.validation) {
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
        // Defaults to the first usable device unless specified by command line
        VkPhysicalDeviceProperties device_properties;
        if (selected_device == -1) {
            // Unified scoring for all platforms: prefer graphics+compute
            // devices with bindless capacity, real GPUs over
            // virtual/CPU devices (avoids picking SwiftShader on Android
            // emulators when hardware exists), then higher API version.
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
                VkPhysicalDeviceProperties candidate_properties{};
                vkGetPhysicalDeviceProperties(
                    physical_devices[i], &candidate_properties);
                auto candidate = detail::DefaultDeviceCandidate{
                    .supports_graphics_compute = supports_graphics_compute,
                    .bindless_heap_capacity =
                        bindless_enabled ?
                            vk_device_detail::query_bindless_heap_capacity(physical_devices[i]) :
                            0u,
                    .device_type = candidate_properties.deviceType,
                    .api_version = candidate_properties.apiVersion};
                if (detail::prefer_default_device_candidate(
                        candidate, selected_candidate, bindless_enabled)) {
                    selected_device = i;
                    selected_candidate = candidate;
                }
            }
            vkGetPhysicalDeviceProperties(
                physical_devices[selected_device], &device_properties);
            LUISA_INFO(
                "Select device: {} (device ID: {:#010x}, type: {}, API "
                "{}.{}.{}, bindless capacity: {})",
                device_properties.deviceName, device_properties.deviceID,
                vks::tools::physical_device_type_string(
                    device_properties.deviceType),
                VK_API_VERSION_MAJOR(device_properties.apiVersion),
                VK_API_VERSION_MINOR(device_properties.apiVersion),
                VK_API_VERSION_PATCH(device_properties.apiVersion),
                selected_candidate.bindless_heap_capacity);
        }
        physical_device = physical_devices[std::min<uint32_t>(selected_device, physical_devices.size() - 1)];
    }

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    auto supported_ext = detail::supported_exts(physical_device);
    VkPhysicalDeviceFeatures device_features{};
    vkGetPhysicalDeviceFeatures(physical_device, &device_features);
    auto storage_image_format_features =
        detail::plan_storage_image_format_features({.read_without_format =
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
    // ---- optional extension negotiation ------------------------------
    // Everything negotiated here is optional: the backend keeps a fallback
    // path, so an unsupported or switched-off extension only costs the
    // optimization it enables. VulkanDeviceConfigExt decides per extension
    // what the application permits, the physical device decides what is
    // possible at all.
    {
        luisa::vector<luisa::string> supported_extension_names;
        supported_extension_names.reserve(supported_ext.size());
        for (auto const &name : supported_ext) {
            supported_extension_names.emplace_back(name.c_str());
        }
        luisa::vector<luisa::string> instance_extension_names;
        for (auto const &name : detail::vk_device_detail_3::vk_instance_enabled_extensions) {
            instance_extension_names.emplace_back(name.c_str());
        }
        // The physical device's API version lets the registry accept an
        // extension that Vulkan has promoted to core even when the driver no
        // longer lists its name (plan P2.2, VK_KHR_descriptor_update_template).
        _optional_extensions.select(
            luisa::span<const luisa::string>{supported_extension_names.data(),
                                             supported_extension_names.size()},
            luisa::span<const luisa::string>{instance_extension_names.data(),
                                             instance_extension_names.size()},
            _config_ext.get(), external_device != VK_NULL_HANDLE,
            _vk_device->properties.apiVersion);
        // One query over every candidate's feature structure. Only extensions
        // the physical device exposes are chained, so an unknown sType is
        // never handed to the driver.
        if (auto probe_chain = _optional_extensions.feature_query_chain()) {
            VkPhysicalDeviceFeatures2 optional_features2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = probe_chain};
            vkGetPhysicalDeviceFeatures2(physical_device, &optional_features2);
        }
    }
    auto api_version = _vk_device->properties.apiVersion;
    auto has_core_timeline_semaphore = api_version >= VK_API_VERSION_1_2;
    auto has_core_synchronization2 = api_version >= VK_API_VERSION_1_3;
    auto has_core_copy_commands2 = api_version >= VK_API_VERSION_1_3;
    auto has_physical_device_api_1_3 =
        api_version >= VK_API_VERSION_1_3;
    auto sync2_extension_available =
        supports_device_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    auto copy2_extension_available =
        supports_device_extension(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
    // When a 1.2 floor is requested (explicitly via config, or because the
    // device only exposes 1.2), the backend must use the classic
    // vkCmdPipelineBarrier/vkCmdCopyBuffer paths instead of the 1.3 core
    // entry points. Capability *support* (used for device acceptance) is
    // evaluated independently of this path choice.
    auto force_legacy_sync_copy =
        _min_api_version != 0u && _min_api_version <= VK_API_VERSION_1_2;
    VkPhysicalDeviceSynchronization2Features supported_synchronization2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .pNext = nullptr};
    if (has_core_synchronization2 || sync2_extension_available) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_synchronization2};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    }
    sync2_capable_bit =
        !force_legacy_sync_copy &&
        (has_core_synchronization2 ||
         (sync2_extension_available &&
          supported_synchronization2.synchronization2 == VK_TRUE));
    copy2_capable_bit =
        !force_legacy_sync_copy &&
        (has_core_copy_commands2 || copy2_extension_available);
    if (force_legacy_sync_copy) {
        LUISA_INFO(
            "Vulkan minimum API version {}.{} requested: using legacy "
            "barrier/copy paths (sync2={}, copy2={}).",
            VK_API_VERSION_MAJOR(_min_api_version),
            VK_API_VERSION_MINOR(_min_api_version),
            static_cast<bool>(sync2_capable_bit),
            static_cast<bool>(copy2_capable_bit));
    }
    auto required_device_features = detail::plan_required_device_features(
        {.timeline_semaphore_core = has_core_timeline_semaphore,
         .timeline_semaphore_extension =
             supports_device_extension(
                 VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME),
         .timeline_semaphore_feature =
             _vk_device->features_12.timelineSemaphore == VK_TRUE,
         .synchronization2_core = has_core_synchronization2,
         .synchronization2_extension = sync2_extension_available,
         .synchronization2_feature =
             supported_synchronization2.synchronization2 == VK_TRUE,
         .copy_commands2_core = has_core_copy_commands2,
         .copy_commands2_extension = copy2_extension_available,
         .physical_device_api_1_3 = has_physical_device_api_1_3});
    if (!required_device_features.supported) [[unlikely]] {
        LUISA_ERROR(
            "The Vulkan backend requires timelineSemaphore and "
            "synchronization2 (core Vulkan 1.2/1.3 or their extensions); "
            "physical device API {}.{}.{} reports timelineSemaphore={} "
            "(core={}, extension={}) and synchronization2={} "
            "(core={}, extension={}).",
            VK_API_VERSION_MAJOR(api_version),
            VK_API_VERSION_MINOR(api_version),
            VK_API_VERSION_PATCH(api_version),
            _vk_device->features_12.timelineSemaphore == VK_TRUE,
            has_core_timeline_semaphore,
            supports_device_extension(
                VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME),
            supported_synchronization2.synchronization2 == VK_TRUE,
            has_core_synchronization2,
            sync2_extension_available);
    }
    {
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &_timeline_semaphore_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
        // Per the Vulkan spec, maxTimelineSemaphoreValueDifference == 0 means
        // "no limit" (not an error). Log it for diagnostics only.
        if (_timeline_semaphore_properties
                .maxTimelineSemaphoreValueDifference == 0u) {
            LUISA_INFO(
                "Vulkan device reports maxTimelineSemaphoreValueDifference=0; "
                "treated as no limit per the Vulkan specification.");
        }
        // Every batch a stream keeps in flight spends one timeline value, so
        // the configured in-flight depth has to fit the device's timeline value
        // window (plan P1.5). A depth that cannot fit would make signal planning
        // reject a legitimate submission later; reject the configuration here
        // instead, while it is still a create-time error. An imported logical
        // device leaves the property at its zeroed default, which this class
        // reports as "no limit", so the check cannot fire there.
        if (static_cast<uint64_t>(_max_frames_in_flight) + 1u >
            max_timeline_semaphore_value_difference()) [[unlikely]] {
            LUISA_ERROR(
                "VulkanDeviceConfigExt::max_frames_in_flight() = {} needs {} "
                "pending timeline values, but this device reports "
                "maxTimelineSemaphoreValueDifference = {}.",
                _max_frames_in_flight,
                static_cast<uint64_t>(_max_frames_in_flight) + 1u,
                max_timeline_semaphore_value_difference());
        }
    }
    // ---- plan P6.1: the 14 core limits -----------------------------------
    // Captured once here from this physical device's own property structures,
    // and read back by both `Device::query` and
    // `VulkanExtensionStatusExt::capability_status()` - one record, so the two
    // reports cannot drift.
    //
    // This block is reporting only. Subgroup-size-control *policy* (whether the
    // backend asks the driver for a specific subgroup size) keeps reading
    // `_subgroup_size_control_properties`, which stays empty unless the feature is
    // really enabled below, so capturing a physical bound here cannot change the
    // warp size any shader is compiled with.
    {
        auto has_subgroup_size_control =
            api_version >= VK_API_VERSION_1_3 ||
            supports_device_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        VkPhysicalDeviceSubgroupProperties subgroup_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        // VK_EXT_subgroup_size_control is core Vulkan 1.3; below that, chaining
        // its properties structure is only legal once the physical device has
        // attested the extension. The base subgroup structure carries
        // subgroupSize and supportedOperations only - it has no lower/upper
        // subgroup-size bound at all, so on a Vulkan 1.2 device without the
        // extension `min_subgroup_size`/`max_subgroup_size` stay unknown instead
        // of being derived from subgroupSize.
        if (has_subgroup_size_control) {
            subgroup_properties.pNext = &subgroup_size_control_properties;
        }
        VkPhysicalDeviceProperties2 core_limit_properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &subgroup_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &core_limit_properties2);
        // A driver that ignores the chain leaves the structures zeroed; a zero
        // bound is reported as "unknown", never as a value.
        auto subgroup_size_control_query =
            has_subgroup_size_control &&
                    subgroup_size_control_properties.minSubgroupSize != 0u &&
                    subgroup_size_control_properties.maxSubgroupSize != 0u ?
                &subgroup_size_control_properties :
                nullptr;
        auto subgroup_properties_query =
            subgroup_properties.subgroupSize != 0u ? &subgroup_properties : nullptr;
        auto capability_limits = detail::make_capability_limits(
            _vk_device->properties.limits,
            subgroup_properties_query,
            subgroup_size_control_query);
        // Only the limit fields are replaced: the SM-count fields were latched in
        // the constructor from the calibration input (plan P6.2), and the
        // descriptor-heap state is read from the optional-extension registry on
        // demand (plan P6.3).
        capability_limits.sm_count = _capability_status.sm_count;
        capability_limits.sm_count_source = _capability_status.sm_count_source;
        _capability_status = capability_limits;
        LUISA_INFO(
            "Vulkan core limits: maxMemoryAllocationCount={}, subgroupOps={:#x}, "
            "subgroupSizeControl={} ({}-{}), maxComputeWorkgroupCount=[{},{},{}], "
            "maxComputeWorkgroupSize=[{},{},{}], maxComputeSharedMemoryBytes={}, "
            "nonCoherentAtomSize={}, minStorageBufferOffsetAlignment={}, "
            "maxStorageBufferRange={}",
            _capability_status.max_memory_allocation_count,
            _capability_status.subgroup_ops,
            _capability_status.subgroup_size_control_available,
            _capability_status.min_subgroup_size,
            _capability_status.max_subgroup_size,
            _capability_status.max_compute_workgroup_count_x,
            _capability_status.max_compute_workgroup_count_y,
            _capability_status.max_compute_workgroup_count_z,
            _capability_status.max_compute_workgroup_size_x,
            _capability_status.max_compute_workgroup_size_y,
            _capability_status.max_compute_workgroup_size_z,
            _capability_status.max_compute_shared_memory_bytes,
            _capability_status.non_coherent_atom_size,
            _capability_status.min_storage_buffer_offset_alignment,
            _capability_status.max_storage_buffer_range);
    }
    if (required_device_features.enable_timeline_semaphore_extension) {
        enable_device_extension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    }
    if (required_device_features.enable_synchronization2_extension) {
        enable_device_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    }
    if (required_device_features.enable_copy_commands2_extension) {
        enable_device_extension(VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME);
    }
    detail::NarrowNumericFeaturePlan narrow_numeric_features{};
    bool enable_float8 = false;
    bool enable_bfloat16 = false;
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
    if (supported_ext.find(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) !=
        supported_ext.end()) {
        VkPhysicalDeviceShaderBfloat16FeaturesKHR supported_bfloat16{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_bfloat16};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (supported_bfloat16.shaderBFloat16Type == VK_TRUE) {
            enable_device_extension(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
            enable_bfloat16 = true;
            LUISA_INFO("VK_KHR_shader_bfloat16 type support enabled on device.");
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
    auto device_supports_untyped_pointers =
        supported_ext.find(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME) !=
        supported_ext.end();
    if (device_supports_untyped_pointers) {
        if (detail::validation_layer_supports_shader_untyped_pointers()) {
            VkPhysicalDeviceShaderUntypedPointersFeaturesKHR
                supported_untyped_pointers{
                    .sType =
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
                    .pNext = nullptr};
            VkPhysicalDeviceFeatures2 features2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &supported_untyped_pointers};
            vkGetPhysicalDeviceFeatures2(physical_device, &features2);
            if (supported_untyped_pointers.shaderUntypedPointers == VK_TRUE) {
                enable_device_extension(
                    VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
                shader_untyped_pointers_enabled = true;
                LUISA_INFO(
                    "VK_KHR_shader_untyped_pointers enabled on device.");
            }
        } else {
            LUISA_WARNING(
                "VK_KHR_shader_untyped_pointers is supported by the device "
                "but the active validation layer (Vulkan < 1.4.325) predates "
                "the extension and rejects its feature structure type; "
                "leaving the extension disabled.");
        }
    }
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
#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
    // VK_NV_cuda_kernel_launch can only be requested on logical devices we
    // create ourselves; an imported device cannot be re-created with the
    // extension enabled, and its enabled features are not queryable.
    if (external_device == VK_NULL_HANDLE &&
        supports_device_extension(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME)) {
        VkPhysicalDeviceCudaKernelLaunchFeaturesNV supported_cuda_kernel_launch{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_cuda_kernel_launch};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        if (supported_cuda_kernel_launch.cudaKernelLaunchFeatures == VK_TRUE) {
            enable_device_extension(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME);
            cuda_kernel_launch_enabled = true;
            LUISA_INFO("VK_NV_cuda_kernel_launch enabled on device.");
        }
    }
#endif
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
    // Probe VK_EXT_shader_replicated_composites on its own. The extension lets
    // SPIR-V lower an exact splat composite to one replicated instruction, which
    // is a pure instruction-size optimization for every shader and has an
    // ordinary-composite fallback. It is therefore negotiated independently of
    // VK_NV_cooperative_vector: the cooperative-vector path below *requires* the
    // feature as a prerequisite, it does not own the negotiation.
    bool enable_replicated_composites = false;
    VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicated_composites_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT,
        .pNext = nullptr,
        .shaderReplicatedComposites = VK_FALSE};
    if (supported_ext.find(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME) != supported_ext.end()) {
        VkPhysicalDeviceFeatures2 rc_features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &replicated_composites_features};
        vkGetPhysicalDeviceFeatures2(physical_device, &rc_features2);
        enable_replicated_composites =
            replicated_composites_features.shaderReplicatedComposites == VK_TRUE;
        if (enable_replicated_composites) {
            enable_device_extension(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
        }
    }
    shader_replicated_composites_enabled = enable_replicated_composites;
    // Probe for cooperative-vector support (VK_NV_cooperative_vector).
    // SPV_NV_cooperative_vector requires the Vulkan memory model, and splat
    // construction goes through SPV_EXT_replicated_composites, so the
    // extension is only usable when both prerequisites are supported.
    bool enable_vulkan_memory_model = false;
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
            VkPhysicalDeviceVulkan12Features vk12_vmm_features{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 vmm_features2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                .pNext = &vk12_vmm_features};
            vkGetPhysicalDeviceFeatures2(physical_device, &vmm_features2);
            enable_vulkan_memory_model = vk12_vmm_features.vulkanMemoryModel == VK_TRUE &&
                                         vk12_vmm_features.vulkanMemoryModelDeviceScope == VK_TRUE;
            // The replicated-composites prerequisite is the standalone probe
            // above; do not re-probe or re-enable the extension here.
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
                  // Reported verbatim through cooperative_status(): the corruption
                  // guard below hides the packed/quantized entries from the log, so
                  // the count is the only evidence of how many the driver really
                  // enumerated.
                  cooperative_vector_config_count = actual_count;
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
          cooperative_vector_enabled = enable_vulkan_memory_model && enable_replicated_composites;
          // Recorded for VulkanExtensionStatusExt::cooperative_status(): the
          // feature bit and compute-stage support already passed above, and the
          // two SPIR-V prerequisites are exactly the values just probed.
          cooperative_vector_physical_support = true;
          cooperative_vector_memory_model = enable_vulkan_memory_model;
          cooperative_vector_replicated_composites = enable_replicated_composites;
          if (cooperative_vector_enabled) {
              enable_device_extension(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
              enable_device_extension(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);
              LUISA_INFO("VK_NV_cooperative_vector extension enabled on device.");
          } else {
              LUISA_INFO("VK_NV_cooperative_vector disabled: vulkanMemoryModel or shaderReplicatedComposites is unavailable.");
          }
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
#endif// ENABLE_HIDDEN_FEATURES
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
            // Honor a config-extension override of the requested bindless heap
            // capacity (mobile builds typically lower it).
            auto requested_capacity =
                detail::requested_bindless_heap_capacity;
            if (_config_ext) {
                auto configured =
                    _config_ext->requested_bindless_heap_capacity();
                if (configured != 0u) { requested_capacity = configured; }
            }
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
                                                         _descriptor_indexing_properties.maxUpdateAfterBindDescriptorsInAllPools},
                                                    requested_capacity);
            supported = _bindless_heap_capacity != 0u;
            if (_bindless_heap_capacity < requested_capacity) {
                LUISA_INFO(
                    "Vulkan bindless heap capacity clamped from {} to {} by "
                    "descriptor-indexing limits.",
                    requested_capacity,
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
    // ---- cluster (mega-geometry) acceleration structures ------------------
    // VK_NV_cluster_acceleration_structure depends only on
    // VK_KHR_acceleration_structure, but building a traceable Cluster BLAS also
    // needs buffer device addresses for the raw CLAS/BLAS/destination regions.
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV supported_cluster_as{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
        .pNext = nullptr};
    auto has_cluster_as_extensions =
        has_ray_query_extensions &&
        supports_device_extension(
            VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    if (has_cluster_as_extensions) {
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_cluster_as};
        vkGetPhysicalDeviceFeatures2(physical_device, &features2);
        _cluster_acceleration_structure_properties =
            VkPhysicalDeviceClusterAccelerationStructurePropertiesNV{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
                .pNext = nullptr};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &_cluster_acceleration_structure_properties};
        vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    }
    auto raytracing_requested = raytracing_enabled;
    // ---- software (fallback) ray tracing ---------------------------------
    // The effective decision is made exactly once, here, and every later
    // branch reads it instead of re-deciding: the user may force the fallback
    // through the config extension, and a physical device that cannot provide
    // the hardware ray-query path (VK_KHR_acceleration_structure plus the
    // ray-query extension and the feature bits the backend requires, all
    // folded into `plan_optional_device_features`) has no other way to trace.
    // `hardware_raytracing_available()` asks that plan with the request forced
    // on, so this is the same capability the hardware path below is enabled
    // from. An imported logical device keeps its established behaviour: the
    // physical device is not asked for the fallback on its behalf, and the
    // backend does not request the ray-tracing features it cannot attest.
    auto feature_support = detail::OptionalDeviceFeatureSupport{
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
            supported_motion_blur.rayTracingMotionBlur == VK_TRUE};
    auto hardware_raytracing = detail::hardware_raytracing_available(
        feature_support);
    use_fallback_rtx_bit =
        (_config_ext != nullptr && _config_ext->use_fallback_rtx()) ||
        // `LUISA_VK_FALLBACK_RTX=1` forces the fallback for tests and
        // command-line programs that cannot hand a `VulkanDeviceConfigExt` to
        // `Context::create_device`; the supported surface stays
        // `DeviceConfigExt::use_fallback_rtx()`.
        luisa::compute::detail::env_flag("LUISA_VK_FALLBACK_RTX") ||
        !hardware_raytracing;
    auto motion_blur_requested =
        raytracing_requested && !use_fallback_rtx_bit &&
        (!_config_ext || _config_ext->enable_motion_blur());
    auto optional_device_features = detail::plan_optional_device_features(
        feature_support,
        // The single place the hardware path is enabled: requested by the user
        // (or by the platform default), supported by the physical device and
        // not replaced by the fallback.
        {.ray_query = raytracing_requested && !use_fallback_rtx_bit,
         .ray_tracing_motion_blur = motion_blur_requested});
    enable_barycentric =
        optional_device_features.fragment_shader_barycentric;
    raytracing_enabled = optional_device_features.ray_query;
    enable_motion_blur =
        optional_device_features.ray_tracing_motion_blur;
    motion_blur_enabled = enable_motion_blur;
    if (use_fallback_rtx_bit) {
        LUISA_INFO(
            "Vulkan ray tracing uses the software fallback "
            "(VulkanDeviceConfigExt::use_fallback_rtx or a physical device "
            "without the hardware ray-tracing path); the native acceleration-"
            "structure extensions are not enabled.");
    }
    if (raytracing_requested && !raytracing_enabled && !use_fallback_rtx_bit) {
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
        enable_bfloat16 = false;
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
          // Physical support and the two SPIR-V prerequisites are physical-device
          // facts, but an imported VkDevice's enabled feature chain is
          // unattestable, so every cooperative answer is cleared here (an
          // imported device must not claim a capability it cannot prove) and the
          // configuration count goes with them.
          cooperative_vector_physical_support = false;
          cooperative_vector_memory_model = false;
          cooperative_vector_replicated_composites = false;
          shader_replicated_composites_enabled = false;
          cooperative_vector_config_count = 0u;
          shader_untyped_pointers_enabled = false;
#if ENABLE_HIDDEN_FEATURES
          async_copy_enabled = false;
#endif
          interop_enabled = false;
          device_address_enabled = false;
          surface_enabled = false;
          _enable_device_exts.clear();
      }
      // Record which cooperative-matrix twin was enabled (if any), for
      // VulkanExtensionStatusExt::cooperative_status(). Done after the
      // imported-device reset above so an imported VkDevice reports none: its
      // enabled feature chain cannot be attested. "Enabled" is all this says --
      // nothing in this tree lowers to a cooperative-matrix instruction yet.
      switch (enabled_cooperative_matrix_ext) {
          case CooperativeMatrixExt::None:
              cooperative_matrix_extension = nullptr;
              cooperative_matrix_workgroup_scope = false;
              break;
          case CooperativeMatrixExt::KHR:
              cooperative_matrix_extension = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
              cooperative_matrix_workgroup_scope = false;
              break;
          case CooperativeMatrixExt::KHRAndNV2:
              cooperative_matrix_extension = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
              cooperative_matrix_workgroup_scope = true;
              break;
          case CooperativeMatrixExt::NV:
              cooperative_matrix_extension = VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME;
              cooperative_matrix_workgroup_scope = false;
              break;
    }
    // Cluster (mega-geometry) acceleration structures: decided here, next to
    // the ray-query decision they depend on (the extension's only dependency is
    // VK_KHR_acceleration_structure, which the ray-query path already enables)
    // and forced off above for an imported logical device, whose enabled
    // feature chain Vulkan cannot attest after creation.
    cluster_acceleration_structure_enabled =
        raytracing_enabled && device_address_enabled &&
        !use_fallback_rtx_bit && has_cluster_as_extensions &&
        supported_cluster_as.clusterAccelerationStructure == VK_TRUE;
    if (has_cluster_as_extensions && !cluster_acceleration_structure_enabled &&
        raytracing_enabled) {
        auto cluster_feature =
            supported_cluster_as.clusterAccelerationStructure == VK_TRUE;
        auto device_address = static_cast<bool>(device_address_enabled);
        LUISA_INFO(
            "Vulkan cluster (mega-geometry) acceleration structures disabled: "
            "clusterAccelerationStructure={} bufferDeviceAddress={} and "
            "required extensions={}",
            cluster_feature, device_address, has_cluster_as_extensions);
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
    if (cluster_acceleration_structure_enabled) {
        // VK_NV_cluster_acceleration_structure only requires
        // VK_KHR_acceleration_structure, which the ray-query block above has
        // already enabled whenever raytracing_enabled is set.
        // TODO(dx): the D3D12 counterpart is a DXR/NVAPI cluster path with its
        // own device-level opt-in; see the RTXMG research report §7.
        enable_device_extension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME);
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
    // VK_KHR_maintenance5 (core in Vulkan 1.4) is enabled whenever the physical
    // device supports it: it adds VkBufferUsageFlags2CreateInfoKHR, which the
    // 64-bit usage masks of newer optional features (memory decompression)
    // require, and it only removes API friction otherwise. Its feature
    // structure is empty apart from the aggregate bool.
    bool maintenance5_enabled = false;
    if (supports_device_extension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
        VkPhysicalDeviceMaintenance5FeaturesKHR supported_maintenance5{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
            .pNext = nullptr};
        VkPhysicalDeviceFeatures2 maintenance5_features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &supported_maintenance5};
        vkGetPhysicalDeviceFeatures2(physical_device, &maintenance5_features2);
        if (supported_maintenance5.maintenance5 == VK_TRUE) {
            enable_device_extension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
            maintenance5_enabled = true;
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
    // VK_EXT/VK_KHR_pipeline_library_group_handles may only be enabled on a
    // device that also enables VK_KHR_pipeline_library
    // (VUID-vkCreateDevice-ppEnabledExtensionNames-01387). The prerequisite has
    // no feature structure, so only its name matters - but it must be added
    // before the registry resolves, and only for a device that actually asked
    // for group handles *and* exposes the prerequisite: enabling it
    // unconditionally would change ppEnabledExtensionNames on devices that never
    // use either extension. When the prerequisite is unavailable the registry's
    // `requires_extension` check rejects the group-handle entry instead.
    if (_optional_extensions.requested(
            detail::OptionalExtension::PIPELINE_LIBRARY_GROUP_HANDLES) &&
        supports_device_extension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)) {
        enable_device_extension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    }
    // Finish the optional extension negotiation after every mandatory
    // extension has been decided, so a prerequisite (swapchain, ray tracing
    // pipeline, ...) is already visible.
    for (auto const &name : _enable_device_exts) {
        _optional_extensions.note_enabled_extension(name.c_str());
    }
    if (surface_enabled) {
        // VK_KHR_swapchain is appended inside create_logical_device(), so the
        // registry cannot see it otherwise. Present/swapchain-maintenance
        // extensions declare it as a prerequisite.
        _optional_extensions.note_enabled_extension(
            VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    _optional_extensions.resolve();
    for (auto name : _optional_extensions.enabled_extension_names()) {
        enable_device_extension(name);
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
    // Append the optional feature structures to the caller-provided tail. The
    // chain was validated above without them: those sTypes are owned by the
    // backend (see backend_owned_device_feature_structure_types), so a config
    // must not be able to supply them.
    feature_next = _optional_extensions.device_feature_chain(feature_next);
    // VK_KHR_maintenance5 is chained only when it is actually enabled: a feature
    // structure whose extension is not enabled is a validation error. (The
    // `ENABLE_HIDDEN_FEATURES` block above carries a dormant second copy of this
    // structure; that macro is never defined, so the two cannot collide.)
    VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
        .pNext = feature_next,
        .maintenance5 = maintenance5_enabled ? VK_TRUE : VK_FALSE};
    if (maintenance5_enabled) {
        feature_next = &maintenance5_features;
    }
    if (auto status = _optional_extensions.status_string(); !status.empty()) {
        LUISA_INFO("Vulkan optional device extensions:\n{}", status);
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
    // Cluster (mega-geometry) acceleration structures ride on the same
    // acceleration-structure chain: the extension only requires
    // VK_KHR_acceleration_structure, so the two are always enabled together.
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV cluster_acceleration_structure_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
        .pNext = &enabled_acceleration_structure_features,
        .clusterAccelerationStructure =
            cluster_acceleration_structure_enabled ? VK_TRUE : VK_FALSE};
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
        if (cluster_acceleration_structure_enabled) {
            // Link the cluster feature into the single already-single-linked
            // chain that starts with the ray-tracing feature structures.
            cluster_acceleration_structure_features.pNext = feature_next;
            feature_next = &cluster_acceleration_structure_features;
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
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
        .pNext = feature_next,
        .shaderBFloat16Type = enable_bfloat16 ? VK_TRUE : VK_FALSE,
        .shaderBFloat16DotProduct = VK_FALSE,
        .shaderBFloat16CooperativeMatrix = VK_FALSE};
    if (enable_bfloat16) {
        feature_next = &bfloat16_features;
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
    // The replicated-composite feature is enabled for every reason that needs
    // it: the standalone splat-composite optimization and the cooperative
    // vector prerequisite. `enable_replicated_composites` is exactly the
    // standalone probe result, and cooperative-vector enablement already
    // requires it, so one chain position covers both.
    if (enable_replicated_composites) {
        replicated_composites_features.pNext = feature_next;
        replicated_composites_features.shaderReplicatedComposites = VK_TRUE;
        feature_next = &replicated_composites_features;
    }
    VkPhysicalDeviceShaderUntypedPointersFeaturesKHR
        untyped_pointers_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
            .pNext = feature_next,
            .shaderUntypedPointers = VK_TRUE};
    if (shader_untyped_pointers_enabled) {
        feature_next = &untyped_pointers_features;
    }
#ifdef LUISA_VULKAN_ENABLE_CUDA_INTEROP
    VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda_kernel_launch_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV,
        .pNext = feature_next,
        .cudaKernelLaunchFeatures =
            cuda_kernel_launch_enabled ? VK_TRUE : VK_FALSE};
    if (cuda_kernel_launch_enabled) {
        feature_next = &cuda_kernel_launch_features;
    }
#endif
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
        .bufferDeviceAddress = device_address_enabled ? VK_TRUE : VK_FALSE,
        // The SPIR-V codegen switches to the Vulkan memory model only for
        // cooperative-vector shaders (target_feature::cooperative_vector), so
        // enable the memory-model features only when the extension is fully
        // usable. On devices missing any prerequisite, keep the pre-existing
        // GLSL450 device-scope behavior instead of toggling the memory model.
        .vulkanMemoryModel = cooperative_vector_enabled ? VK_TRUE : VK_FALSE,
        .vulkanMemoryModelDeviceScope =
            cooperative_vector_enabled ? VK_TRUE : VK_FALSE};
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
            .shader_bfloat16 = enable_bfloat16,
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

    // ---- plan P2.2: per-dispatch descriptor update templates ----------
    // One decision, at creation time: the negotiation must have enabled the
    // extension (which folds in "the device supports it" and the
    // VulkanDeviceConfigExt switch) *and* the loader must have resolved all
    // three entry points. Otherwise every set-0 update keeps the pre-P2.2
    // vkUpdateDescriptorSets call. Nothing re-evaluates this per dispatch.
    {
        auto negotiated = _optional_extensions.enabled(
            detail::OptionalExtension::DESCRIPTOR_UPDATE_TEMPLATE);
#ifdef VK_KHR_descriptor_update_template
        // Core spelling first: the entry points are core Vulkan 1.1, so a
        // driver that implements them exposes the core names. The KHR aliases
        // are the fallback (same rule as cmd_pipeline_barrier / queue_submit).
        if (negotiated) {
            _create_descriptor_update_template =
                ::vkCreateDescriptorUpdateTemplate != nullptr ?
                    ::vkCreateDescriptorUpdateTemplate :
                    ::vkCreateDescriptorUpdateTemplateKHR;
            _destroy_descriptor_update_template =
                ::vkDestroyDescriptorUpdateTemplate != nullptr ?
                    ::vkDestroyDescriptorUpdateTemplate :
                    ::vkDestroyDescriptorUpdateTemplateKHR;
            _update_descriptor_set_with_template =
                ::vkUpdateDescriptorSetWithTemplate != nullptr ?
                    ::vkUpdateDescriptorSetWithTemplate :
                    ::vkUpdateDescriptorSetWithTemplateKHR;
        }
#else
        if (negotiated) {
            _create_descriptor_update_template =
                ::vkCreateDescriptorUpdateTemplate;
            _destroy_descriptor_update_template =
                ::vkDestroyDescriptorUpdateTemplate;
            _update_descriptor_set_with_template =
                ::vkUpdateDescriptorSetWithTemplate;
        }
#endif
        descriptor_update_template_capable_bit =
            _create_descriptor_update_template != nullptr &&
            _destroy_descriptor_update_template != nullptr &&
            _update_descriptor_set_with_template != nullptr;
        if (negotiated && !descriptor_update_template_capable_bit) {
            LUISA_WARNING(
                "Vulkan descriptor update templates were negotiated but the "
                "create/update/destroy entry points are not resolved; keeping "
                "vkUpdateDescriptorSets for per-dispatch descriptor updates.");
        }
        LUISA_INFO("Vulkan per-dispatch descriptor updates: {}",
                   descriptor_update_template_capable_bit ?
                       "VkDescriptorUpdateTemplate" :
                       "vkUpdateDescriptorSets");
    }

    // ---- plan P2.3: set-0 push descriptors ------------------------------
    // One decision, at creation time: the registry must have enabled
    // VK_KHR_push_descriptor (which folds in "the physical device exposes the
    // name" and the VulkanDeviceConfigExt switch), the loader must have resolved
    // vkCmdPushDescriptorSetKHR, and the physical device must report a usable
    // maxPushDescriptors. Otherwise `push_descriptor_capable_bit` stays false and
    // every shader is created with the classic set-0 layout. Nothing re-evaluates
    // this per dispatch and no code path can switch mid-frame: the push
    // descriptor set is a property of the descriptor-set layout, which is created
    // and interned at shader creation.
    {
        auto negotiated = _optional_extensions.enabled(
            detail::OptionalExtension::PUSH_DESCRIPTOR);
        if (negotiated) {
#ifdef VK_KHR_push_descriptor
            // The KHR entry point is the one the extension enables.
            _cmd_push_descriptor_set = ::vkCmdPushDescriptorSetKHR;
#endif
#if defined(VK_VERSION_1_4)
            // A Vulkan 1.4 driver may expose only the core spelling; it is the
            // same command with the same dispatch semantics, and the extension
            // itself still had to be negotiated above.
            if (_cmd_push_descriptor_set == nullptr) {
                _cmd_push_descriptor_set = ::vkCmdPushDescriptorSet;
            }
#endif
        }
        if (negotiated && _cmd_push_descriptor_set != nullptr) {
            // maxPushDescriptors is the only limit a push layout must respect
            // (VUID-VkDescriptorSetLayoutCreateInfo-flags-00281). The property
            // structure is only chained when the extension is exposed, i.e. when
            // the driver accepts its structure type.
            VkPhysicalDevicePushDescriptorPropertiesKHR push_properties{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
                .pNext = nullptr,
                .maxPushDescriptors = 0u};
            VkPhysicalDeviceProperties2 push_properties2{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                .pNext = &push_properties};
            vkGetPhysicalDeviceProperties2(physical_device, &push_properties2);
            _max_push_descriptors = push_properties.maxPushDescriptors;
        }
        push_descriptor_capable_bit =
            _cmd_push_descriptor_set != nullptr && _max_push_descriptors != 0u;
        if (negotiated && !push_descriptor_capable_bit) {
            LUISA_WARNING(
                "Vulkan push descriptors were negotiated but the entry point or "
                "maxPushDescriptors is unavailable (maxPushDescriptors={}); "
                "keeping classic set-0 allocation for every shader.",
                _max_push_descriptors);
        }
        LUISA_INFO(
            "Vulkan set-0 descriptor binding: {} (maxPushDescriptors={})",
            push_descriptor_capable_bit ?
                "vkCmdPushDescriptorSetKHR" :
                "classic allocate/update/bind",
            _max_push_descriptors);
    }

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
detail::ShaderCompilePool &Device::shader_compile_pool() noexcept {
    // Constructed by `Device::Device`, which is the only place that can capture the
    // executor: this accessor runs on whatever thread a batch or the destructor uses,
    // so it must never select one. A missing pool is a construction bug, and the
    // fail-safe is the OS thread executor (no scheduler captured) rather than a
    // scheduler lookup on the wrong thread.
    if (!_shader_compile_pool) [[unlikely]] {
        LUISA_WARNING("Vulkan shader compile pool was not constructed with its "
                      "device; preparation falls back to the pool's own threads "
                      "because no scheduler was captured on the device-constructing "
                      "thread.");
        _shader_compile_pool = luisa::make_unique<detail::ShaderCompilePool>(
            this, static_cast<marl::Scheduler *>(nullptr));
    }
    return *_shader_compile_pool;
}
void Device::shutdown_shader_compile_pool() noexcept {
    // The pool object itself already exists (the constructor builds it with the
    // captured executor), but it owns nothing until a batch needs it; either way a
    // shutdown before the first batch has to leave state that refuses later work with
    // `POOL_SHUTDOWN`, otherwise a caller that shuts the pool down early would
    // silently get the parallel path back.
    static_cast<void>(shader_compile_pool());
    _shader_compile_pool->stop();
}
Device::~Device() {
    // Ordered batch compilation (plan P5.1) stops first: it holds threads that may
    // still be preparing a batch, and a prepared record names no Vulkan object, so
    // joining them here - before the replay drain, the interned layouts, the
    // pipeline cache, the BinaryIO and the logical device - is both sufficient and
    // the last point at which it is safe.
    shutdown_shader_compile_pool();
    // Extension-owned streams must go first: their destructors synchronize and
    // destroy the internal COPY stream they own, which needs both the logical
    // device and `_streams` to still be alive.
    {
        vstd::vector<CustomStream *> leftovers;
        {
            std::lock_guard lck{_custom_stream_mtx};
            leftovers.reserve(_custom_streams.size());
            for (auto &i : _custom_streams) {
                leftovers.emplace_back(i.second);
            }
            _custom_streams.clear();
        }
        for (auto *stream : leftovers) {
            delete stream;
        }
    }
    if (_vk_device) {
        // Replay graphs (plan P4) are drained here: after the extension-owned
        // streams above, which is where every stream has been synchronized, and
        // before the interned layouts, the descriptor pools and the native device
        // are destroyed. Each graph frees its recorded command buffer, its state
        // (pools, descriptor sets, staging, retained pipelines) and finally the
        // interned layout references it retained by id, so ordering the drain
        // before `_layout_interner.destroy_all` is what keeps those releases
        // meaningful instead of no-ops.
        {
            // A graph whose stream is still alive may have a replay in flight;
            // `RecordedBatch` must not be freed under a pending submission, so
            // those streams are synchronized once here. Streams that were already
            // destroyed synchronized in their own destructor.
            for (auto stream_handle : _replay_registry.stream_handles()) {
                if (auto *stream = find_stream(stream_handle); stream != nullptr) {
                    stream->sync();
                }
            }
            auto drained = _replay_registry.destroy_all_instances(
                "the device was destroyed");
            if (drained != 0u) {
                LUISA_INFO(
                    "Vulkan replay: freed {} graph(s) at device teardown.",
                    drained);
            }
        }
        // Lazy kernels are members and would otherwise be destroyed after this
        // body. Destroy them now: their descriptor update templates and pipeline
        // layouts must not outlive the descriptor-set layouts drained below.
        set_bindless_kernel.reset();
        set_accel_kernel.reset();
        prepare_indirect_kernel.reset();
        // Interned shader layouts are device-owned (plan P1.1): destroy every
        // survivor exactly once, here, while the logical device is still alive.
        // A shader that outlives this point only drops references, which are
        // no-ops after the drain.
        _layout_interner.destroy_all(logic_device());
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
        // The device pipeline cache dies with the Device (plan P1.3): persist it
        // to its single artifact and destroy the handle here, while the VkDevice
        // and the BinaryIO are still valid. `_default_file_io` is released right
        // after this block, so this is the last safe point. Shaders only borrow the
        // handle, so a shader destroyed later cannot touch it.
        _pipeline_cache.release(this);
    }
    _exts.clear();
    _default_file_io = nullptr;
#ifndef LC_NO_HLSL_BUILTIN
    {
        std::lock_guard lck(vk_device_detail_2::g_dxc_mutex);
        if (vk_device_detail_2::g_dxc_ref_count > 0 && --vk_device_detail_2::g_dxc_ref_count == 0) {
            if (vk_device_detail_2::g_dxc_compiler_initialized) {
                vk_device_detail_2::g_dxc_compiler.destroy();
                vk_device_detail_2::g_dxc_compiler_initialized = false;
            }
            vk_device_detail_2::g_dxc_runtime_directory.clear();
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
    // Replay graphs (plan P4) are invalidated *before* the wrapper and its native
    // buffer go away: a later replay() returns RESOURCE_DESTROYED instead of
    // submitting a command buffer that names a freed VkBuffer. The graphs store the
    // native identity by value, so this comparison never dereferences anything.
    if (auto native = detail::handle_to_uint64(buffer->vk_buffer()); native != 0u) {
        auto message = luisa::format(
            "buffer 0x{:x} was destroyed while a replay graph still named it",
            handle);
        _replay_registry.invalidate_buffers(&native, 1u, message);
    }
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
void Device::add_custom_stream(uint64_t handle, CustomStream *stream) noexcept {
    std::lock_guard lck{_custom_stream_mtx};
    auto emplaced = _custom_streams.try_emplace(handle, stream).second;
    LUISA_ASSERT(emplaced, "Duplicate Vulkan custom-stream handle registered.");
}
CustomStream *Device::find_custom_stream(uint64_t handle) noexcept {
    std::lock_guard lck{_custom_stream_mtx};
    auto iter = _custom_streams.find(handle);
    return iter == _custom_streams.end() ? nullptr : iter->second;
}
CustomStream *Device::remove_custom_stream(uint64_t handle) noexcept {
    std::lock_guard lck{_custom_stream_mtx};
    auto iter = _custom_streams.find(handle);
    if (iter == _custom_streams.end()) { return nullptr; }
    auto *stream = iter->second;
    _custom_streams.erase(iter);
    return stream;
}
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
Stream *Device::find_stream(uint64_t handle) noexcept {
    if (handle == 0u) { return nullptr; }
    std::lock_guard lck{_stream_mtx};
    auto *stream = reinterpret_cast<Stream *>(handle);
    // Only a handle the device really registered is returned: a public extension
    // must not reinterpret an unchecked integer, and an extension-owned
    // `CustomStream` is not a recordable stream.
    for (auto *candidate : _streams) {
        if (candidate == stream) { return stream; }
    }
    return nullptr;
}

void Device::destroy_stream(uint64_t handle) noexcept {
    if (auto *custom = remove_custom_stream(handle)) {
        // Unregister before deleting: the destructor must not be able to
        // re-enter this path for its own handle.
        delete custom;
        return;
    }
    auto *stream = reinterpret_cast<Stream *>(handle);
    {
        std::lock_guard lck{_stream_mtx};
        _streams.erase(stream);
        // Phase 1 of the replay teardown (plan P4): mark every graph bound to this
        // stream before anything is deleted. The graphs free no native state here
        // (their command buffer may still be pending on this stream); they become
        // tombstones that answer STREAM_DESTROYED and are freed later by
        // `destroy()` or by the device drain, which runs after the stream's own
        // synchronization made that legal.
        auto message = luisa::format(
            "stream 0x{:x} was destroyed while a replay graph was bound to it",
            handle);
        _replay_registry.invalidate_stream(handle, message);
    }
    delete stream;
}
void Device::synchronize_stream(uint64_t stream_handle) noexcept {
    if (auto *custom = find_custom_stream(stream_handle)) {
        custom->synchronize();
        return;
    }
    reinterpret_cast<Stream *>(stream_handle)->sync();
}
void Device::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    if (auto *custom = find_custom_stream(stream_handle)) {
        custom->dispatch(std::move(list));
        return;
    }
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
    // Plan P6.1: the 14 core Vulkan limits. Answered from the one capability
    // record captured at creation, which is also what
    // VulkanExtensionStatusExt::capability_status() reports, so a query answer and
    // a status report cannot disagree. An empty answer is "this Vulkan device has
    // no value for this limit" (only min/max subgroup size and subgroup ops can be
    // unknown), never zero.
    if (auto limit = detail::core_limit_from_name(property)) {
        return detail::query_core_limit(_capability_status, *limit);
    }
    // Plan P6.2: no Vulkan physical SM-count query exists, so `sm_count` is
    // answered only from a caller-supplied calibrated estimate and is empty
    // otherwise; `sm_count_source` says which of the two it is. A consumer that
    // needs a device attribute must check the source.
    if (property == "sm_count") {
        return detail::query_sm_count(_capability_status);
    }
    if (property == "sm_count_source") {
        return detail::query_sm_count_source(_capability_status);
    }
    // Counter-only submit profiler (plan P0.2): every vk_profile_* key is
    // answered by the profiler itself, which never feeds a backend decision.
    if (property.starts_with("vk_profile_")) {
        return submit_profile::query(property);
    }
    if (property == "vk_queue_submit_path") {
        // Diagnostic of the queue-submission entry point this device uses (plan
        // P2.1): "submit2" for vkQueueSubmit2 (+ VkSemaphoreSubmitInfo stage
        // masks), "legacy" for the Vulkan 1.2 vkQueueSubmit path. Reported, never
        // read back by the backend: `sync2_capable()` already decides it.
        return sync2_capable() ? luisa::string{"submit2"} : luisa::string{"legacy"};
    }
if (property == "vk_descriptor_update_template_path") {
          // Diagnostic of the per-dispatch descriptor-update path (plan P2.2):
          // "template" when set-0 updates go through a cached
          // VkDescriptorUpdateTemplate, "legacy" when the device, the config
          // switch or the entry points force vkUpdateDescriptorSets. Latched at
          // creation; reported, never read back to change behaviour.
          return descriptor_update_template_capable() ?
                     luisa::string{"template"} :
                     luisa::string{"legacy"};
      }
      if (property == "vk_push_descriptor_path") {
          // Diagnostic of the set-0 binding path (plan P2.3): "push" when DSL
          // shaders are created with a push descriptor set 0 and the encoder
          // records vkCmdPushDescriptorSetKHR, "classic" otherwise. Latched at
          // creation; every shader still decides for itself whether its own set 0
          // qualifies, and a shader that does not keeps the classic layout -
          // the counters `vk_profile_push_descriptor_shaders` /
          // `_classic_shaders` report that split.
          return push_descriptor_capable() ?
                     luisa::string{"push"} :
                     luisa::string{"classic"};
      }
      if (property == "vk_push_descriptor_max_count") {
          // VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors, or 0
          // when the push path is unavailable (no extension, no entry point, no
          // property query). Reporter only.
          return luisa::format("{}", max_push_descriptors());
      }
    if (property == "vk_max_frames_in_flight") {
        // Diagnostic of the in-flight depth latched at creation (plan P1.5).
        // Reported, never read back by the backend: the value is the contract the
        // streams already follow, so answering it cannot change behaviour.
        return luisa::format("{}", _max_frames_in_flight);
    }
    if (property == "vk_spirv_codegen_route") {
     // Diagnostic of the *default* user-compute codegen route: "native" when this
     // backend was built with the native XIR-to-SPIR-V codegen
     // (lc_vk_backend_use_xir_spirv / LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV, which is
     // what gives a compute kernel the indirect-dispatch metadata ABI), "hlsl" when
     // it was not and every kernel therefore takes the compatibility HLSL route.
     // It is the device default, not a per-kernel promise: a kernel that needs the
     // HLSL fallback (printing, native_include, async copy, motion blur,
     // accel.size(), BF16 without the feature, ray-query cluster id) takes it even
     // on a "native" device. Reported, never read back to change behaviour.
#ifdef LUISA_XIR_TO_SPIRV
     return luisa::string{"native"};
#else
     return luisa::string{"hlsl"};
#endif
 }
 if (property == "vk_replay_supported") {
        // Plan P4: replayable recording of a direct compute dispatch and a
        // device-to-device buffer copy, with UNIFORM/BUFFER arguments only. Always
        // "1" for this backend, because the contract is a property of the
        // implementation rather than of the device; a narrowed build would answer
        // "0" here and nowhere else.
        return luisa::string{"1"};
    }
    if (property.starts_with("vk_shader_compile_")) {
        // Diagnostics of the ordered batch-compilation pool (plan P5.1). They report
        // state and counters, never a decision: no code path reads them. Read from
        // the pool's atomics, so a query on one thread while another publishes a
        // batch cannot see a torn value; a device that never batched answers zeroes,
        // and the executor it captured needs no batch at all.
        auto stats = _shader_compile_pool ?
                         _shader_compile_pool->stats() :
                         VulkanShaderCompileStats{};
        // Which executor the pool captured on the device-constructing thread, and
        // never re-selected: "fiber" (a marl scheduler was bound there, so the pool
        // owns no thread; worker-backed schedulers run preparation as marl tasks)
        // or "std_thread" (nothing was bound there, so the pool owns its own workers).
        // Reported, never read back to change behaviour: it exists so a test can
        // prove the selection without reaching into the backend.
        if (property == "vk_shader_compile_executor") {
            return luisa::string{_shader_compile_pool ?
                                         _shader_compile_pool->executor_name() :
                                         "none"};
        }
        // Worker threads of the captured scheduler: 0 in the "std_thread" executor.
        if (property == "vk_shader_compile_scheduler_workers") {
            return luisa::format("{}", _shader_compile_pool ?
                                         _shader_compile_pool->captured_scheduler_worker_count() :
                                         0u);
        }
        if (property == "vk_shader_compile_parallel") {
            return luisa::string{"1"};
        }
        if (property == "vk_shader_compile_worker_count") {
            return luisa::format("{}", stats.worker_count);
        }
        if (property == "vk_shader_compile_pool_state") {
            if (stats.pool_shutdown) { return luisa::string{"shutdown"}; }
            return luisa::string{stats.pool_created ? "running" : "not_created"};
        }
        if (property == "vk_shader_compile_batches") {
            return luisa::format("{}", stats.batches);
        }
        if (property == "vk_shader_compile_requests") {
            return luisa::format("{}", stats.requests);
        }
        if (property == "vk_shader_compile_parallel_prepares") {
            return luisa::format("{}", stats.parallel_prepares);
        }
        if (property == "vk_shader_compile_serial_prepares") {
            return luisa::format("{}", stats.serial_prepares);
        }
        if (property == "vk_shader_compile_cache_hits") {
            return luisa::format("{}", stats.cache_hits);
        }
        if (property == "vk_shader_compile_serial_fallbacks") {
            return luisa::format("{}", stats.serial_fallbacks);
        }
        if (property == "vk_shader_compile_publish_order_waits") {
            return luisa::format("{}", stats.publish_order_waits);
        }
        if (property == "vk_shader_compile_publish_order_wait_ns") {
            return luisa::format("{}", stats.publish_order_wait_ns);
        }
        return luisa::string{};
    }
    if (property == "vk_replay_graphs") {
        // Live recordings of this device, read from the registry on every query so
        // it can never drift from the graphs that actually exist.
        return luisa::format("{}", _replay_registry.size());
    }
    if (property.starts_with("vk_pipeline_cache_")) {
        // Diagnostics of the device-scoped pipeline cache (plan P1.3). They
        // report state, never a decision: no code path reads them.
        if (property == "vk_pipeline_cache_handle") {
            return luisa::format("{}", reinterpret_cast<uintptr_t>(_pipeline_cache.handle()));
        }
        if (property == "vk_pipeline_cache_state") {
            return luisa::string{_pipeline_cache.available() ? "ready" : "unavailable"};
        }
        if (property == "vk_pipeline_cache_seed") {
            switch (_pipeline_cache.seed_status()) {
                case detail::DevicePipelineCache::SeedStatus::kLoaded:
                    return luisa::string{"loaded"};
                case detail::DevicePipelineCache::SeedStatus::kAbsent:
                    return luisa::string{"absent"};
                case detail::DevicePipelineCache::SeedStatus::kRejected:
                    return luisa::string{"rejected"};
                case detail::DevicePipelineCache::SeedStatus::kUnavailable:
                    return luisa::string{"unavailable"};
            }
            return {};
        }
        if (property == "vk_pipeline_cache_seed_reason") {
            return luisa::string{_pipeline_cache.seed_reason()};
        }
        if (property == "vk_pipeline_cache_seed_bytes") {
            return luisa::format("{}", _pipeline_cache.seed_bytes());
        }
        if (property == "vk_pipeline_cache_artifact_name") {
            return luisa::string{_pipeline_cache.artifact_name()};
        }
        if (property == "vk_pipeline_cache_merged_seeds") {
            return luisa::format("{}", _pipeline_cache.merged_seeds());
        }
        if (property == "vk_pipeline_cache_rejected_seeds") {
            return luisa::format("{}", _pipeline_cache.rejected_seeds());
        }
        return {};
    }
    if (property == "shader_untyped_pointers") {
        return shader_untyped_pointers_enabled ? "true" : "false";
    }
    if (property == "opacity_micromap") {
        // "1"/"0": a micromap is usable only when an attach would actually be legal
        // on this device, not merely when the extension was negotiated.
        auto legal = optional_extension_enabled(detail::OptionalExtension::OPACITY_MICROMAP) &&
                     !use_fallback_rtx_bit &&
                     (OpacityMicromap::ext_alias_negotiated(this) ||
                      opacity_micromap_execution_mode_available());
        return legal ? "1" : "0";
    }
    if (property == "opacity_micromap_max_two_state_level") {
        return luisa::format("{}", opacity_micromap_max_subdivision_level(false));
    }
    if (property == "opacity_micromap_max_four_state_level") {
        return luisa::format("{}", opacity_micromap_max_subdivision_level(true));
    }
    if (property == "vk_cluster_acceleration_structure") {
        // "1"/"0" (not the "true"/"false" of the neighbouring feature queries):
        // the integration test parses this value to decide whether to skip, and
        // LC_CLUSTER_IMPL_SPEC §5.1 fixes the encoding.
        // DirectX exposes the corresponding capability as
        // "dx_cluster_acceleration_structure".
        return cluster_acceleration_structure_enabled ? "1" : "0";
    }
    if (property == "shader_device_clock") {
        return _shader_device_clock_enabled ? "true" : "false";
    }
    if (property == "buffer_device_address") {
        return device_address_enabled ? "true" : "false";
    }
    if (property == "motion_blur") {
        return enable_motion_blur() ? "true" : "false";
    }
    if (property == "shader_int64") {
        return _numeric_features.shader_int64 ? "true" : "false";
    }
    if (property == "shader_bfloat16") {
#ifdef LC_NO_HLSL_BUILTIN
        // Without the bundled HLSL compiler there is no software BF16 either.
        return _numeric_features.shader_bfloat16 ? "true" : "false";
#else
        // A device without shaderBFloat16Type still serves BF16: such a kernel is
        // routed to the compatibility HLSL route and emulates BF16 in software
        // (src/backends/common/hlsl/builtin/bfloat16_fallback.bytes).
        return "true";
#endif
    }
    if (property == "timeline_semaphore_max_value_difference") {
        // The largest one-shot timeline-semaphore jump from the current
        // counter (VkPhysicalDeviceTimelineSemaphoreProperties::
        // maxTimelineSemaphoreValueDifference). 0 (no limit) is normalized to
        // UINT64_MAX by max_timeline_semaphore_value_difference().
        return luisa::format("{}", max_timeline_semaphore_value_difference());
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
    // Ray-query modules must declare `OpExecutionModeId OpacityMicromapIdKHR` when
    // the *KHR* spelling of the extension was negotiated; the EXT alias does not
    // need one. `opacity_micromap_execution_mode_available()` is the single source of
    // truth, and `query("opacity_micromap")` uses it too.
    enable(owned_logical_device && opacity_micromap_execution_mode_available(),
           target_feature::ray_tracing_opacity_micromap);
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
    enable(_numeric_features.shader_bfloat16,
           target_feature::shader_bfloat16);
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
    enable(owned_logical_device && shader_untyped_pointers_enabled,
           target_feature::shader_untyped_pointers);
    enable(owned_logical_device && cooperative_vector_enabled,
           target_feature::cooperative_vector);
    enable(owned_logical_device && shader_replicated_composites_enabled,
           target_feature::shader_replicated_composites);
    // Optional shader capability extensions. `optional_extension_enabled` is
    // already false for an imported logical device and for a switch the
    // application turned off, so the artifact requirement bits can never claim a
    // capability the device did not attest.
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_FMA),
           target_feature::shader_fma_float32);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_64BIT_INDEXING),
           target_feature::shader_64bit_indexing);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY),
           target_feature::shader_uniform_buffer_unsized_array);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_LONG_VECTOR),
           target_feature::shader_long_vector);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_SUBGROUP_PARTITIONED),
           target_feature::shader_subgroup_partitioned);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_SPLIT_BARRIER),
           target_feature::shader_split_barrier);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_OCP_MICROSCALING_TYPES),
           target_feature::shader_ocp_microscaling_types);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_CONSTANT_DATA),
           target_feature::shader_constant_data);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_ABORT),
           target_feature::shader_abort);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_MIXED_FLOAT_DOT_PRODUCT),
           target_feature::shader_mixed_float_dot_product_float16_acc_float32);
    enable(optional_extension_enabled(detail::OptionalExtension::SHADER_EXPECT_ASSUME),
           target_feature::shader_expect_assume);
    return mask;
}

#ifdef LUISA_XIR_TO_SPIRV
namespace vk_device_detail_2 {
[[nodiscard]] static uint64_t xir_spirv_environment_hash() noexcept {
    auto hash_env = [](const char *name) noexcept {
        auto value = luisa::get_environment_variable(name);
        return !value ? 0ull :
                        luisa::hash_value(luisa::string_view{*value});
    };
    return luisa::hash_combine({
        hash_env("LUISA_XIR_DISABLE_OPTIMIZATION"),
        hash_env("LUISA_XIR_ENABLE_SCALARIZER"),
        hash_env("LUISA_SPIRV_OPT_LEVEL"),
        hash_env("LUISA_SPIRV_OPT_PASSES"),
        // Backend-owned loop-control hints change the emitted OpLoopMerge
        // literals, so a device-wide policy change must invalidate artifacts.
        hash_env("LUISA_SPIRV_LOOP_HINTS"),
    });
}

}// namespace vk_device_detail_2
namespace vk_device_detail_2 {
[[nodiscard]] static vstd::MD5 compute_shader_cache_md5(
    Function kernel, const ShaderOption &option,
    lc::spirv::SpirvTargetFeatures target_features,
    luisa::optional<uint8_t> required_subgroup_size) noexcept {
    using namespace std::string_view_literals;
    auto option_flags =
        (static_cast<uint64_t>(option.enable_fast_math) << 0u) |
        (static_cast<uint64_t>(option.enable_debug_info) << 1u) |
        (static_cast<uint64_t>(option.enable_extended_accel_limits) << 2u) |
        (static_cast<uint64_t>(option.enable_scalarizer) << 3u);
    auto block_size = kernel.block_size();
    uint64_t data[] = {
        luisa::hash_value("luisa-vk-xir-spv-cache-v16"sv),
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
        // ROUND lowering is a codegen policy, not a feature bit, so it must be
        // part of the cache identity as well.
        static_cast<uint64_t>(target_features.round_policy),
        vk_device_detail_2::xir_spirv_environment_hash(),
        static_cast<uint64_t>(required_subgroup_size.value_or(0u)),
    };
    return vstd::MD5{vstd::span<const uint8_t>{
        reinterpret_cast<const uint8_t *>(data), sizeof(data)}};
}
}// namespace vk_device_detail_2
#endif

ShaderCreationInfo Device::_create_shader_hlsl(
    const ShaderOption &option, Function kernel,
    bool requires_sampler_anisotropy,
    bool fallback_rtx) noexcept {
#ifdef LC_NO_HLSL_BUILTIN
    if (fallback_rtx) {
        LUISA_ERROR(
            "This Vulkan shader traces rays through the software fallback "
            "acceleration structure "
            "(VulkanDeviceConfigExt::use_fallback_rtx()), which is emitted by "
            "the compatibility HLSL-to-SPIR-V codegen, but this Vulkan backend "
            "was built without the bundled HLSL compiler.");
    }
    LUISA_ERROR(
        "This Vulkan shader requires the HLSL-to-SPIR-V fallback, but the "
        "backend was built without the bundled HLSL compiler.");
#else
    // The fallback traversal is HLSL text, not a hardware ray query, so it
    // does not need VK_KHR_ray_query; only the hardware path does.
    if (kernel.requires_raytracing() && !raytracing_enabled && !fallback_rtx) {
        LUISA_ERROR(
            "Vulkan shader '{}' requires ray tracing, but ray-query support "
            "is not enabled on this device.",
            kernel.name());
    }
    ShaderCreationInfo info;
    auto required_subgroup_size =
        effective_required_subgroup_size(kernel, option.compile_only);
    uint mask = 0u;
    if (option.enable_fast_math) { mask |= 1u; }
    if (option.enable_debug_info) { mask |= 2u; }
    if (required_subgroup_size) {
        mask |= (static_cast<uint>(*required_subgroup_size) << 8u);
    }

    // The shader-capability extensions this device negotiated, in the form the HLSL
    // codegen takes them: the generated source only contains an optional
    // instruction when its flag is set here (SpirvShaderCapabilities).  The mask is
    // the one the native SPIR-V route publishes for its artifacts, so both routes of
    // the backend agree on what the device can execute.
    auto artifact_features = enabled_spirv_artifact_features();
    hlsl::SpirvShaderCapabilities spirv_capabilities;
    spirv_capabilities.fma =
        (artifact_features & spirv::target_feature::shader_fma_float32) != 0u;
    spirv_capabilities.abort =
        (artifact_features & spirv::target_feature::shader_abort) != 0u;
    spirv_capabilities.mixed_float_dot_product =
        (artifact_features & spirv::target_feature::shader_mixed_float_dot_product_float16_acc_float32) != 0u;
    // VK_EXT_shader_split_barrier is a shader-capability opt-in published in
    // the artifact-feature mask; compute only (raster never enables it).
    spirv_capabilities.split_barrier =
        (artifact_features & spirv::target_feature::shader_split_barrier) != 0u;
    // VK_NV_cluster_acceleration_structure is negotiated as a device feature
    // (clusterAccelerationStructure) rather than as a shader-capability opt-in, so
    // the flag follows the device decision directly.  Its SPIR-V counterpart
    // SPV_NV_cluster_acceleration_structure is what the HLSL codegen then enables
    // for dxc, and both are required for
    // OpRayQueryGetIntersectionClusterIdNV: the module declares
    // RayTracingClusterAccelerationStructureNV, which a device that did not
    // enable the extension rejects at pipeline creation.
    spirv_capabilities.cluster_acceleration_structure =
        cluster_acceleration_structure_enabled;
    // VK_KHR_maintenance9 only relaxes the operand width of the bitwise
    // instructions, so it declares no shader capability of its own and has no
    // artifact-feature bit: the flag follows the negotiated device extension
    // directly.  Off (the default) keeps the legacy 32-bit emulation.
    spirv_capabilities.maintenance9_bitwise =
        optional_extension_enabled(detail::OptionalExtension::MAINTENANCE9);
    auto code = [&]() -> hlsl::CodegenResult {
        hlsl::CodegenUtility util;
        // The traversal and the argument layout of a fallback acceleration
        // structure belong to the HLSL codegen, which is asked for them with
        // the trailing flag below; a build whose codegen predates them refuses
        // the shader instead of compiling a hardware traversal.
        return detail::codegen_compat_compute(
            util, kernel, option.native_include, mask,
            option.enable_debug_info, option.enable_fast_math,
            fallback_rtx, /*bf16_fallback=*/!supports_native_bfloat16(),
            spirv_capabilities);
    }();
    if (fallback_rtx &&
        luisa::compute::detail::env_flag("LUISA_VK_DUMP_FALLBACK_HLSL")) {
        LUISA_INFO("fallback HLSL:\n{}",
                   luisa::string_view{code.result.view().data(),
                                      code.result.view().size()});
    }
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(
                             code.result.data() + code.immutableHeaderSize),
                         code.result.size() - code.immutableHeaderSize});
    // The push-constant block of the HLSL route is the fixed dispatch block the
    // codegen always declares, plus one 32-bit word per fallback `accel`
    // argument (the tree's region base, written by the command encoder:
    // stream.cpp). The property table carries one `ConstantValue` marker per
    // word, so its marker count is what sizes the block.
    auto push_constant_marker_count = static_cast<uint32_t>(std::count_if(
        code.properties.begin(), code.properties.end(),
        [](const hlsl::Property &property) noexcept {
            return property.type == hlsl::ShaderVariableType::ConstantValue;
        }));
    auto push_constant_size = static_cast<uint32_t>(
        sizeof(IndirectDispatchPushConstants) +
        4u * (push_constant_marker_count > 0u ? push_constant_marker_count - 1u : 0u));
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
                 detail::ShaderCodegenDialect::HLSL_SPIRV,
             .required_subgroup_size = required_subgroup_size},
            hlsl::binding_to_arg(kernel.bound_arguments()),
            option.name,
            SerdeType::kByteCode,
            _binary_io,
            push_constant_size,
            option.enable_driver_optimization);
        if (cache_result.shader) {
            delete static_cast<ComputeShader *>(cache_result.shader);
            LUISA_VERBOSE("ComputeShader (HLSL compile-only) loaded from cache.");
            info.block_size = kernel.block_size();
            return info;
        }
        auto comp_result = compile_hlsl_compute_to_spirv(
            Device::compiler(),
            code.result.view(),
            !option.enable_debug_info,
            shader_model,
            option.enable_fast_math,
            option.enable_debug_info,
            /*entry_point=*/{},
            /*include_dirs=*/{},
            /*spirv_capabilities=*/spirv_capabilities,
            option.name);
        if (comp_result.is_type_of<vstd::string>()) {
            LUISA_WARNING("DXC compile error for Vulkan shader '{}': {}",
                          option.name, comp_result.get<1>());
            info.block_size = kernel.block_size();
            info.compile_ok = false;
            return info;
        }
        auto &buffer = comp_result.get<0>();
        {
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
                required_subgroup_size,
                vk_device_detail::conservative_spirv_artifact_requirements(
                    this, requires_sampler_anisotropy));
        }
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
        auto dump_name_view = !option.name.empty() ? vstd::string_view{option.name} : vstd::string_view{kernel.name()};
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
            spirv_capabilities,
            validation_count,
            required_subgroup_size,
            requires_sampler_anisotropy,
            push_constant_size,
            detail::ShaderCodegenDialect::HLSL_SPIRV,
            option.enable_driver_optimization,
            dump_name_view);
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
//
// Compilation of one compute shader is split in two phases so that a batch of
// shaders can be prepared on the device's worker pool and published in request
// order (plan P5.1, `shader_compile_pool.h`). A serial `create_shader` is exactly
// `publish(prepare(...))` on the calling thread, with no pool involved, so it
// behaves precisely as it did before.
detail::PreparedComputeShader Device::_prepare_compute_shader(
    const ShaderOption &option, Function kernel,
    detail::WorkerContext *worker) noexcept {
    // The native XIR-to-SPIR-V route needs no per-worker compiler state: the XIR
    // translation, the SPIRV-Tools optimizer and validator, and the artifact codec
    // are all built per call and share no mutable state. Only the compatibility
    // HLSL-to-SPIR-V route would need one, and that route stays on the serial path.
    static_cast<void>(worker);
    detail::PreparedComputeShader prepared;
#ifdef LUISA_XIR_TO_SPIRV
    constexpr auto native_xir_spirv_compiled = true;
#else
    constexpr auto native_xir_spirv_compiled = false;
#endif
    auto require_native = vk_device_detail::require_native_xir_spirv();
    auto builtin_calls = kernel.propagated_builtin_callables();
    auto requires_motion_blur =
        builtin_calls.uses_raytracing_motion_blur();
    // The native XIR-to-SPIR-V route cannot answer the `accel.size()` query
    // (CallOp::ACCEL_SIZE); the compatibility HLSL route reads its result from
    // the host-injected validation slot of the argument-block trailer.
    auto requires_accel_size = builtin_calls.test(CallOp::ACCEL_SIZE);
    // The device traces in software *and* this kernel uses the ray-tracing call
    // ops: a software traversal is HLSL text
    // (src/backends/common/hlsl/builtin/fallback_rtx_header.bytes), so the
    // shader has to take the compatibility HLSL route - the native
    // XIR-to-SPIR-V route can only emit the hardware query/trace operations.
    // Nothing here changes for a device whose fallback is off.
    auto uses_raytracing_ops = builtin_calls.uses_raytracing();
    // Instance updates (set transform/visibility/opacity/user id) do not
    // count as ray-tracing ops, but on a fallback device they touch the
    // same fallback instance buffer: the shader has to be compiled
    // against the fallback acceleration-structure ABI as well, otherwise
    // its property list declares the native SPIR-V accel views and the
    // dispatch finds no fallback descriptor for the argument.
    auto uses_accel_instance_update =
        builtin_calls.test(CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM) ||
        builtin_calls.test(CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY) ||
        builtin_calls.test(CallOp::RAY_TRACING_SET_INSTANCE_OPACITY) ||
        builtin_calls.test(CallOp::RAY_TRACING_SET_INSTANCE_USER_ID);
    auto requires_fallback_rtx_traversal =
        use_fallback_rtx_bit &&
        (uses_raytracing_ops || uses_accel_instance_update);
    // BF16 has no software form in the native XIR-to-SPIR-V route (a BFloat16
    // type needs the device's shaderBFloat16Type), so a kernel that mentions
    // BF16 on a device without that feature takes the compatibility HLSL route,
    // where builtin/bfloat16_fallback.bytes emulates it.
    auto requires_bfloat16_fallback =
        !supports_native_bfloat16() &&
        detail::function_uses_bfloat16(kernel);
    detail::UserComputeCodegenRequirements codegen_requirements{
        .native_include = !option.native_include.empty(),
        .printing = kernel.requires_printing(),
        .async_copy = builtin_calls.test(CallOp::ASYNC_COPY) ||
                      builtin_calls.test(CallOp::PIPELINE_COMMIT) ||
                      builtin_calls.test(CallOp::PIPELINE_WAIT_PRIOR),
        .motion_blur = requires_motion_blur,
        .accel_size = requires_accel_size,
        .fallback_rtx = requires_fallback_rtx_traversal,
        .bfloat16 = requires_bfloat16_fallback,
        // The cluster ID of a ray-query intersection is
        // OpRayQueryGetIntersectionClusterIdNV
        // (SPV_NV_cluster_acceleration_structure), which the native
        // XIR-to-SPIR-V route does not emit, so a kernel that reads it
        // takes the compatibility HLSL route.
        .ray_query_cluster_id =
            builtin_calls.test(CallOp::RAY_QUERY_CLUSTER_ID)};
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
        fallback_reasons = vk_device_detail::describe_hlsl_fallback_reasons(codegen_route);
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
        vk_device_detail::validate_sampler_anisotropy_requirement(
            kernel, option.native_include,
            sampler_anisotropy_enabled);
    if (requires_fallback_rtx_traversal) {
        // What the software traversal does not implement is refused here, with
        // the reason and the way out, instead of being compiled into
        // something that renders differently (see the design brief's
        // "Out (must fail closed)" list).
        if (builtin_calls.uses_ray_query()) {
            LUISA_ERROR(
                "Vulkan shader '{}' uses a ray query (RAY_TRACING_QUERY_*), "
                "which the software ray-tracing fallback of this device does "
                "not implement "
                "(VulkanDeviceConfigExt::use_fallback_rtx()). Use "
                "Accel::intersect()/Accel::intersect_any() on this device, or "
                "run without the fallback.",
                kernel.name());
        }
        if (builtin_calls.uses_raytracing_motion_blur()) {
            LUISA_ERROR(
                "Vulkan shader '{}' uses motion-blur ray tracing, which the "
                "software ray-tracing fallback of this device does not "
                "implement "
                "(VulkanDeviceConfigExt::use_fallback_rtx()). Run without the "
                "fallback.",
                kernel.name());
        }
        LUISA_ASSERT(
            builtin_calls.test(CallOp::RAY_TRACING_TRACE_CLOSEST) ||
                builtin_calls.test(CallOp::RAY_TRACING_TRACE_ANY) ||
                uses_accel_instance_update,
            "Vulkan shader '{}' was routed to the software ray-tracing "
            "fallback without a trace or instance-update operation.",
            kernel.name());
    }
    if (requires_motion_blur && !motion_blur_enabled) {
        LUISA_ERROR("Vulkan device does not support VK_NV_ray_tracing_motion_blur; "
                    "motion-time compute tracing cannot be compiled.");
    }
#ifdef LUISA_XIR_TO_SPIRV
    if (codegen_route.requires_hlsl_fallback()) {
        // Fail closed to the unchanged serial path, on the thread that publishes:
        // the compatibility route is compiled by the process-global DXC instance,
        // whose thread contract this backend does not own.
        LUISA_VERBOSE(
            "Vulkan shader '{}' requires the HLSL-to-SPIR-V fallback for: "
            "{}.",
            kernel.name(), fallback_reasons);
        prepared.kind = detail::PreparedComputeShader::Kind::SERIAL_FALLBACK;
        prepared.fallback_requires_sampler_anisotropy = requires_sampler_anisotropy;
        prepared.fallback_rtx = requires_fallback_rtx_traversal;
        return prepared;
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
        luisa::compute::detail::env_flag("LUISA_VULKAN_PROFILE_COMPILATION");
    auto required_subgroup_size =
        effective_required_subgroup_size(kernel, option.compile_only);
    auto shader_md5 = vk_device_detail_2::compute_shader_cache_md5(
        kernel, option, target_features, required_subgroup_size);
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
    auto artifact_requirements = [&]() noexcept {
        return ComputeShaderArtifactLoadRequirements{
            .shader_md5 = shader_md5,
            .type_md5 = type_md5,
            .codegen_dialect = detail::ShaderCodegenDialect::XIR_SPIRV,
            .required_subgroup_size = required_subgroup_size};
    };
    // Reads and writes of the shader cache are serialized device-wide (plan P5.1):
    // a `BinaryIO` is one resource that a user implementation need not make
    // thread-safe, and one writer keeps the artifact sequence deterministic. Only
    // the copy of the bytes happens under the lock - decoding, validating and code
    // generation all run outside it, so a slow kernel cannot serialize the cache.
    auto read_cached_artifact = [&] {
        std::lock_guard cache_lock{_shader_cache_mtx};
        return ShaderSerializer::read_shader_artifact(
            serde_type, _binary_io, shader_name);
    };
    if (require_print_code || !use_binary_io ||
        ShaderSerializer::require_recompile_from_bytes(
            read_cached_artifact(), artifact_requirements())) {
        spv_result = compile_native_spirv();
    }

    if (!spv_result && !option.compile_only && use_binary_io) {
        ShaderSerializer::ComputeShaderArtifactLoad load;
        if (ShaderSerializer::load_compute_artifact_from_bytes(
                this, artifact_requirements(), read_cached_artifact(),
                shader_name, 32u, option.enable_driver_optimization, load)) {
            prepared.kind = detail::PreparedComputeShader::Kind::READY;
            prepared.from_cache = true;
            prepared.block_size = load.block_size;
            prepared.reported_block_size = load.block_size;
            prepared.properties = std::move(load.properties);
            prepared.saved_arguments = std::move(load.arguments);
            prepared.captured = hlsl::binding_to_arg(kernel.bound_arguments());
            prepared.spirv = std::move(load.spirv);
            prepared.use_tex2d_bindless = load.use_tex2d_bindless;
            prepared.use_tex3d_bindless = load.use_tex3d_bindless;
            prepared.use_buffer_bindless = load.use_buffer_bindless;
            prepared.printers = std::move(load.printers);
            prepared.constant_ubo_data = std::move(load.constant_ubo_data);
            prepared.validation_count = load.validation_count;
            prepared.required_subgroup_size = load.required_subgroup_size;
            prepared.push_constant_size = load.push_constant_size;
            prepared.codegen_dialect = load.codegen_dialect;
            prepared.enable_driver_optimization = option.enable_driver_optimization;
            prepared.artifact_name = shader_name;
            prepared.pso_seed_name = std::move(load.pso_seed_name);
            return prepared;
        }
        spv_result = compile_native_spirv();
    }

    if (spv_result) {
        auto artifact_requirements =
            vk_device_detail::validated_spirv_artifact_requirements(
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
        auto saved_args = ShaderSerializer::serialize_saved_args(
            luisa::span{spv_result->argument_usages}, true,
            luisa::span{spv_result->argument_roles});
        if (use_binary_io) {
            Clock serialization_clock;
            prepared.artifact_bytes = ShaderSerializer::encode_bytecode_artifact(
                spv_result->properties,
                saved_args,
                shader_md5,
                type_md5,
                kernel.block_size(),
                spv_result->spv_bin,
                spv_result->useTex2DBindless,
                spv_result->useTex3DBindless,
                spv_result->useBufferBindless,
                spv_result->printers,
                0,
                spv_result->constant_ubo_data,
                required_subgroup_size,
                artifact_requirements,
                detail::ShaderCodegenDialect::XIR_SPIRV);
            prepared.write_artifact = true;
            prepared.artifact_name = shader_name;
            prepared.artifact_type = serde_type;
            if (profile) {
                LUISA_INFO(
                    "Vulkan native shader-artifact serialization: {:.3f} ms",
                    serialization_clock.toc());
            }
        }
        prepared.properties = std::move(spv_result->properties);
        prepared.saved_arguments = std::move(saved_args);
        prepared.captured = hlsl::binding_to_arg(kernel.bound_arguments());
        prepared.spirv = std::move(spv_result->spv_bin);
        prepared.use_tex2d_bindless = spv_result->useTex2DBindless;
        prepared.use_tex3d_bindless = spv_result->useTex3DBindless;
        prepared.use_buffer_bindless = spv_result->useBufferBindless;
        prepared.printers = std::move(spv_result->printers);
        prepared.constant_ubo_data = std::move(spv_result->constant_ubo_data);
        prepared.validation_count = 0u;
        prepared.required_subgroup_size = required_subgroup_size;
        prepared.push_constant_size = 32u;
        prepared.codegen_dialect = detail::ShaderCodegenDialect::XIR_SPIRV;
        prepared.enable_driver_optimization = option.enable_driver_optimization;
        prepared.reported_block_size = kernel.block_size();
        if (option.compile_only) {
            assert(!option.name.empty());
            // Nothing to construct: the artifact above is the whole answer, and the
            // publishing phase writes it.
            prepared.kind = detail::PreparedComputeShader::Kind::READY;
            prepared.construct_shader = false;
            return prepared;
        }
        prepared.kind = detail::PreparedComputeShader::Kind::READY;
        prepared.construct_shader = true;
        prepared.block_size = kernel.block_size();
        if (profile) {
            // The construction cost of a native shader is reported by
            // `_publish_compute_shader` (it happens there now).
            LUISA_INFO("Vulkan native shader prepared: {} word(s) SPIR-V, {} bind(s)",
                       prepared.spirv.size(), prepared.properties.size());
        }
        return prepared;
    }
    LUISA_ASSERT(spv_result, "Vulkan SPIR-V cache load failed without recompilation.");
    return prepared;
#else
    // This build has no native XIR-to-SPIR-V route, so every shader takes the
    // serial path (LLVM-to-SPIR-V or HLSL-to-SPIR-V); see `_create_shader_serial`.
    prepared.kind = detail::PreparedComputeShader::Kind::SERIAL_FALLBACK;
    prepared.fallback_requires_sampler_anisotropy = requires_sampler_anisotropy;
    prepared.fallback_rtx = requires_fallback_rtx_traversal;
    return prepared;
#endif
}

// The unchanged serial compilation routes: the compatibility HLSL-to-SPIR-V
// compiler, the LLVM-to-SPIR-V route, and every build without the native
// XIR-to-SPIR-V route. Reached by a `SERIAL_FALLBACK` prepared record, i.e. on the
// thread that publishes, exactly like the pre-P5.1 code path.
ShaderCreationInfo Device::_create_shader_serial(
    const ShaderOption &option, Function kernel,
    bool requires_sampler_anisotropy, bool requires_fallback_rtx_traversal) noexcept {
#ifdef LUISA_AST_LLVM_TO_SPIRV
    ShaderCreationInfo info;
    // === AST LLVM to SPIR-V codegen path ===
    if (requires_fallback_rtx_traversal) {
        LUISA_ERROR(
            "Vulkan shader '{}' traces rays through the software fallback "
            "acceleration structure "
            "(VulkanDeviceConfigExt::use_fallback_rtx()), but this Vulkan "
            "backend was built with the experimental AST-LLVM SPIR-V codegen "
            "route, which has no software traversal.",
            kernel.name());
    }
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
            vk_device_detail::conservative_spirv_artifact_requirements(
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
    info.block_size = kernel.block_size();
    return info;
#else
    return _create_shader_hlsl(
        option, kernel, requires_sampler_anisotropy,
        requires_fallback_rtx_traversal);
#endif
}

// Publication: the only phase that creates Vulkan objects (plan P5.1).
ShaderCreationInfo Device::_publish_compute_shader(
    detail::PreparedComputeShader &&prepared,
    const ShaderOption &option, Function kernel) noexcept {
    if (prepared.kind == detail::PreparedComputeShader::Kind::SERIAL_FALLBACK) {
        // Serial fallback: the unchanged pre-P5.1 path, on this thread. It reads and
        // writes the shader cache without knowing about the pool, so it runs under
        // the device-wide cache lock - a batch that mixes this route with native
        // requests therefore serializes its cache traffic instead of racing it. A
        // batch of only this route is exactly today's serial loop.
        std::lock_guard cache_lock{_shader_cache_mtx};
        return _create_shader_serial(
            option, kernel, prepared.fallback_requires_sampler_anisotropy,
            prepared.fallback_rtx);
    }
    ShaderCreationInfo info;
    if (prepared.kind != detail::PreparedComputeShader::Kind::READY) {
        info.invalidate();
        info.block_size = prepared.reported_block_size;
        return info;
    }
    // One construction at a time, device-wide. This is the ordering point of a
    // batch (`compile_shaders` publishes sequence number 0, 1, 2, ... from one
    // thread) and the safety point for a caller that compiles from several threads:
    // the interned layout table, the device pipeline cache's write path and the
    // driver's pipeline compiler see one creation at a time.
    std::lock_guard construct_lock{_shader_construct_mtx};
    // Pre-P1.3 per-shader PSO blobs are add-only inputs of the device cache, and
    // merging one is a Vulkan call: it happens here, not on a worker.
    if (!prepared.pso_seed_name.empty()) {
        std::lock_guard cache_lock{_shader_cache_mtx};
        ShaderSerializer::merge_legacy_pso_seed(
            this, prepared.artifact_name, prepared.pso_seed_name);
    }
    if (prepared.write_artifact) {
        std::lock_guard cache_lock{_shader_cache_mtx};
        ShaderSerializer::write_bytecode_artifact(
            prepared.artifact_type, _binary_io,
            prepared.artifact_name, prepared.artifact_bytes);
    }
    if (!prepared.construct_shader) {
        info.invalidate();
        info.block_size = prepared.reported_block_size;
        return info;
    }
    auto profile =
        luisa::compute::detail::env_flag("LUISA_VULKAN_PROFILE_COMPILATION");
    Clock pipeline_clock;
    auto shader = new ComputeShader(
        this,
        prepared.block_size,
        prepared.properties,
        std::move(prepared.saved_arguments),
        {reinterpret_cast<const uint *>(prepared.spirv.data()), prepared.spirv.size()},
        std::move(prepared.captured),
        prepared.use_tex2d_bindless,
        prepared.use_tex3d_bindless,
        prepared.use_buffer_bindless,
        std::move(prepared.printers),
        {prepared.constant_ubo_data.data(), prepared.constant_ubo_data.size()},
        prepared.validation_count,
        prepared.required_subgroup_size,
        prepared.push_constant_size,
        prepared.codegen_dialect,
        prepared.enable_driver_optimization);
    if (profile) {
        LUISA_INFO(
            "Vulkan native ComputeShader construction: {:.3f} ms",
            pipeline_clock.toc());
    }
    LUISA_VERBOSE("ComputeShader created successfully, pipeline: {}",
                  reinterpret_cast<void *>(shader->pipeline()));
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = shader->pipeline();
    info.block_size = prepared.reported_block_size;
    return info;
}

ShaderCreationInfo Device::create_shader(const ShaderOption &option, Function kernel) noexcept {
    // A serial compile is the two phases back to back on the calling thread: no
    // pool, no ticket, no behaviour change.
    auto prepared = _prepare_compute_shader(option, kernel, nullptr);
    return _publish_compute_shader(std::move(prepared), option, kernel);
}

ShaderCreationInfo Device::load_shader(luisa::string_view name, luisa::span<const luisa::compute::Type *const> arg_types) noexcept {
    ShaderCreationInfo info;
    luisa::optional<detail::ShaderCodegenDialect> required_dialect;
    if (vk_device_detail::require_native_xir_spirv()) {
        required_dialect = detail::ShaderCodegenDialect::XIR_SPIRV;
    }
    auto type_md5 = hlsl::CodegenUtility::GetTypeMD5(arg_types);
    // AOT load: same device-wide shader-cache lock as every other reader/writer
    // (plan P5.1), so it cannot interleave with a compile batch.
    std::lock_guard cache_lock{_shader_cache_mtx};
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
    if (auto *custom = find_custom_stream(stream_handle)) {
        custom->signal(reinterpret_cast<Event *>(handle), fence_value);
        return;
    }
    reinterpret_cast<Stream *>(stream_handle)->signal(reinterpret_cast<Event *>(handle), fence_value);
}
void Device::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    if (auto *custom = find_custom_stream(stream_handle)) {
        custom->wait(reinterpret_cast<Event *>(handle), fence_value);
        return;
    }
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
    std::lock_guard dispatch_lock{detail::vk_device_detail_3::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::vk_device_detail_3::live_device_count == 0u,
        "Cannot enumerate Vulkan devices while a Vulkan Device is live; "
        "Volk uses process-global dispatch tables.");
    std::lock_guard lck{detail::vk_device_detail_3::instance_mtx};
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
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(detail::vk_device_detail_3::vk_instance, &gpu_count, nullptr));
    if (gpu_count == 0) {
        return;
    }
    // Enumerate devices
    luisa::enlarge_by(physical_devices, gpu_count);
    auto err = vkEnumeratePhysicalDevices(detail::vk_device_detail_3::vk_instance, &gpu_count, physical_devices.data());
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
    std::lock_guard lck(vk_device_detail_2::g_dxc_mutex);
    if (!vk_device_detail_2::g_dxc_compiler_initialized) {
        if (vk_device_detail_2::g_dxc_runtime_directory.empty()) [[unlikely]] {
            LUISA_ERROR("Vulkan internal HLSL compiler requested before device initialization.");
        }
        vk_device_detail_2::g_dxc_compiler.create(vk_device_detail_2::g_dxc_runtime_directory, true);
        vk_device_detail_2::g_dxc_compiler_initialized = true;
    }
    return vk_device_detail_2::g_dxc_compiler.ptr();
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
    std::lock_guard dispatch_lock{detail::vk_device_detail_3::dispatch_lifetime_mtx};
    LUISA_ASSERT(
        detail::vk_device_detail_3::live_device_count == 0u,
        "Cannot initialize the process Vulkan instance while a Vulkan Device "
        "is live; Volk uses process-global dispatch tables.");
    std::lock_guard lck{detail::vk_device_detail_3::instance_mtx};
    enable_validation |= detail::validation_enabled_by_default();
    // custom_vk_lib_path is application-supplied narrow text: decode it with
    // path_from_narrow() so an unrepresentable path simply means "no custom
    // library" instead of throwing inside path's constructor.
    luisa::filesystem::path custom_lib_path;
    if (custom_vk_lib_path != nullptr &&
        !luisa::path_from_narrow(luisa::string_view{custom_vk_lib_path}, custom_lib_path)) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Custom Vulkan library path '{}' cannot be represented in the ANSI code page.",
            custom_vk_lib_path);
        custom_lib_path.clear();
    }
    detail::load_or_create_process_instance(enable_validation, enable_surface, custom_lib_path, custom_vk_lib_name ? luisa::string_view{custom_vk_lib_name} : luisa::string_view{}, luisa::span{extra_instance_exts, extra_instance_ext_count});
    return detail::vk_device_detail_3::vk_instance;
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
void Device::LazyLoadShader::reset() noexcept { _shader = nullptr; }
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
luisa::span<const vstd::string>
Device::enabled_instance_extensions() const noexcept {
    // A borrowed instance keeps an empty record: the backend did not create it
    // and cannot attest its enabled extension list.
    auto &recorded = detail::vk_device_detail_3::vk_instance_enabled_extensions;
    return luisa::span<const vstd::string>{recorded.data(), recorded.size()};
}

DeviceExtension *Device::extension(vstd::string_view name) noexcept {
    if (name == CommandReorderExt::name) {
        return &_command_reorder_ext;
    }
    if (name == VulkanShaderCompileExt::name) {
        // Plan P5.1: like the replay engine, the extension has no per-device
        // allocation of its own (the pool lives on the Device), so there is no
        // factory to run and no lifetime for a caller to manage.
        return &_shader_compile_ext;
    }
    if (name == VulkanReplayExt::name) {
        // Plan P4: the engine has no per-device allocation (every graph lives in
        // `_replay_registry`), so there is no factory to run and no lifetime for
        // the caller to manage.
        return &_replay_ext;
    }
    if (name == VulkanExtensionStatusExt::name) {
        return &_optional_extension_status;
    }
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
    if (auto *custom = find_custom_stream(stream_handle)) {
        custom->set_log_callback(callback);
        return;
    }
    reinterpret_cast<Stream *>(stream_handle)->logger = callback;
}
namespace {
// Per-name answer for the two cooperative families, which the backend negotiates
// outside the optional-extension registry (see VulkanCooperativeStatus).
// `known == false` means the name is neither of them and the caller must fall
// back to the registry lookup.
//
// Semantics, deliberately narrower than the registry's:
//   supported/enabled -> this exact extension name is the one the backend
//                        enabled, i.e. one the physical device attested. A twin
//                        the backend did not pick reports false (not attested),
//                        never "the device lacks it" -- cooperative_status()
//                        reports which twin was picked.
//   active            -> the backend routes work through it. False for every
//                        cooperative-matrix name on purpose: nothing in this tree
//                        lowers to a cooperative-matrix instruction.
struct CooperativeNameAnswer {
    bool known{false};
    bool supported{false};
    bool enabled{false};
    bool active{false};
};

[[nodiscard]] CooperativeNameAnswer cooperative_name_answer(
    luisa::string_view name,
    const char *matrix_extension /* nullptr = none */,
    bool matrix_workgroup_scope,
    bool vector_physical_support,
    bool vector_enabled) noexcept {
    CooperativeNameAnswer answer{.known = true};
    if (name == luisa::string_view{"VK_NV_cooperative_vector"}) {
        answer.supported = vector_physical_support;
        answer.enabled = vector_enabled;
        // The SPIR-V target feature is enabled exactly when the extension is,
        // so the codegen does route cooperative-vector operations to it.
        answer.active = vector_enabled;
        return answer;
    }
    if (name == luisa::string_view{"VK_NV_cooperative_matrix"}) {
        answer.supported = matrix_extension != nullptr &&
                           luisa::string_view{matrix_extension} ==
                               luisa::string_view{"VK_NV_cooperative_matrix"};
        answer.enabled = answer.supported;
        return answer;
    }
    if (name == luisa::string_view{"VK_KHR_cooperative_matrix"}) {
        // The KHR spelling is also what carries the NV2 workgroup-scope feature.
        answer.supported = matrix_extension != nullptr;
        answer.enabled = answer.supported;
        return answer;
    }
    if (name == luisa::string_view{"VK_NV_cooperative_matrix2"}) {
        answer.supported = matrix_extension != nullptr && matrix_workgroup_scope;
        answer.enabled = answer.supported;
        return answer;
    }
    answer.known = false;
    return answer;
}
}// namespace

void Device::VulkanExtensionStatusExtImpl::build() noexcept {
    if (!_cache.empty()) { return; }
    auto states = _device->optional_extensions();
    _cache.reserve(states.size());
    for (auto const &state : states) {
        VulkanExtensionStatus status{};
        status.name = state.name == nullptr ? luisa::string{"unknown"} :
                                              luisa::string{state.name};
        status.supported = state.supported_by_device;
        status.requested = state.requested_by_config;
        status.enabled = state.enabled;
        status.active = state.enabled &&
                        _device->optional_extension_active(state.id);
        _cache.emplace_back(std::move(status));
    }
}

luisa::span<const VulkanExtensionStatus>
Device::VulkanExtensionStatusExtImpl::extensions() const noexcept {
    auto self = const_cast<VulkanExtensionStatusExtImpl *>(this);
    self->build();
    return luisa::span<const VulkanExtensionStatus>{self->_cache.data(),
                                                   self->_cache.size()};
}

bool Device::VulkanExtensionStatusExtImpl::is_supported(
    luisa::string_view extension) const noexcept {
    // The cooperative families are negotiated outside the registry; answer for
    // them from the recorded negotiation instead of reporting "unknown name".
    auto cooperative = cooperative_name_answer(
        extension, _device->cooperative_matrix_extension,
        _device->cooperative_matrix_workgroup_scope,
        _device->cooperative_vector_physical_support,
        _device->cooperative_vector_enabled);
    if (cooperative.known) { return cooperative.supported; }
    auto id = detail::optional_extension_from_name(extension);
    return id != detail::OptionalExtension::COUNT &&
           _device->_optional_extensions.supported(id);
}

bool Device::VulkanExtensionStatusExtImpl::is_enabled(
    luisa::string_view extension) const noexcept {
    auto cooperative = cooperative_name_answer(
        extension, _device->cooperative_matrix_extension,
        _device->cooperative_matrix_workgroup_scope,
        _device->cooperative_vector_physical_support,
        _device->cooperative_vector_enabled);
    if (cooperative.known) { return cooperative.enabled; }
    auto id = detail::optional_extension_from_name(extension);
    return id != detail::OptionalExtension::COUNT &&
           _device->_optional_extensions.enabled(id);
}

bool Device::VulkanExtensionStatusExtImpl::is_active(
    luisa::string_view extension) const noexcept {
    auto cooperative = cooperative_name_answer(
        extension, _device->cooperative_matrix_extension,
        _device->cooperative_matrix_workgroup_scope,
        _device->cooperative_vector_physical_support,
        _device->cooperative_vector_enabled);
    if (cooperative.known) { return cooperative.active; }
    auto id = detail::optional_extension_from_name(extension);
    return id != detail::OptionalExtension::COUNT &&
           _device->_optional_extensions.enabled(id) &&
           _device->optional_extension_active(id);
}

VulkanCooperativeStatus
Device::VulkanExtensionStatusExtImpl::cooperative_status() const noexcept {
    VulkanCooperativeStatus status{};
    auto const &device = *_device;
    if (device.cooperative_matrix_extension != nullptr) {
        status.matrix_extension = luisa::string{device.cooperative_matrix_extension};
        status.matrix_enabled = true;
        status.matrix_workgroup_scope = device.cooperative_matrix_workgroup_scope;
    }
    status.vector_physical_support = device.cooperative_vector_physical_support;
    status.vector_memory_model = device.cooperative_vector_memory_model;
    status.vector_replicated_composites =
        device.cooperative_vector_replicated_composites;
    status.vector_enabled = device.cooperative_vector_enabled;
    status.vector_fp32_config = device.cooperative_vector_fp32_enabled;
    status.vector_config_count = device.cooperative_vector_config_count;
    return status;
}
VulkanCapabilityStatus
Device::VulkanExtensionStatusExtImpl::capability_status() const noexcept {
    // Plan P6.3. The limit fields and the SM-count answer are a copy of the one
    // capability record the device captured at creation - the same record
    // `Device::query` answers from - so a value read here and a value read
    // through the matching query key are the same value by construction, not by
    // two mappings that happen to agree.
    auto status = _device->capability_status();
    // The descriptor-heap state is the registry entry's negotiation state plus
    // the backend's own active bit, exactly like `extensions()` reports it. It is
    // read here instead of cached so it cannot go stale against the registry.
    auto heap = detail::OptionalExtension::DESCRIPTOR_HEAP;
    status.descriptor_heap.supported = _device->_optional_extensions.supported(heap);
    status.descriptor_heap.requested = _device->_optional_extensions.requested(heap);
    status.descriptor_heap.enabled = _device->_optional_extensions.enabled(heap);
    status.descriptor_heap.active =
        status.descriptor_heap.enabled && _device->optional_extension_active(heap);
    return status;
}

}// namespace lc::vk


#include "../common/export_version.inl.h"
