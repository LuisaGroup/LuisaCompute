#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace lc::vk::detail {

// The three global update-after-bind heaps contain C storage buffers, C 2D
// sampled images, and C 3D sampled images. Vulkan's update-after-bind pipeline
// layout limits count descriptors in ordinary layouts as well as descriptors
// in update-after-bind layouts. Reserve one conservative budget for ordinary
// buffer/image descriptors. The fixed sampler layout is accounted separately:
// standalone samplers do not consume maxPerStageUpdateAfterBindResources.
//
// The DSL currently caps a callable at 64 arguments; 256 leaves room for
// multi-descriptor arguments and backend-internal bindings without materially
// reducing the requested global heaps on conformant descriptor-indexing GPUs.
inline constexpr uint32_t requested_bindless_heap_capacity = 262144u;
inline constexpr uint32_t local_per_stage_descriptor_budget = 256u;
inline constexpr uint32_t fixed_sampler_descriptor_count = 16u;

struct BindlessHeapLimits {
    uint32_t max_per_stage_update_after_bind_samplers{};
    uint32_t max_descriptor_set_update_after_bind_samplers{};
    uint32_t max_per_stage_update_after_bind_storage_buffers{};
    uint32_t max_descriptor_set_update_after_bind_storage_buffers{};
    uint32_t max_per_stage_update_after_bind_sampled_images{};
    uint32_t max_descriptor_set_update_after_bind_sampled_images{};
    uint32_t max_per_stage_update_after_bind_resources{};
    uint32_t max_update_after_bind_descriptors_in_all_pools{};
};

[[nodiscard]] constexpr uint32_t plan_bindless_heap_capacity(
    BindlessHeapLimits limits,
    uint32_t requested_capacity = requested_bindless_heap_capacity,
    uint32_t local_descriptor_budget =
        local_per_stage_descriptor_budget) noexcept {
    auto reserve_local_descriptors =
        [local_descriptor_budget](uint32_t limit) noexcept {
            return limit > local_descriptor_budget ?
                       limit - local_descriptor_budget :
                       0u;
        };
    if (limits.max_per_stage_update_after_bind_samplers <
            fixed_sampler_descriptor_count ||
        limits.max_descriptor_set_update_after_bind_samplers <
            fixed_sampler_descriptor_count) {
        return 0u;
    }
    // Both update-after-bind limit families are pipeline-layout aggregates:
    // local storage + C, local sampled + 2C, and local resources + 3C. The
    // all-pools limit covers only the three persistent update-after-bind pools.
    return std::min({requested_capacity,
                     reserve_local_descriptors(
                         limits.max_per_stage_update_after_bind_storage_buffers),
                     reserve_local_descriptors(
                         limits.max_descriptor_set_update_after_bind_storage_buffers),
                     reserve_local_descriptors(
                         limits.max_per_stage_update_after_bind_sampled_images) /
                         2u,
                     reserve_local_descriptors(
                         limits.max_descriptor_set_update_after_bind_sampled_images) /
                         2u,
                     reserve_local_descriptors(
                         limits.max_per_stage_update_after_bind_resources) /
                         3u,
                     limits.max_update_after_bind_descriptors_in_all_pools / 3u});
}

// robustBufferAccess and storage-buffer update-after-bind can only be enabled
// together when the descriptor-indexing properties explicitly permit that
// combination. The backend does not rely on robustBufferAccess, so prefer the
// bindless ABI and disable robustness when the two features conflict.
struct RobustBufferAccessSupport {
    bool physical_device_feature{false};
    bool storage_buffer_update_after_bind{false};
    bool robust_buffer_access_update_after_bind{false};
};

[[nodiscard]] constexpr bool plan_robust_buffer_access(
    RobustBufferAccessSupport support) noexcept {
    return support.physical_device_feature &&
           (!support.storage_buffer_update_after_bind ||
            support.robust_buffer_access_update_after_bind);
}

// Formatless storage-image access is optional in Vulkan. Preserve the exact
// physical-device support bits for backend-owned devices; never fabricate
// support, and do not advertise physical support as enabled for an imported
// logical device whose enabled VkPhysicalDeviceFeatures are unknowable.
struct StorageImageFormatFeatureSupport {
    bool read_without_format{false};
    bool write_without_format{false};
    bool imported_device{false};
};

struct StorageImageFormatFeaturePlan {
    bool read_without_format{false};
    bool write_without_format{false};
};

[[nodiscard]] constexpr auto plan_storage_image_format_features(
    StorageImageFormatFeatureSupport support) noexcept {
    if (support.imported_device) {
        return StorageImageFormatFeaturePlan{};
    }
    return StorageImageFormatFeaturePlan{
        .read_without_format = support.read_without_format,
        .write_without_format = support.write_without_format};
}

// device_feature_settings() may extend the logical-device feature chain, but
// it must not provide structures whose state is owned by this backend. Besides
// exact duplicate sTypes, Vulkan forbids mixing a promoted aggregate structure
// with any of its alias feature structures in the same pNext chain. Reserve
// both sides of every aggregate/alias pair used by the backend so callers
// cannot accidentally create an invalid chain as backend features evolve.
inline constexpr VkStructureType backend_owned_device_feature_structure_types[]{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,

    // Aliases promoted into VkPhysicalDeviceVulkan11Features (VUID 02829).
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,

    // Aliases promoted into VkPhysicalDeviceVulkan12Features (VUID 02830).
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES,

    // The backend uses synchronization2/subgroup-size-control aliases and the
    // maintenance5 alias rather than the Vulkan 1.3/1.4 aggregate structures.
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
#ifdef VK_VERSION_1_4
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
#endif
};

[[nodiscard]] constexpr bool device_feature_structure_type_is_backend_owned(
    VkStructureType type) noexcept {
    for (auto owned : backend_owned_device_feature_structure_types) {
        if (type == owned) { return true; }
    }
    return false;
}

enum class DeviceFeatureChainValidationError : uint8_t {
    NONE,
    BACKEND_OWNED_STRUCTURE,
    DUPLICATE_STRUCTURE,
    CYCLE,
};

struct DeviceFeatureChainValidationResult {
    DeviceFeatureChainValidationError error{
        DeviceFeatureChainValidationError::NONE};
    VkStructureType structure_type{VK_STRUCTURE_TYPE_MAX_ENUM};
    size_t node_index{};
    size_t related_node_index{};

    [[nodiscard]] constexpr bool succeeded() const noexcept {
        return error == DeviceFeatureChainValidationError::NONE;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return succeeded();
    }
};

// This validates chain topology and sType ownership only. As with Vulkan
// itself, the caller remains responsible for every non-null node referring to
// a live structure whose first two members are sType and pNext.
[[nodiscard]] inline DeviceFeatureChainValidationResult
validate_device_feature_settings_chain(
    const VkBaseInStructure *head) noexcept {
    auto *slow = head;
    auto *fast = head;
    while (fast != nullptr && fast->pNext != nullptr) {
        slow = slow->pNext;
        fast = fast->pNext->pNext;
        if (slow == fast) {
            return {.error = DeviceFeatureChainValidationError::CYCLE};
        }
    }
    size_t node_index = 0u;
    for (auto *node = head; node != nullptr;
         node = node->pNext, ++node_index) {
        if (device_feature_structure_type_is_backend_owned(node->sType)) {
            return {
                .error = DeviceFeatureChainValidationError::BACKEND_OWNED_STRUCTURE,
                .structure_type = node->sType,
                .node_index = node_index};
        }
        size_t previous_index = 0u;
        for (auto *previous = head; previous != node;
             previous = previous->pNext, ++previous_index) {
            if (previous->sType == node->sType) {
                return {
                    .error = DeviceFeatureChainValidationError::DUPLICATE_STRUCTURE,
                    .structure_type = node->sType,
                    .node_index = node_index,
                    .related_node_index = previous_index};
            }
        }
    }
    return {};
}

struct RequiredDeviceFeatureSupport {
    bool timeline_semaphore_core{false};
    bool timeline_semaphore_extension{false};
    bool timeline_semaphore_feature{false};
    bool synchronization2_core{false};
    bool synchronization2_extension{false};
    bool synchronization2_feature{false};
    bool physical_device_api_1_3{false};
};

struct RequiredDeviceFeaturePlan {
    bool supported{false};
    bool enable_timeline_semaphore_extension{false};
    bool enable_synchronization2_extension{false};
};

struct InstanceRuntimeCapabilityPlan {
    bool surface{false};
    bool debug_utils{false};
};

// Enabled extensions cannot be queried from an existing VkInstance. Without
// explicit import attestations, borrowed instances must not enter extension-
// dependent surface or debug-utils paths.
[[nodiscard]] constexpr InstanceRuntimeCapabilityPlan
plan_instance_runtime_capabilities(
    bool borrowed_instance, bool request_surface,
    bool request_debug_utils) noexcept {
    if (borrowed_instance) { return {}; }
    return {.surface = request_surface,
            .debug_utils = request_debug_utils};
}

struct SparseResidencyFeatureSupport {
    bool sparse_binding{false};
    bool sparse_residency_buffer{false};
    bool sparse_residency_image_2d{false};
    bool sparse_residency_image_3d{false};
};

enum class SparseResidencyFeatureStatus : uint8_t {
    SUCCESS,
    MISSING_SPARSE_BINDING,
    MISSING_BUFFER_RESIDENCY,
    UNSUPPORTED_IMAGE_DIMENSION,
    MISSING_IMAGE_2D_RESIDENCY,
    MISSING_IMAGE_3D_RESIDENCY
};

struct SparseResidencyFeatureResult {
    SparseResidencyFeatureStatus status;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == SparseResidencyFeatureStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr const char *sparse_residency_feature_status_name(
    SparseResidencyFeatureStatus status) noexcept {
    switch (status) {
        case SparseResidencyFeatureStatus::SUCCESS: return "success";
        case SparseResidencyFeatureStatus::MISSING_SPARSE_BINDING: return "sparseBinding is not enabled";
        case SparseResidencyFeatureStatus::MISSING_BUFFER_RESIDENCY: return "sparseResidencyBuffer is not enabled";
        case SparseResidencyFeatureStatus::UNSUPPORTED_IMAGE_DIMENSION: return "sparse residency images must be 2D or 3D";
        case SparseResidencyFeatureStatus::MISSING_IMAGE_2D_RESIDENCY: return "sparseResidencyImage2D is not enabled";
        case SparseResidencyFeatureStatus::MISSING_IMAGE_3D_RESIDENCY: return "sparseResidencyImage3D is not enabled";
    }
    return "unknown sparse-residency feature error";
}

[[nodiscard]] constexpr SparseResidencyFeatureResult
validate_sparse_buffer_features(
    SparseResidencyFeatureSupport support) noexcept {
    if (!support.sparse_binding) {
        return {SparseResidencyFeatureStatus::MISSING_SPARSE_BINDING};
    }
    if (!support.sparse_residency_buffer) {
        return {SparseResidencyFeatureStatus::MISSING_BUFFER_RESIDENCY};
    }
    return {SparseResidencyFeatureStatus::SUCCESS};
}

[[nodiscard]] constexpr SparseResidencyFeatureResult
validate_sparse_texture_features(
    SparseResidencyFeatureSupport support, uint32_t dimension) noexcept {
    if (!support.sparse_binding) {
        return {SparseResidencyFeatureStatus::MISSING_SPARSE_BINDING};
    }
    if (dimension == 2u) {
        return {support.sparse_residency_image_2d ?
                    SparseResidencyFeatureStatus::SUCCESS :
                    SparseResidencyFeatureStatus::MISSING_IMAGE_2D_RESIDENCY};
    }
    if (dimension == 3u) {
        return {support.sparse_residency_image_3d ?
                    SparseResidencyFeatureStatus::SUCCESS :
                    SparseResidencyFeatureStatus::MISSING_IMAGE_3D_RESIDENCY};
    }
    // Vulkan forbids VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT on 1D images.
    return {SparseResidencyFeatureStatus::UNSUPPORTED_IMAGE_DIMENSION};
}

enum class ExternalRequiredFeatureStatus : uint8_t {
    SUCCESS,
    MISSING_TIMELINE_SEMAPHORE,
    MISSING_SYNCHRONIZATION2
};

enum class ExternalInstanceApiStatus : uint8_t {
    SUCCESS,
    MISSING_VERSION,
    UNSUPPORTED_VARIANT,
    VERSION_TOO_OLD
};

struct ExternalInstanceApiResult {
    ExternalInstanceApiStatus status;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == ExternalInstanceApiStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr const char *external_instance_api_status_name(
    ExternalInstanceApiStatus status) noexcept {
    switch (status) {
        case ExternalInstanceApiStatus::SUCCESS: return "success";
        case ExternalInstanceApiStatus::MISSING_VERSION: return "effective instance API version was not supplied";
        case ExternalInstanceApiStatus::UNSUPPORTED_VARIANT: return "effective instance API uses a non-Vulkan variant";
        case ExternalInstanceApiStatus::VERSION_TOO_OLD: return "effective instance API version is below Vulkan 1.3";
    }
    return "unknown imported-instance API error";
}

// vkGetPhysicalDeviceProperties reports the device API version, not the API
// version selected when a borrowed VkInstance was created. The latter bounds
// which core entry points the backend may use and must be attested separately.
[[nodiscard]] constexpr ExternalInstanceApiResult
validate_external_instance_api(
    bool imported_instance, uint32_t api_version) noexcept {
    if (!imported_instance) {
        return {ExternalInstanceApiStatus::SUCCESS};
    }
    if (api_version == 0u) {
        return {ExternalInstanceApiStatus::MISSING_VERSION};
    }
    if (VK_API_VERSION_VARIANT(api_version) != 0u) {
        return {ExternalInstanceApiStatus::UNSUPPORTED_VARIANT};
    }
    auto major = VK_API_VERSION_MAJOR(api_version);
    auto minor = VK_API_VERSION_MINOR(api_version);
    if (major < 1u || (major == 1u && minor < 3u)) {
        return {ExternalInstanceApiStatus::VERSION_TOO_OLD};
    }
    return {ExternalInstanceApiStatus::SUCCESS};
}

struct ExternalRequiredFeatureResult {
    ExternalRequiredFeatureStatus status;

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return status == ExternalRequiredFeatureStatus::SUCCESS;
    }
};

[[nodiscard]] constexpr const char *external_required_feature_status_name(
    ExternalRequiredFeatureStatus status) noexcept {
    switch (status) {
        case ExternalRequiredFeatureStatus::SUCCESS: return "success";
        case ExternalRequiredFeatureStatus::MISSING_TIMELINE_SEMAPHORE: return "timelineSemaphore was not attested as enabled";
        case ExternalRequiredFeatureStatus::MISSING_SYNCHRONIZATION2: return "synchronization2 was not attested as enabled";
    }
    return "unknown imported-device feature error";
}

// Vulkan has no query for the enabled feature chain of an existing VkDevice.
// Imported-device callers must therefore attest the mandatory feature bits.
[[nodiscard]] constexpr ExternalRequiredFeatureResult
validate_external_required_features(
    bool imported_device,
    bool timeline_semaphore_enabled,
    bool synchronization2_enabled) noexcept {
    if (!imported_device) {
        return {ExternalRequiredFeatureStatus::SUCCESS};
    }
    if (!timeline_semaphore_enabled) {
        return {
            ExternalRequiredFeatureStatus::MISSING_TIMELINE_SEMAPHORE};
    }
    if (!synchronization2_enabled) {
        return {ExternalRequiredFeatureStatus::MISSING_SYNCHRONIZATION2};
    }
    return {ExternalRequiredFeatureStatus::SUCCESS};
}

[[nodiscard]] constexpr auto plan_required_device_features(
    RequiredDeviceFeatureSupport support) noexcept {
    auto timeline_available =
        support.timeline_semaphore_core ||
        support.timeline_semaphore_extension;
    auto synchronization2_available =
        support.synchronization2_core ||
        support.synchronization2_extension;
    return RequiredDeviceFeaturePlan{
        .supported = timeline_available &&
                     support.timeline_semaphore_feature &&
                     synchronization2_available &&
                     support.synchronization2_feature &&
                     support.physical_device_api_1_3,
        .enable_timeline_semaphore_extension =
            support.physical_device_api_1_3 &&
            !support.timeline_semaphore_core &&
            support.timeline_semaphore_extension &&
            support.timeline_semaphore_feature,
        .enable_synchronization2_extension =
            support.physical_device_api_1_3 &&
            !support.synchronization2_core &&
            support.synchronization2_extension &&
            support.synchronization2_feature};
}

// Vulkan exposes arithmetic and storage support for narrow scalar widths as
// independent feature bits. In particular, storageBuffer16BitAccess does not
// imply shaderFloat16 or shaderInt16, and the three Vulkan 1.2 8-bit bits are
// independent as well. Keep this as an explicit plan so logical-device
// creation cannot accidentally turn a valid storage-only capability into an
// all-or-nothing bundle.
struct NarrowNumericFeatureSupport {
    bool shader_float16{false};
    bool shader_int8{false};
    bool storage_buffer_8bit_access{false};
    bool uniform_storage_buffer_8bit_access{false};
    bool storage_buffer_16bit_access{false};
    bool uniform_storage_buffer_16bit_access{false};
};

struct NarrowNumericFeaturePlan {
    bool shader_float16{false};
    bool shader_int8{false};
    bool storage_buffer_8bit_access{false};
    bool uniform_storage_buffer_8bit_access{false};
    bool storage_buffer_16bit_access{false};
    bool uniform_storage_buffer_16bit_access{false};
};

[[nodiscard]] constexpr auto plan_narrow_numeric_features(
    NarrowNumericFeatureSupport support) noexcept {
    return NarrowNumericFeaturePlan{
        .shader_float16 = support.shader_float16,
        .shader_int8 = support.shader_int8,
        .storage_buffer_8bit_access =
            support.storage_buffer_8bit_access,
        .uniform_storage_buffer_8bit_access =
            support.uniform_storage_buffer_8bit_access,
        .storage_buffer_16bit_access =
            support.storage_buffer_16bit_access,
        .uniform_storage_buffer_16bit_access =
            support.uniform_storage_buffer_16bit_access};
}

struct OptionalDeviceFeatureSupport {
    bool fragment_shader_barycentric_extension{false};
    bool fragment_shader_barycentric_feature{false};
    bool ray_query_extension{false};
    bool ray_query_feature{false};
    bool acceleration_structure_extension{false};
    bool acceleration_structure_feature{false};
    bool deferred_host_operations_extension{false};
    bool buffer_device_address{false};
    bool ray_tracing_pipeline_extension{false};
    bool ray_tracing_pipeline_feature{false};
    bool ray_traversal_primitive_culling_feature{false};
    bool ray_tracing_motion_blur_extension{false};
    bool ray_tracing_motion_blur_feature{false};
};

struct OptionalDeviceFeatureRequest {
    bool ray_query{false};
    bool ray_tracing_motion_blur{false};
};

struct OptionalDeviceFeaturePlan {
    bool fragment_shader_barycentric{false};
    bool ray_query{false};
    bool acceleration_structure{false};
    bool ray_tracing_pipeline{false};
    bool ray_tracing_motion_blur{false};
};

[[nodiscard]] constexpr auto plan_optional_device_features(
    OptionalDeviceFeatureSupport support,
    OptionalDeviceFeatureRequest request) noexcept {
    auto fragment_shader_barycentric =
        support.fragment_shader_barycentric_extension &&
        support.fragment_shader_barycentric_feature;
    auto ray_query =
        request.ray_query &&
        support.ray_query_extension &&
        support.ray_query_feature &&
        support.acceleration_structure_extension &&
        support.acceleration_structure_feature &&
        support.deferred_host_operations_extension &&
        support.buffer_device_address;
    auto ray_tracing_motion_blur =
        ray_query &&
        request.ray_tracing_motion_blur &&
        support.ray_tracing_pipeline_extension &&
        support.ray_tracing_pipeline_feature &&
        support.ray_traversal_primitive_culling_feature &&
        support.ray_tracing_motion_blur_extension &&
        support.ray_tracing_motion_blur_feature;
    return OptionalDeviceFeaturePlan{
        .fragment_shader_barycentric = fragment_shader_barycentric,
        .ray_query = ray_query,
        .acceleration_structure = ray_query,
        .ray_tracing_pipeline = ray_tracing_motion_blur,
        .ray_tracing_motion_blur = ray_tracing_motion_blur};
}

}// namespace lc::vk::detail
