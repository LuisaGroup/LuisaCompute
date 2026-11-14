#include "vk_extension_registry.h"

#include <algorithm>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/backends/ext/vk_config_ext.h>

#if defined(__APPLE__)
#include <vulkan/vulkan_metal.h>
#endif
#ifdef VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan_beta.h>
#endif

namespace lc::vk::detail {
namespace optional_extension_detail {

// A single VkPhysicalDevice*Features* bit. `Member` names one VkBool32 field of
// `T`; the helper reports support and turns exactly that bit on.
template<typename T, VkBool32 T::*Member>
[[nodiscard]] bool enable_bit(T *feature) noexcept {
    if (feature->*Member != VK_TRUE) { return false; }
    feature->*Member = VK_TRUE;
    return true;
}

// `enable_bit` as a `check_and_enable` callback.
template<typename T, VkBool32 T::*Member>
[[nodiscard]] bool check_single_bit(void *feature) noexcept {
    return enable_bit<T, Member>(static_cast<T *>(feature));
}

// A set of VkPhysicalDevice*Features* bits that must *all* be supported. The
// callback turns each bit on individually, so a driver that reports only some
// of them still enables the subset it has after the "all" test succeeded.
template<typename T, VkBool32 T::*...Members>
[[nodiscard]] bool check_all_bits(void *feature) noexcept {
    auto value = static_cast<T *>(feature);
    auto supported = ((value->*Members == VK_TRUE) && ...);
    if (!supported) { return false; }
    ((value->*Members = VK_TRUE), ...);
    return true;
}

// At least one of the bits must be supported; every supported bit is turned on.
template<typename T, VkBool32 T::*...Members>
[[nodiscard]] bool check_any_bits(void *feature) noexcept {
    auto value = static_cast<T *>(feature);
    if (!((value->*Members == VK_TRUE) || ...)) { return false; }
    ((value->*Members = (value->*Members == VK_TRUE) ? VK_TRUE : VK_FALSE), ...);
    return true;
}

template<typename T>
[[nodiscard]] void *make_feature(void *storage) noexcept {
    auto feature = new (storage) T{};
    return feature;
}

// Every entry that carries a feature structure uses the same construction
// shape: zero-initialize, then fix up sType/pNext so a designated-initializer
// ordering mistake cannot silently produce a wrong structure type.
template<typename T, VkStructureType StructureType>
[[nodiscard]] void *make_feature_of(void *storage) noexcept {
    auto feature = new (storage) T{};
    feature->sType = StructureType;
    feature->pNext = nullptr;
    return feature;
}

// Switch lambdas. Written as named helpers so the table below stays readable.
#define LC_VK_OPTIONAL_EXTENSION_SWITCH(Name, Getter)                       \
    [[nodiscard]] inline bool requested_##Name(                            \
        const luisa::compute::VulkanDeviceConfigExt &config) noexcept {    \
        return config.Getter();                                            \
    }

LC_VK_OPTIONAL_EXTENSION_SWITCH(descriptor_heap, enable_descriptor_heap)
LC_VK_OPTIONAL_EXTENSION_SWITCH(device_address_commands, enable_device_address_commands)
LC_VK_OPTIONAL_EXTENSION_SWITCH(unified_image_layouts, enable_unified_image_layouts)
LC_VK_OPTIONAL_EXTENSION_SWITCH(copy_memory_indirect, enable_copy_memory_indirect)
LC_VK_OPTIONAL_EXTENSION_SWITCH(memory_decompression, enable_memory_decompression)
LC_VK_OPTIONAL_EXTENSION_SWITCH(zero_initialize_device_memory, enable_zero_initialize_device_memory)
LC_VK_OPTIONAL_EXTENSION_SWITCH(internally_synchronized_queues, enable_internally_synchronized_queues)
LC_VK_OPTIONAL_EXTENSION_SWITCH(extended_flags, enable_extended_flags)
LC_VK_OPTIONAL_EXTENSION_SWITCH(image_tiling_control, enable_image_tiling_control)
LC_VK_OPTIONAL_EXTENSION_SWITCH(present_id2, enable_present_id2)
LC_VK_OPTIONAL_EXTENSION_SWITCH(present_wait2, enable_present_wait2)
LC_VK_OPTIONAL_EXTENSION_SWITCH(present_mode_fifo_latest_ready, enable_present_mode_fifo_latest_ready)
LC_VK_OPTIONAL_EXTENSION_SWITCH(surface_maintenance1, enable_surface_maintenance1)
LC_VK_OPTIONAL_EXTENSION_SWITCH(swapchain_maintenance1, enable_swapchain_maintenance1)
LC_VK_OPTIONAL_EXTENSION_SWITCH(present_timing, enable_present_timing)
LC_VK_OPTIONAL_EXTENSION_SWITCH(multisampled_render_to_swapchain, enable_multisampled_render_to_swapchain)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_fma, enable_shader_fma)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_64bit_indexing, enable_shader_64bit_indexing)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_uniform_buffer_unsized_array, enable_shader_uniform_buffer_unsized_array)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_long_vector, enable_shader_long_vector)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_subgroup_partitioned, enable_shader_subgroup_partitioned)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_split_barrier, enable_shader_split_barrier)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_ocp_microscaling_types, enable_shader_ocp_microscaling_types)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_constant_data, enable_shader_constant_data)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_abort, enable_shader_abort)
LC_VK_OPTIONAL_EXTENSION_SWITCH(ray_tracing_invocation_reorder, enable_ray_tracing_invocation_reorder)
LC_VK_OPTIONAL_EXTENSION_SWITCH(opacity_micromap, enable_opacity_micromap)
LC_VK_OPTIONAL_EXTENSION_SWITCH(custom_resolve, enable_custom_resolve)
LC_VK_OPTIONAL_EXTENSION_SWITCH(fragment_density_map_offset, enable_fragment_density_map_offset)
LC_VK_OPTIONAL_EXTENSION_SWITCH(primitive_restart_index, enable_primitive_restart_index)
LC_VK_OPTIONAL_EXTENSION_SWITCH(texture_compression_astc_3d, enable_texture_compression_astc_3d)
LC_VK_OPTIONAL_EXTENSION_SWITCH(pipeline_library_group_handles, enable_pipeline_library_group_handles)
LC_VK_OPTIONAL_EXTENSION_SWITCH(external_memory_metal, enable_external_memory_metal)
LC_VK_OPTIONAL_EXTENSION_SWITCH(depth_clamp_zero_one, enable_depth_clamp_zero_one)
LC_VK_OPTIONAL_EXTENSION_SWITCH(maintenance8, enable_maintenance8)
LC_VK_OPTIONAL_EXTENSION_SWITCH(maintenance9, enable_maintenance9)
LC_VK_OPTIONAL_EXTENSION_SWITCH(maintenance10, enable_maintenance10)
LC_VK_OPTIONAL_EXTENSION_SWITCH(maintenance11, enable_maintenance11)
LC_VK_OPTIONAL_EXTENSION_SWITCH(maintenance12, enable_maintenance12)
LC_VK_OPTIONAL_EXTENSION_SWITCH(robustness2, enable_robustness2)
LC_VK_OPTIONAL_EXTENSION_SWITCH(tensor_arm, enable_tensor_arm)
LC_VK_OPTIONAL_EXTENSION_SWITCH(data_graph_arm, enable_data_graph_arm)
LC_VK_OPTIONAL_EXTENSION_SWITCH(tile_shading_qcom, enable_tile_shading_qcom)
LC_VK_OPTIONAL_EXTENSION_SWITCH(tile_memory_heap_qcom, enable_tile_memory_heap_qcom)
LC_VK_OPTIONAL_EXTENSION_SWITCH(partitioned_acceleration_structure, enable_partitioned_acceleration_structure)
LC_VK_OPTIONAL_EXTENSION_SWITCH(ray_tracing_linear_swept_spheres, enable_ray_tracing_linear_swept_spheres)
LC_VK_OPTIONAL_EXTENSION_SWITCH(dense_geometry_format, enable_dense_geometry_format)
LC_VK_OPTIONAL_EXTENSION_SWITCH(shader_mixed_float_dot_product, enable_shader_mixed_float_dot_product)

#undef LC_VK_OPTIONAL_EXTENSION_SWITCH

// VK_EXT_external_memory_metal only exists on Apple platforms. Requesting it
// elsewhere would be pointless: the device never exposes it.
[[nodiscard]] inline bool requested_external_memory_metal_platform(
    const luisa::compute::VulkanDeviceConfigExt &config) noexcept {
#if defined(__APPLE__)
    return config.enable_external_memory_metal();
#else
    static_cast<void>(config);
    return false;
#endif
}

// Robustness2 is a debugging aid: null descriptors and out-of-range accesses
// become well-defined, at a small runtime cost. Only non-NDEBUG builds request
// it, and the switch can still turn it off.
[[nodiscard]] inline bool requested_robustness2_debug(
    const luisa::compute::VulkanDeviceConfigExt &config) noexcept {
#ifndef NDEBUG
    return config.enable_robustness2();
#else
    static_cast<void>(config);
    return false;
#endif
}

// VK_EXT_external_memory_metal only has a macro in the Apple platform header.
#if defined(__APPLE__)
inline constexpr char external_memory_metal_extension_name[] =
    VK_EXT_EXTERNAL_MEMORY_METAL_EXTENSION_NAME;
#else
inline constexpr char external_memory_metal_extension_name[] =
    "VK_EXT_external_memory_metal";
#endif

// Table of every optional extension the backend negotiates.
inline constexpr OptionalExtensionDescriptor optional_extensions[] = {
    // ---- resources, descriptors, GPU driven rendering -------------------
    {OptionalExtension::DESCRIPTOR_HEAP,
     VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, nullptr, nullptr,
     requested_descriptor_heap,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceDescriptorHeapFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceDescriptorHeapFeaturesEXT,
                      &VkPhysicalDeviceDescriptorHeapFeaturesEXT::descriptorHeap>},
    {OptionalExtension::DEVICE_ADDRESS_COMMANDS,
     VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME, nullptr, nullptr,
     requested_device_address_commands,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR,
                      &VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR::deviceAddressCommands>},
    {OptionalExtension::UNIFIED_IMAGE_LAYOUTS,
     VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME, nullptr, nullptr,
     requested_unified_image_layouts,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR,
                      &VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR::unifiedImageLayouts>},
    {OptionalExtension::COPY_MEMORY_INDIRECT,
     VK_KHR_COPY_MEMORY_INDIRECT_EXTENSION_NAME,
     VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME, nullptr,
     requested_copy_memory_indirect,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR,
                      &VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR::indirectMemoryCopy>,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_NV},
    {OptionalExtension::MEMORY_DECOMPRESSION,
     VK_EXT_MEMORY_DECOMPRESSION_EXTENSION_NAME,
     VK_NV_MEMORY_DECOMPRESSION_EXTENSION_NAME, nullptr,
     requested_memory_decompression,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceMemoryDecompressionFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceMemoryDecompressionFeaturesEXT,
                      &VkPhysicalDeviceMemoryDecompressionFeaturesEXT::memoryDecompression>},
    {OptionalExtension::ZERO_INITIALIZE_DEVICE_MEMORY,
     VK_EXT_ZERO_INITIALIZE_DEVICE_MEMORY_EXTENSION_NAME, nullptr, nullptr,
     requested_zero_initialize_device_memory,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_DEVICE_MEMORY_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT,
                      &VkPhysicalDeviceZeroInitializeDeviceMemoryFeaturesEXT::zeroInitializeDeviceMemory>},
    {OptionalExtension::INTERNALLY_SYNCHRONIZED_QUEUES,
     VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME, nullptr, nullptr,
     requested_internally_synchronized_queues,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR,
                      &VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR::internallySynchronizedQueues>},
    {OptionalExtension::EXTENDED_FLAGS,
     VK_KHR_EXTENDED_FLAGS_EXTENSION_NAME, nullptr, nullptr,
     requested_extended_flags,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_FLAGS_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceExtendedFlagsFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_FLAGS_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceExtendedFlagsFeaturesKHR,
                      &VkPhysicalDeviceExtendedFlagsFeaturesKHR::extendedFlags>},
    {OptionalExtension::IMAGE_TILING_CONTROL,
     VK_EXT_IMAGE_TILING_CONTROL_EXTENSION_NAME, nullptr, nullptr,
     requested_image_tiling_control,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_TILING_CONTROL_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceImageTilingControlFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_TILING_CONTROL_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceImageTilingControlFeaturesEXT,
                      &VkPhysicalDeviceImageTilingControlFeaturesEXT::imageTilingControl>},

    // ---- presentation, low latency, frame pacing ------------------------
    {OptionalExtension::PRESENT_ID2,
     VK_KHR_PRESENT_ID_2_EXTENSION_NAME, nullptr, VK_KHR_SWAPCHAIN_EXTENSION_NAME,
     requested_present_id2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR,
     make_feature_of<VkPhysicalDevicePresentId2FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR>,
     check_single_bit<VkPhysicalDevicePresentId2FeaturesKHR,
                      &VkPhysicalDevicePresentId2FeaturesKHR::presentId2>},
    {OptionalExtension::PRESENT_WAIT2,
     VK_KHR_PRESENT_WAIT_2_EXTENSION_NAME, nullptr, VK_KHR_SWAPCHAIN_EXTENSION_NAME,
     requested_present_wait2,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR,
     make_feature_of<VkPhysicalDevicePresentWait2FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR>,
     check_single_bit<VkPhysicalDevicePresentWait2FeaturesKHR,
                      &VkPhysicalDevicePresentWait2FeaturesKHR::presentWait2>},
    {OptionalExtension::PRESENT_MODE_FIFO_LATEST_READY,
     VK_KHR_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME,
     VK_EXT_PRESENT_MODE_FIFO_LATEST_READY_EXTENSION_NAME, VK_KHR_SWAPCHAIN_EXTENSION_NAME,
     requested_present_mode_fifo_latest_ready,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR,
     make_feature_of<VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR>,
     check_single_bit<VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR,
                      &VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR::presentModeFifoLatestReady>},
    // VK_KHR_surface_maintenance1 is an *instance* extension (vk.xml type
    // "instance"), so it is matched against the enabled instance extension list
    // and only when the device also provides VK_KHR_swapchain, i.e. when a
    // surface/swapchain exists at all.
    {OptionalExtension::SURFACE_MAINTENANCE1,
     VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
     VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME, VK_KHR_SWAPCHAIN_EXTENSION_NAME,
     requested_surface_maintenance1,
     VK_STRUCTURE_TYPE_MAX_ENUM, nullptr, nullptr,
     VK_STRUCTURE_TYPE_MAX_ENUM, true},
    {OptionalExtension::SWAPCHAIN_MAINTENANCE1,
     VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
     VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME, VK_KHR_SWAPCHAIN_EXTENSION_NAME,
     requested_swapchain_maintenance1,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR,
                      &VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR::swapchainMaintenance1>},
    {OptionalExtension::PRESENT_TIMING,
     VK_EXT_PRESENT_TIMING_EXTENSION_NAME, nullptr, VK_KHR_SWAPCHAIN_EXTENSION_NAME,
     requested_present_timing,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT,
     make_feature_of<VkPhysicalDevicePresentTimingFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT>,
     check_single_bit<VkPhysicalDevicePresentTimingFeaturesEXT,
                      &VkPhysicalDevicePresentTimingFeaturesEXT::presentTiming>},
    {OptionalExtension::MULTISAMPLED_RENDER_TO_SWAPCHAIN,
     VK_EXT_MULTISAMPLED_RENDER_TO_SWAPCHAIN_EXTENSION_NAME, nullptr, nullptr,
     requested_multisampled_render_to_swapchain,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SWAPCHAIN_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceMultisampledRenderToSwapchainFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SWAPCHAIN_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceMultisampledRenderToSwapchainFeaturesEXT,
                      &VkPhysicalDeviceMultisampledRenderToSwapchainFeaturesEXT::multisampledRenderToSwapchain>},

    // ---- shader capability ----------------------------------------------
    {OptionalExtension::SHADER_FMA,
     VK_KHR_SHADER_FMA_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_fma,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceShaderFmaFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceShaderFmaFeaturesKHR,
                      &VkPhysicalDeviceShaderFmaFeaturesKHR::shaderFmaFloat32>},
    {OptionalExtension::SHADER_64BIT_INDEXING,
     VK_EXT_SHADER_64BIT_INDEXING_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_64bit_indexing,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceShader64BitIndexingFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_64_BIT_INDEXING_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceShader64BitIndexingFeaturesEXT,
                      &VkPhysicalDeviceShader64BitIndexingFeaturesEXT::shader64BitIndexing>},
    {OptionalExtension::SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY,
     VK_EXT_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_uniform_buffer_unsized_array,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT,
                      &VkPhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT::shaderUniformBufferUnsizedArray>},
    {OptionalExtension::SHADER_LONG_VECTOR,
     VK_EXT_SHADER_LONG_VECTOR_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_long_vector,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_LONG_VECTOR_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceShaderLongVectorFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_LONG_VECTOR_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceShaderLongVectorFeaturesEXT,
                      &VkPhysicalDeviceShaderLongVectorFeaturesEXT::longVector>},
    {OptionalExtension::SHADER_SUBGROUP_PARTITIONED,
     VK_EXT_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME,
     VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, nullptr,
     requested_shader_subgroup_partitioned,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_PARTITIONED_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceShaderSubgroupPartitionedFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_PARTITIONED_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceShaderSubgroupPartitionedFeaturesEXT,
                      &VkPhysicalDeviceShaderSubgroupPartitionedFeaturesEXT::shaderSubgroupPartitioned>},
    {OptionalExtension::SHADER_SPLIT_BARRIER,
     VK_EXT_SHADER_SPLIT_BARRIER_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_split_barrier,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SPLIT_BARRIER_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceShaderSplitBarrierFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SPLIT_BARRIER_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceShaderSplitBarrierFeaturesEXT,
                      &VkPhysicalDeviceShaderSplitBarrierFeaturesEXT::shaderSplitBarrier>},
    {OptionalExtension::SHADER_OCP_MICROSCALING_TYPES,
     VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_ocp_microscaling_types,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT>,
     check_any_bits<VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT,
                    &VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT::shaderFloat4,
                    &VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT::shaderFloat6,
                    &VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT::shaderFloat8UnsignedE8M0,
                    &VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT::shaderMXInt8>},
    {OptionalExtension::SHADER_CONSTANT_DATA,
     VK_KHR_SHADER_CONSTANT_DATA_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_constant_data,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CONSTANT_DATA_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceShaderConstantDataFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CONSTANT_DATA_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceShaderConstantDataFeaturesKHR,
                      &VkPhysicalDeviceShaderConstantDataFeaturesKHR::shaderConstantData>},
    {OptionalExtension::SHADER_ABORT,
     VK_KHR_SHADER_ABORT_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_abort,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ABORT_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceShaderAbortFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ABORT_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceShaderAbortFeaturesKHR,
                      &VkPhysicalDeviceShaderAbortFeaturesKHR::shaderAbort>},

    // ---- ray tracing and rendering --------------------------------------
    {OptionalExtension::RAY_TRACING_INVOCATION_REORDER,
     VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME,
     VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
     requested_ray_tracing_invocation_reorder,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT,
                      &VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT::rayTracingInvocationReorder>,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV},
    {OptionalExtension::OPACITY_MICROMAP,
     VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME,
     VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
     requested_opacity_micromap,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceOpacityMicromapFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceOpacityMicromapFeaturesKHR,
                      &VkPhysicalDeviceOpacityMicromapFeaturesKHR::micromap>,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT},
    {OptionalExtension::CUSTOM_RESOLVE,
     VK_EXT_CUSTOM_RESOLVE_EXTENSION_NAME, nullptr, nullptr,
     requested_custom_resolve,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_RESOLVE_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceCustomResolveFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_RESOLVE_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceCustomResolveFeaturesEXT,
                      &VkPhysicalDeviceCustomResolveFeaturesEXT::customResolve>},
    {OptionalExtension::FRAGMENT_DENSITY_MAP_OFFSET,
     VK_EXT_FRAGMENT_DENSITY_MAP_OFFSET_EXTENSION_NAME,
     VK_QCOM_FRAGMENT_DENSITY_MAP_OFFSET_EXTENSION_NAME, nullptr,
     requested_fragment_density_map_offset,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceFragmentDensityMapOffsetFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_OFFSET_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceFragmentDensityMapOffsetFeaturesEXT,
                      &VkPhysicalDeviceFragmentDensityMapOffsetFeaturesEXT::fragmentDensityMapOffset>},
    {OptionalExtension::PRIMITIVE_RESTART_INDEX,
     VK_EXT_PRIMITIVE_RESTART_INDEX_EXTENSION_NAME, nullptr, nullptr,
     requested_primitive_restart_index,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_RESTART_INDEX_FEATURES_EXT,
     make_feature_of<VkPhysicalDevicePrimitiveRestartIndexFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_RESTART_INDEX_FEATURES_EXT>,
     check_single_bit<VkPhysicalDevicePrimitiveRestartIndexFeaturesEXT,
                      &VkPhysicalDevicePrimitiveRestartIndexFeaturesEXT::primitiveRestartIndex>},
    {OptionalExtension::TEXTURE_COMPRESSION_ASTC_3D,
     VK_EXT_TEXTURE_COMPRESSION_ASTC_3D_EXTENSION_NAME, nullptr, nullptr,
     requested_texture_compression_astc_3d,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_3D_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceTextureCompressionASTC3DFeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_3D_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceTextureCompressionASTC3DFeaturesEXT,
                      &VkPhysicalDeviceTextureCompressionASTC3DFeaturesEXT::textureCompressionASTC_3D>},
    {OptionalExtension::PIPELINE_LIBRARY_GROUP_HANDLES,
     VK_KHR_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME,
     VK_EXT_PIPELINE_LIBRARY_GROUP_HANDLES_EXTENSION_NAME, nullptr,
     requested_pipeline_library_group_handles,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_KHR,
     make_feature_of<VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_LIBRARY_GROUP_HANDLES_FEATURES_KHR>,
     check_single_bit<VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesKHR,
                      &VkPhysicalDevicePipelineLibraryGroupHandlesFeaturesKHR::pipelineLibraryGroupHandles>},
    {OptionalExtension::EXTERNAL_MEMORY_METAL,
     external_memory_metal_extension_name, nullptr, nullptr,
     requested_external_memory_metal_platform,
     VK_STRUCTURE_TYPE_MAX_ENUM, nullptr, nullptr},
    {OptionalExtension::DEPTH_CLAMP_ZERO_ONE,
     VK_KHR_DEPTH_CLAMP_ZERO_ONE_EXTENSION_NAME,
     VK_EXT_DEPTH_CLAMP_ZERO_ONE_EXTENSION_NAME, nullptr,
     requested_depth_clamp_zero_one,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceDepthClampZeroOneFeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceDepthClampZeroOneFeaturesKHR,
                      &VkPhysicalDeviceDepthClampZeroOneFeaturesKHR::depthClampZeroOne>},

    // ---- maintenance -----------------------------------------------------
    {OptionalExtension::MAINTENANCE8,
     VK_KHR_MAINTENANCE_8_EXTENSION_NAME, nullptr, nullptr,
     requested_maintenance8,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceMaintenance8FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceMaintenance8FeaturesKHR,
                      &VkPhysicalDeviceMaintenance8FeaturesKHR::maintenance8>},
    {OptionalExtension::MAINTENANCE9,
     VK_KHR_MAINTENANCE_9_EXTENSION_NAME, nullptr, nullptr,
     requested_maintenance9,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceMaintenance9FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceMaintenance9FeaturesKHR,
                      &VkPhysicalDeviceMaintenance9FeaturesKHR::maintenance9>},
    {OptionalExtension::MAINTENANCE10,
     VK_KHR_MAINTENANCE_10_EXTENSION_NAME, nullptr, nullptr,
     requested_maintenance10,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_10_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceMaintenance10FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_10_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceMaintenance10FeaturesKHR,
                      &VkPhysicalDeviceMaintenance10FeaturesKHR::maintenance10>},
    {OptionalExtension::MAINTENANCE11,
     VK_KHR_MAINTENANCE_11_EXTENSION_NAME, nullptr, nullptr,
     requested_maintenance11,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_11_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceMaintenance11FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_11_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceMaintenance11FeaturesKHR,
                      &VkPhysicalDeviceMaintenance11FeaturesKHR::maintenance11>},
    {OptionalExtension::MAINTENANCE12,
     VK_KHR_MAINTENANCE_12_EXTENSION_NAME, nullptr, nullptr,
     requested_maintenance12,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_12_FEATURES_KHR,
     make_feature_of<VkPhysicalDeviceMaintenance12FeaturesKHR,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_12_FEATURES_KHR>,
     check_single_bit<VkPhysicalDeviceMaintenance12FeaturesKHR,
                      &VkPhysicalDeviceMaintenance12FeaturesKHR::maintenance12>},
    {OptionalExtension::ROBUSTNESS2,
     VK_KHR_ROBUSTNESS_2_EXTENSION_NAME,
     VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, nullptr,
     requested_robustness2_debug,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
     make_feature_of<VkPhysicalDeviceRobustness2FeaturesEXT,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT>,
     check_single_bit<VkPhysicalDeviceRobustness2FeaturesEXT,
                      &VkPhysicalDeviceRobustness2FeaturesEXT::nullDescriptor>},

    // ---- vendor ----------------------------------------------------------
    {OptionalExtension::TENSOR_ARM,
     VK_ARM_TENSORS_EXTENSION_NAME, nullptr, nullptr,
     requested_tensor_arm,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM,
     make_feature_of<VkPhysicalDeviceTensorFeaturesARM,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM>,
     check_single_bit<VkPhysicalDeviceTensorFeaturesARM,
                      &VkPhysicalDeviceTensorFeaturesARM::tensors>},
    {OptionalExtension::DATA_GRAPH_ARM,
     VK_ARM_DATA_GRAPH_EXTENSION_NAME, nullptr, nullptr,
     requested_data_graph_arm,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM,
     make_feature_of<VkPhysicalDeviceDataGraphFeaturesARM,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM>,
     check_single_bit<VkPhysicalDeviceDataGraphFeaturesARM,
                      &VkPhysicalDeviceDataGraphFeaturesARM::dataGraph>},
    {OptionalExtension::TILE_SHADING_QCOM,
     VK_QCOM_TILE_SHADING_EXTENSION_NAME, nullptr, nullptr,
     requested_tile_shading_qcom,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM,
     make_feature_of<VkPhysicalDeviceTileShadingFeaturesQCOM,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM>,
     check_single_bit<VkPhysicalDeviceTileShadingFeaturesQCOM,
                      &VkPhysicalDeviceTileShadingFeaturesQCOM::tileShading>},
    {OptionalExtension::TILE_MEMORY_HEAP_QCOM,
     VK_QCOM_TILE_MEMORY_HEAP_EXTENSION_NAME, nullptr, nullptr,
     requested_tile_memory_heap_qcom,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES_QCOM,
     make_feature_of<VkPhysicalDeviceTileMemoryHeapFeaturesQCOM,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES_QCOM>,
     check_single_bit<VkPhysicalDeviceTileMemoryHeapFeaturesQCOM,
                      &VkPhysicalDeviceTileMemoryHeapFeaturesQCOM::tileMemoryHeap>},
    {OptionalExtension::PARTITIONED_ACCELERATION_STRUCTURE,
     VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME, nullptr, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
     requested_partitioned_acceleration_structure,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV,
     make_feature_of<VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV>,
     check_single_bit<VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV,
                      &VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV::partitionedAccelerationStructure>},
    {OptionalExtension::RAY_TRACING_LINEAR_SWEPT_SPHERES,
     VK_NV_RAY_TRACING_LINEAR_SWEPT_SPHERES_EXTENSION_NAME, nullptr, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
     requested_ray_tracing_linear_swept_spheres,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV,
     make_feature_of<VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_LINEAR_SWEPT_SPHERES_FEATURES_NV>,
     check_single_bit<VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV,
                      &VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV::linearSweptSpheres>},
#ifdef VK_ENABLE_BETA_EXTENSIONS
    {OptionalExtension::DENSE_GEOMETRY_FORMAT,
     VK_AMDX_DENSE_GEOMETRY_FORMAT_EXTENSION_NAME, nullptr, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
     requested_dense_geometry_format,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DENSE_GEOMETRY_FORMAT_FEATURES_AMDX,
     make_feature_of<VkPhysicalDeviceDenseGeometryFormatFeaturesAMDX,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DENSE_GEOMETRY_FORMAT_FEATURES_AMDX>,
     check_single_bit<VkPhysicalDeviceDenseGeometryFormatFeaturesAMDX,
                      &VkPhysicalDeviceDenseGeometryFormatFeaturesAMDX::denseGeometryFormat>},
#endif
    {OptionalExtension::SHADER_MIXED_FLOAT_DOT_PRODUCT,
     VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_EXTENSION_NAME, nullptr, nullptr,
     requested_shader_mixed_float_dot_product,
     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE,
     make_feature_of<VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE>,
     // Only the FP16-input/FP32-accumulator pair is negotiated: it is the only
     // shape HLSL's `dot2add` can express.
     check_single_bit<VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE,
                      &VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE::shaderMixedFloatDotProductFloat16AccFloat32>},
};

inline constexpr size_t descriptor_count =
    sizeof(optional_extensions) / sizeof(optional_extensions[0]);

static_assert(sizeof(VkPhysicalDeviceTileShadingFeaturesQCOM) <=
                  optional_extension_feature_storage_size,
              "feature storage must hold the largest negotiated feature structure");

// A DeviceConfig without a VulkanDeviceConfigExt must behave exactly like the
// default switches (everything allowed, the physical device still decides).
[[nodiscard]] const luisa::compute::VulkanDeviceConfigExt &
default_device_config() noexcept {
    static const luisa::compute::VulkanDeviceConfigExt config;
    return config;
}

[[nodiscard]] bool name_matches(luisa::string_view candidate,
                                const char *name) noexcept {
    return name != nullptr && candidate == luisa::string_view{name};
}

}// namespace optional_extension_detail

const char *optional_extension_name(OptionalExtension ext) noexcept {
    for (auto const &desc : optional_extension_detail::optional_extensions) {
        if (desc.id == ext) { return desc.name; }
    }
    return "unknown";
}

OptionalExtension optional_extension_from_name(luisa::string_view name) noexcept {
    for (auto const &desc : optional_extension_detail::optional_extensions) {
        if (optional_extension_detail::name_matches(name, desc.name) ||
            optional_extension_detail::name_matches(name, desc.alias_name)) {
            return desc.id;
        }
    }
    return OptionalExtension::COUNT;
}

OptionalExtensionRegistry::OptionalExtensionRegistry() noexcept {
    _entries.reserve(optional_extension_detail::descriptor_count);
    for (auto const &desc : optional_extension_detail::optional_extensions) {
        _entries.emplace_back();
        _entries.back().desc = desc;
        _entries.back().name = desc.name;
    }
    refresh_states();
}

OptionalExtensionRegistry::~OptionalExtensionRegistry() noexcept = default;
OptionalExtensionRegistry::OptionalExtensionRegistry(
    OptionalExtensionRegistry &&) noexcept = default;
OptionalExtensionRegistry &OptionalExtensionRegistry::operator=(
    OptionalExtensionRegistry &&) noexcept = default;

void OptionalExtensionRegistry::select(
    luisa::span<const luisa::string> supported_device_extensions,
    luisa::span<const luisa::string> enabled_instance_extensions,
    const luisa::compute::VulkanDeviceConfigExt *config,
    bool imported_device) noexcept {
    _supported_bits = 0u;
    _requested_bits = 0u;
    _enabled_bits = 0u;
    _enabled_names.clear();
    _query_head = nullptr;
    _device_head = nullptr;
    for (auto &entry : _entries) {
        auto const &desc = entry.desc;
        // An imported logical device cannot attest which extensions or feature
        // bits it was created with, so no optional extension is ever assumed.
        const char *matched_name = nullptr;
        if (!imported_device) {
            // Instance-level entries are matched against the instance extension
            // list the backend actually enabled; everything else against the
            // physical device's device extension list.
            auto const &candidate_names = desc.instance_level ?
                                              enabled_instance_extensions :
                                              supported_device_extensions;
            for (auto const &name : candidate_names) {
                if (optional_extension_detail::name_matches(name, desc.name)) {
                    matched_name = desc.name;
                    break;
                }
                if (optional_extension_detail::name_matches(name, desc.alias_name)) {
                    matched_name = desc.alias_name;
                    break;
                }
            }
        }
        entry.name = matched_name != nullptr ? matched_name : desc.name;
        entry.supported = matched_name != nullptr;
        entry.requested = entry.supported &&
                          desc.requested(config != nullptr ?
                                             *config :
                                             optional_extension_detail::default_device_config());
        entry.enabled = false;
        entry.feature = nullptr;
        if (entry.supported) {
            _supported_bits |= uint64_t{1u} << static_cast<uint32_t>(desc.id);
        }
        if (entry.requested) {
            _requested_bits |= uint64_t{1u} << static_cast<uint32_t>(desc.id);
        }
    }
    // Chain the feature structures of the candidates for one probe call. A
    // structure is only chained when its extension is exposed by the physical
    // device, so an unknown sType can never reach the driver.
    void *head = nullptr;
    for (auto &entry : _entries) {
        auto const &desc = entry.desc;
        if (!entry.requested || desc.make_feature == nullptr) { continue; }
        entry.feature = desc.make_feature(entry.storage);
        // A KHR/EXT twin that defines its own sType value must be probed with
        // the spelling the physical device actually exposes.
        if (desc.alias_feature_structure_type != VK_STRUCTURE_TYPE_MAX_ENUM &&
            desc.alias_name != nullptr && entry.name != nullptr &&
            std::strcmp(entry.name, desc.alias_name) == 0) {
            static_cast<VkBaseOutStructure *>(entry.feature)->sType =
                desc.alias_feature_structure_type;
        }
        auto base = static_cast<VkBaseOutStructure *>(entry.feature);
        base->pNext = static_cast<VkBaseOutStructure *>(head);
        head = entry.feature;
    }
    _query_head = head;
    refresh_states();
}

void *OptionalExtensionRegistry::feature_query_chain() const noexcept {
    return _query_head;
}

void OptionalExtensionRegistry::note_enabled_extension(
    luisa::string_view name) noexcept {
    _registered_extensions.emplace_back(luisa::string{name});
}

void OptionalExtensionRegistry::resolve() noexcept {
    auto known = [this](luisa::string_view name) noexcept {
        for (auto const &external : _registered_extensions) {
            if (external == name) { return true; }
        }
        for (auto &entry : _entries) {
            if (!entry.enabled) { continue; }
            if (optional_extension_detail::name_matches(name, entry.desc.name) ||
                optional_extension_detail::name_matches(name, entry.desc.alias_name)) {
                return true;
            }
        }
        return false;
    };
    // Two passes: a second pass drops entries whose prerequisite was itself
    // rejected, so a dependency chain of any depth settles.
    for (auto pass = 0u; pass < 2u; ++pass) {
        for (auto &entry : _entries) {
            auto const &desc = entry.desc;
            if (!entry.requested) { continue; }
            auto usable = true;
            if (entry.feature != nullptr && desc.check_and_enable != nullptr) {
                usable = desc.check_and_enable(entry.feature);
            }
            if (usable && desc.requires_extension != nullptr) {
                usable = known(desc.requires_extension);
            }
            entry.enabled = usable;
        }
    }
    _enabled_bits = 0u;
    _enabled_names.clear();
    _device_head = nullptr;
    for (auto &entry : _entries) {
        if (!entry.enabled) {
            // A rejected entry keeps its (all clear) feature structure out of
            // the logical-device chain: including a feature structure whose
            // extension is not enabled is a validation error.
            continue;
        }
        _enabled_bits |= uint64_t{1u} << static_cast<uint32_t>(entry.desc.id);
        // Instance-level extensions are enabled at vkCreateInstance time, never
        // in VkDeviceCreateInfo: passing one to vkCreateDevice fails with
        // "enabled device extension is not present at device level".
        if (entry.name != nullptr && !entry.desc.instance_level) {
            _enabled_names.emplace_back(entry.name);
        }
    }
    refresh_states();
}

void *OptionalExtensionRegistry::device_feature_chain(void *tail) noexcept {
    void *head = tail;
    for (auto &entry : _entries) {
        if (!entry.enabled || entry.feature == nullptr) { continue; }
        auto base = static_cast<VkBaseOutStructure *>(entry.feature);
        base->pNext = static_cast<VkBaseOutStructure *>(head);
        head = entry.feature;
    }
    _device_head = head;
    return head;
}

luisa::span<const char *const>
OptionalExtensionRegistry::enabled_extension_names() const noexcept {
    return luisa::span<const char *const>{_enabled_names.data(),
                                          _enabled_names.size()};
}

bool OptionalExtensionRegistry::supported(OptionalExtension ext) const noexcept {
    return ext != OptionalExtension::COUNT &&
           (_supported_bits & (uint64_t{1u} << static_cast<uint32_t>(ext))) != 0u;
}

bool OptionalExtensionRegistry::requested(OptionalExtension ext) const noexcept {
    return ext != OptionalExtension::COUNT &&
           (_requested_bits & (uint64_t{1u} << static_cast<uint32_t>(ext))) != 0u;
}

bool OptionalExtensionRegistry::enabled(OptionalExtension ext) const noexcept {
    return ext != OptionalExtension::COUNT &&
           (_enabled_bits & (uint64_t{1u} << static_cast<uint32_t>(ext))) != 0u;
}

OptionalExtensionState const &OptionalExtensionRegistry::state(
    OptionalExtension ext) const noexcept {
    if (ext == OptionalExtension::COUNT) { ext = OptionalExtension::DESCRIPTOR_HEAP; }
    return _states[static_cast<size_t>(ext)];
}

luisa::span<const OptionalExtensionState>
OptionalExtensionRegistry::states() const noexcept {
    return luisa::span<const OptionalExtensionState>{
        _states, optional_extension_detail::descriptor_count};
}

bool OptionalExtensionRegistry::force_enable(OptionalExtension ext) noexcept {
    if (!supported(ext)) { return false; }
    for (auto &entry : _entries) {
        if (entry.desc.id != ext) { continue; }
        if (entry.enabled) { return true; }
        // `select()` chains a feature structure only for an extension the config
        // asked for, so an extension that was switched off has no probed feature
        // state at all. Enabling it here would append its name to
        // VkDeviceCreateInfo::ppEnabledExtensionNames while every bit of its
        // VkPhysicalDevice*Features* structure stays VK_FALSE - a logical device
        // that exposes the extension but not the feature it exists for. The
        // structure cannot be probed any more either (the single
        // vkGetPhysicalDeviceFeatures2 call already ran), so fail closed.
        if (entry.desc.make_feature != nullptr) {
            if (entry.feature == nullptr) { return false; }
            // The structure exists, so the probe answered: refuse when the
            // device did not report every bit the backend needs.
            if (entry.desc.check_and_enable != nullptr &&
                !entry.desc.check_and_enable(entry.feature)) {
                return false;
            }
        }
        entry.enabled = true;
        entry.requested = true;
        _requested_bits |= uint64_t{1u} << static_cast<uint32_t>(ext);
        _enabled_bits |= uint64_t{1u} << static_cast<uint32_t>(ext);
        if (entry.name != nullptr && !entry.desc.instance_level) {
            _enabled_names.emplace_back(entry.name);
        }
        refresh_states();
        return true;
    }
    return false;
}

luisa::string OptionalExtensionRegistry::status_string() const noexcept {
    luisa::string result;
    for (auto const &state : states()) {
        if (!state.supported_by_device && !state.requested_by_config &&
            !state.enabled) {
            continue;
        }
        result.append(state.name == nullptr ? "unknown" : state.name)
            .append(state.enabled ?
                        " [enabled]" :
                        (state.supported_by_device ? " [supported, off]" :
                                                     " [unsupported]"))
            .append("\n");
    }
    return result;
}

void OptionalExtensionRegistry::refresh_states() noexcept {
    for (auto i = 0u; i < _entries.size(); ++i) {
        auto const &entry = _entries[i];
        auto &out = _states[i];
        out.id = entry.desc.id;
        out.name = entry.name;
        out.supported_by_device = entry.supported;
        out.requested_by_config = entry.requested;
        out.enabled = entry.enabled;
    }
}

}// namespace lc::vk::detail
