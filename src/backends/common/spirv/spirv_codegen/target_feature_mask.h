#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <luisa/core/stl/string.h>

namespace lc::spirv {

// Persistent SPIR-V artifacts store target requirements as this named bitset.
// The values are an ABI: append new bits, but never reorder or reuse old ones.
// In particular, do not serialize SpirvTargetFeatures itself or derive these
// values from its bool-field order.
using SpirvTargetFeatureMask = uint64_t;

namespace target_feature {

inline constexpr SpirvTargetFeatureMask sampled_image_array_dynamic_indexing = 0x0000000000000001ull;
inline constexpr SpirvTargetFeatureMask sampled_image_array_non_uniform_indexing = 0x0000000000000002ull;
inline constexpr SpirvTargetFeatureMask shader_resource_min_lod = 0x0000000000000004ull;
inline constexpr SpirvTargetFeatureMask shader_buffer_float32_atomics = 0x0000000000000008ull;
inline constexpr SpirvTargetFeatureMask shader_buffer_float32_atomic_add = 0x0000000000000010ull;
inline constexpr SpirvTargetFeatureMask shader_buffer_float32_atomic_min_max = 0x0000000000000020ull;
inline constexpr SpirvTargetFeatureMask shader_shared_float32_atomics = 0x0000000000000040ull;
inline constexpr SpirvTargetFeatureMask shader_shared_float32_atomic_add = 0x0000000000000080ull;
inline constexpr SpirvTargetFeatureMask shader_shared_float32_atomic_min_max = 0x0000000000000100ull;
inline constexpr SpirvTargetFeatureMask subgroup_basic = 0x0000000000000200ull;
inline constexpr SpirvTargetFeatureMask subgroup_vote = 0x0000000000000400ull;
inline constexpr SpirvTargetFeatureMask subgroup_arithmetic = 0x0000000000000800ull;
inline constexpr SpirvTargetFeatureMask subgroup_ballot = 0x0000000000001000ull;
inline constexpr SpirvTargetFeatureMask subgroup_shuffle = 0x0000000000002000ull;
inline constexpr SpirvTargetFeatureMask subgroup_extended_types = 0x0000000000004000ull;
inline constexpr SpirvTargetFeatureMask storage_image_read_without_format = 0x0000000000008000ull;
inline constexpr SpirvTargetFeatureMask storage_image_write_without_format = 0x0000000000010000ull;
inline constexpr SpirvTargetFeatureMask shader_float8 = 0x0000000000020000ull;
inline constexpr SpirvTargetFeatureMask shader_float16 = 0x0000000000040000ull;
inline constexpr SpirvTargetFeatureMask shader_float64 = 0x0000000000080000ull;
inline constexpr SpirvTargetFeatureMask shader_int8 = 0x0000000000100000ull;
inline constexpr SpirvTargetFeatureMask shader_int16 = 0x0000000000200000ull;
inline constexpr SpirvTargetFeatureMask shader_int64 = 0x0000000000400000ull;
inline constexpr SpirvTargetFeatureMask storage_buffer_8bit_access = 0x0000000000800000ull;
inline constexpr SpirvTargetFeatureMask uniform_storage_buffer_8bit_access = 0x0000000001000000ull;
inline constexpr SpirvTargetFeatureMask storage_buffer_16bit_access = 0x0000000002000000ull;
inline constexpr SpirvTargetFeatureMask uniform_storage_buffer_16bit_access = 0x0000000004000000ull;
inline constexpr SpirvTargetFeatureMask ray_query = 0x0000000008000000ull;
inline constexpr SpirvTargetFeatureMask sampler_anisotropy = 0x0000000010000000ull;
inline constexpr SpirvTargetFeatureMask shader_buffer_int64_atomics = 0x0000000020000000ull;
inline constexpr SpirvTargetFeatureMask shader_shared_int64_atomics = 0x0000000040000000ull;
inline constexpr SpirvTargetFeatureMask descriptor_indexing = 0x0000000080000000ull;
inline constexpr SpirvTargetFeatureMask runtime_descriptor_array = 0x0000000100000000ull;
inline constexpr SpirvTargetFeatureMask descriptor_binding_partially_bound = 0x0000000200000000ull;
inline constexpr SpirvTargetFeatureMask storage_buffer_array_non_uniform_indexing = 0x0000000400000000ull;
inline constexpr SpirvTargetFeatureMask descriptor_binding_sampled_image_update_after_bind = 0x0000000800000000ull;
inline constexpr SpirvTargetFeatureMask descriptor_binding_storage_buffer_update_after_bind = 0x0000001000000000ull;
inline constexpr SpirvTargetFeatureMask storage_buffer_array_dynamic_indexing = 0x0000002000000000ull;
inline constexpr SpirvTargetFeatureMask shader_device_clock = 0x0000004000000000ull;
inline constexpr SpirvTargetFeatureMask buffer_device_address = 0x0000008000000000ull;
inline constexpr SpirvTargetFeatureMask shader_untyped_pointers = 0x0000010000000000ull;
inline constexpr SpirvTargetFeatureMask cooperative_vector = 0x0000020000000000ull;
inline constexpr SpirvTargetFeatureMask shader_bfloat16 = 0x0000040000000000ull;

// ---- optional shader capability extensions -------------------------------
// One bit per negotiated shader capability extension of the Vulkan backend
// (see src/backends/vk/vk_shader_capabilities.h). A bit may only be required by
// an artifact that really contains the corresponding SPIR-V declaration, so
// codegen records it at the exact emission site through that planner instead of
// deriving it from the device.
//
// VK_KHR_shader_fma: OpFmaKHR (SPV_KHR_fma). The backend negotiates only the
// shaderFmaFloat32 bit, so the requirement is exactly the float32 form.
inline constexpr SpirvTargetFeatureMask shader_fma_float32 = 0x0000080000000000ull;
// VK_EXT_shader_64bit_indexing: the Shader64BitIndexingEXT capability and
// execution mode of SPV_EXT_shader_64bit_indexing.
inline constexpr SpirvTargetFeatureMask shader_64bit_indexing = 0x0000100000000000ull;
// VK_EXT_shader_uniform_buffer_unsized_array: OpTypeRuntimeArray as the last
// member of a uniform block. The extension reuses existing SPIR-V capabilities,
// so this bit describes a Vulkan feature requirement with no OpCapability.
inline constexpr SpirvTargetFeatureMask shader_uniform_buffer_unsized_array = 0x0000200000000000ull;
// VK_EXT_shader_long_vector: LongVectorEXT vectors longer than four components.
inline constexpr SpirvTargetFeatureMask shader_long_vector = 0x0000400000000000ull;
// VK_EXT_shader_subgroup_partitioned: GroupNonUniformPartitionedEXT and
// OpGroupNonUniformPartitionEXT.
inline constexpr SpirvTargetFeatureMask shader_subgroup_partitioned = 0x0000800000000000ull;
// VK_EXT_shader_split_barrier: SplitBarrierEXT with
// OpControlBarrierArriveEXT/OpControlBarrierWaitEXT.
inline constexpr SpirvTargetFeatureMask shader_split_barrier = 0x0001000000000000ull;
// VK_EXT_shader_ocp_microscaling_types: the OCP microscaling scalar types
// (Float4EXT/Float6EXT/Float8UnsignedE8M0EXT/MXInt8EXT). The extension enables
// whichever of those physical feature bits the driver reports.
inline constexpr SpirvTargetFeatureMask shader_ocp_microscaling_types = 0x0002000000000000ull;
// VK_KHR_shader_constant_data: ConstantDataKHR with OpConstantDataKHR /
// OpSpecConstantDataKHR.
inline constexpr SpirvTargetFeatureMask shader_constant_data = 0x0004000000000000ull;
// VK_KHR_shader_abort: AbortKHR with OpAbortKHR.
inline constexpr SpirvTargetFeatureMask shader_abort = 0x0008000000000000ull;
// VK_KHR_opacity_micromap, KHR spelling only: a module that traverses an
// acceleration structure carrying an opacity micromap must declare
// `OpCapability RayTracingOpacityMicromapExecutionModeKHR` and
// `OpExecutionModeId OpacityMicromapIdKHR` with an OpSpecConstantTrue operand.
// VK_EXT_opacity_micromap (the alias) needs neither, so the bit is only ever set
// when the KHR name is the negotiated one.
inline constexpr SpirvTargetFeatureMask ray_tracing_opacity_micromap = 0x0010000000000000ull;
// VK_VALVE_shader_mixed_float_dot_product: DotProductFloat16AccFloat32VALVE
// with OpFDot2MixAcc32VALVE (SPV_VALVE_mixed_float_dot_product). The backend
// negotiates only the shaderMixedFloatDotProductFloat16AccFloat32 bit, so the
// requirement is exactly the FP16-input/FP32-accumulator form.
inline constexpr SpirvTargetFeatureMask shader_mixed_float_dot_product_float16_acc_float32 = 0x0020000000000000ull;

inline constexpr SpirvTargetFeatureMask known_mask =
    0x000000ffffffffffull | shader_untyped_pointers | cooperative_vector |
    shader_bfloat16 | shader_fma_float32 | shader_64bit_indexing |
    shader_uniform_buffer_unsized_array | shader_long_vector |
    shader_subgroup_partitioned | shader_split_barrier |
    shader_ocp_microscaling_types | shader_constant_data | shader_abort |
    ray_tracing_opacity_micromap |
    shader_mixed_float_dot_product_float16_acc_float32;

}// namespace target_feature

struct SpirvTargetFeatureDescription {
    SpirvTargetFeatureMask bit{};
    luisa::string_view name{};
};

inline constexpr std::array spirv_target_feature_descriptions{
    SpirvTargetFeatureDescription{target_feature::sampled_image_array_dynamic_indexing, "shaderSampledImageArrayDynamicIndexing"},
    SpirvTargetFeatureDescription{target_feature::sampled_image_array_non_uniform_indexing, "shaderSampledImageArrayNonUniformIndexing"},
    SpirvTargetFeatureDescription{target_feature::shader_resource_min_lod, "shaderResourceMinLod"},
    SpirvTargetFeatureDescription{target_feature::shader_buffer_float32_atomics, "shaderBufferFloat32Atomics"},
    SpirvTargetFeatureDescription{target_feature::shader_buffer_float32_atomic_add, "shaderBufferFloat32AtomicAdd"},
    SpirvTargetFeatureDescription{target_feature::shader_buffer_float32_atomic_min_max, "shaderBufferFloat32AtomicMinMax"},
    SpirvTargetFeatureDescription{target_feature::shader_shared_float32_atomics, "shaderSharedFloat32Atomics"},
    SpirvTargetFeatureDescription{target_feature::shader_shared_float32_atomic_add, "shaderSharedFloat32AtomicAdd"},
    SpirvTargetFeatureDescription{target_feature::shader_shared_float32_atomic_min_max, "shaderSharedFloat32AtomicMinMax"},
    SpirvTargetFeatureDescription{target_feature::subgroup_basic, "subgroupBasic"},
    SpirvTargetFeatureDescription{target_feature::subgroup_vote, "subgroupVote"},
    SpirvTargetFeatureDescription{target_feature::subgroup_arithmetic, "subgroupArithmetic"},
    SpirvTargetFeatureDescription{target_feature::subgroup_ballot, "subgroupBallot"},
    SpirvTargetFeatureDescription{target_feature::subgroup_shuffle, "subgroupShuffle"},
    SpirvTargetFeatureDescription{target_feature::subgroup_extended_types, "shaderSubgroupExtendedTypes"},
    SpirvTargetFeatureDescription{target_feature::storage_image_read_without_format, "shaderStorageImageReadWithoutFormat"},
    SpirvTargetFeatureDescription{target_feature::storage_image_write_without_format, "shaderStorageImageWriteWithoutFormat"},
    SpirvTargetFeatureDescription{target_feature::shader_float8, "shaderFloat8"},
    SpirvTargetFeatureDescription{target_feature::shader_float16, "shaderFloat16"},
    SpirvTargetFeatureDescription{target_feature::shader_float64, "shaderFloat64"},
    SpirvTargetFeatureDescription{target_feature::shader_int8, "shaderInt8"},
    SpirvTargetFeatureDescription{target_feature::shader_int16, "shaderInt16"},
    SpirvTargetFeatureDescription{target_feature::shader_int64, "shaderInt64"},
    SpirvTargetFeatureDescription{target_feature::storage_buffer_8bit_access, "storageBuffer8BitAccess"},
    SpirvTargetFeatureDescription{target_feature::uniform_storage_buffer_8bit_access, "uniformAndStorageBuffer8BitAccess"},
    SpirvTargetFeatureDescription{target_feature::storage_buffer_16bit_access, "storageBuffer16BitAccess"},
    SpirvTargetFeatureDescription{target_feature::uniform_storage_buffer_16bit_access, "uniformAndStorageBuffer16BitAccess"},
    SpirvTargetFeatureDescription{target_feature::ray_query, "rayQuery"},
    SpirvTargetFeatureDescription{target_feature::sampler_anisotropy, "samplerAnisotropy"},
    SpirvTargetFeatureDescription{target_feature::shader_buffer_int64_atomics, "shaderBufferInt64Atomics"},
    SpirvTargetFeatureDescription{target_feature::shader_shared_int64_atomics, "shaderSharedInt64Atomics"},
    SpirvTargetFeatureDescription{target_feature::descriptor_indexing, "descriptorIndexing"},
    SpirvTargetFeatureDescription{target_feature::runtime_descriptor_array, "runtimeDescriptorArray"},
    SpirvTargetFeatureDescription{target_feature::descriptor_binding_partially_bound, "descriptorBindingPartiallyBound"},
    SpirvTargetFeatureDescription{target_feature::storage_buffer_array_non_uniform_indexing, "shaderStorageBufferArrayNonUniformIndexing"},
    SpirvTargetFeatureDescription{target_feature::descriptor_binding_sampled_image_update_after_bind, "descriptorBindingSampledImageUpdateAfterBind"},
    SpirvTargetFeatureDescription{target_feature::descriptor_binding_storage_buffer_update_after_bind, "descriptorBindingStorageBufferUpdateAfterBind"},
    SpirvTargetFeatureDescription{target_feature::storage_buffer_array_dynamic_indexing, "shaderStorageBufferArrayDynamicIndexing"},
    SpirvTargetFeatureDescription{target_feature::shader_device_clock, "shaderDeviceClock"},
    SpirvTargetFeatureDescription{target_feature::buffer_device_address, "bufferDeviceAddress"},
    SpirvTargetFeatureDescription{target_feature::shader_untyped_pointers, "shaderUntypedPointers"},
    SpirvTargetFeatureDescription{target_feature::cooperative_vector, "cooperativeVector"},
    SpirvTargetFeatureDescription{target_feature::shader_bfloat16, "shaderBFloat16Type"},
    SpirvTargetFeatureDescription{target_feature::shader_fma_float32, "shaderFmaFloat32"},
    SpirvTargetFeatureDescription{target_feature::shader_64bit_indexing, "shader64BitIndexing"},
    SpirvTargetFeatureDescription{target_feature::shader_uniform_buffer_unsized_array, "shaderUniformBufferUnsizedArray"},
    SpirvTargetFeatureDescription{target_feature::shader_long_vector, "shaderLongVector"},
    SpirvTargetFeatureDescription{target_feature::shader_subgroup_partitioned, "shaderSubgroupPartitioned"},
    SpirvTargetFeatureDescription{target_feature::shader_split_barrier, "shaderSplitBarrier"},
    SpirvTargetFeatureDescription{target_feature::shader_ocp_microscaling_types, "shaderOCPMicroscalingTypes"},
    SpirvTargetFeatureDescription{target_feature::shader_constant_data, "shaderConstantData"},
    SpirvTargetFeatureDescription{target_feature::shader_abort, "shaderAbort"},
    SpirvTargetFeatureDescription{target_feature::ray_tracing_opacity_micromap, "rayTracingOpacityMicromapExecutionMode"},
    SpirvTargetFeatureDescription{target_feature::shader_mixed_float_dot_product_float16_acc_float32, "shaderMixedFloatDotProductFloat16AccFloat32"}};

[[nodiscard]] constexpr luisa::string_view spirv_target_feature_name(
    SpirvTargetFeatureMask bit) noexcept {
    for (auto feature : spirv_target_feature_descriptions) {
        if (feature.bit == bit) { return feature.name; }
    }
    return {};
}

struct SpirvTargetFeatureList {
    std::array<SpirvTargetFeatureDescription,
               spirv_target_feature_descriptions.size()>
        features{};
    std::size_t count{};
    SpirvTargetFeatureMask unknown_bits{};

    [[nodiscard]] constexpr auto begin() const noexcept {
        return features.begin();
    }
    [[nodiscard]] constexpr auto end() const noexcept {
        return features.begin() + count;
    }
};

[[nodiscard]] constexpr SpirvTargetFeatureList list_spirv_target_features(
    SpirvTargetFeatureMask mask) noexcept {
    SpirvTargetFeatureList result{
        .unknown_bits = mask & ~target_feature::known_mask};
    for (auto feature : spirv_target_feature_descriptions) {
        if ((mask & feature.bit) != 0u) {
            result.features[result.count++] = feature;
        }
    }
    return result;
}

struct SpirvTargetFeatureRequirementCheck {
    SpirvTargetFeatureMask unknown_required_bits{};
    SpirvTargetFeatureMask missing_required_bits{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return unknown_required_bits == 0u &&
               missing_required_bits == 0u;
    }
};

[[nodiscard]] constexpr SpirvTargetFeatureRequirementCheck
check_spirv_target_feature_requirements(
    SpirvTargetFeatureMask required,
    SpirvTargetFeatureMask enabled) noexcept {
    auto known_required = required & target_feature::known_mask;
    return {
        .unknown_required_bits = required & ~target_feature::known_mask,
        .missing_required_bits = known_required & ~enabled};
}

namespace detail {

[[nodiscard]] constexpr bool valid_spirv_target_feature_descriptions() noexcept {
    SpirvTargetFeatureMask described{};
    for (auto feature : spirv_target_feature_descriptions) {
        if (feature.bit == 0u ||
            (feature.bit & (feature.bit - 1u)) != 0u ||
            feature.name.empty() ||
            (described & feature.bit) != 0u) {
            return false;
        }
        described |= feature.bit;
    }
    return described == target_feature::known_mask;
}

}// namespace detail

static_assert(detail::valid_spirv_target_feature_descriptions());

}// namespace lc::spirv
