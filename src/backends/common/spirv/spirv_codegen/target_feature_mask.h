#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

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

inline constexpr SpirvTargetFeatureMask known_mask = 0x0000003fffffffffull;

}// namespace target_feature

struct SpirvTargetFeatureDescription {
    SpirvTargetFeatureMask bit{};
    std::string_view name{};
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
    SpirvTargetFeatureDescription{target_feature::storage_buffer_array_dynamic_indexing, "shaderStorageBufferArrayDynamicIndexing"}};

[[nodiscard]] constexpr std::string_view spirv_target_feature_name(
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
