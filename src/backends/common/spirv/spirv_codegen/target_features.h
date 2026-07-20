#pragma once

#include <cstdint>

#include <luisa/xir/op.h>

#include "target_feature_mask.h"

namespace lc::spirv {

// This is a code-generation preference, not a Vulkan feature bit. Keep the
// physical/logical-device feature snapshot truthful and let the Vulkan
// backend select this policy independently from extension enablement.
enum class SpirvBufferFloat32AtomicRmwPolicy : uint8_t {
    NATIVE_IF_SUPPORTED,
    PREFER_WORD_CAS,
};

// These bits describe features enabled on the target logical device, not just
// features supported by its physical device. For features represented here,
// codegen may emit optional SPIR-V instructions and capabilities only when the
// corresponding bit is true.
struct SpirvTargetFeatures {
    bool sampled_image_array_dynamic_indexing{false};
    bool sampled_image_array_non_uniform_indexing{false};
    bool sampler_anisotropy{false};
    bool shader_resource_min_lod{false};
    bool storage_image_read_without_format{false};
    bool storage_image_write_without_format{false};
    bool ray_query{false};
    bool shader_float8{false};
    bool shader_float16{false};
    bool shader_float64{false};
    bool shader_int8{false};
    bool shader_int16{false};
    bool shader_int64{false};
    bool storage_buffer_8bit_access{false};
    bool uniform_storage_buffer_8bit_access{false};
    bool storage_buffer_16bit_access{false};
    bool uniform_storage_buffer_16bit_access{false};
    bool shader_buffer_float32_atomics{false};
    bool shader_buffer_float32_atomic_add{false};
    bool shader_buffer_float32_atomic_min_max{false};
    bool shader_shared_float32_atomics{false};
    bool shader_shared_float32_atomic_add{false};
    bool shader_shared_float32_atomic_min_max{false};
    bool subgroup_basic{false};
    bool subgroup_vote{false};
    bool subgroup_arithmetic{false};
    bool subgroup_ballot{false};
    bool subgroup_shuffle{false};
    bool subgroup_extended_types{false};
    bool shader_buffer_int64_atomics{false};
    bool shader_shared_int64_atomics{false};
    bool descriptor_indexing{false};
    bool runtime_descriptor_array{false};
    bool descriptor_binding_partially_bound{false};
    bool storage_buffer_array_non_uniform_indexing{false};
    bool descriptor_binding_sampled_image_update_after_bind{false};
    bool descriptor_binding_storage_buffer_update_after_bind{false};
    bool storage_buffer_array_dynamic_indexing{false};
    SpirvBufferFloat32AtomicRmwPolicy buffer_float32_atomic_rmw_policy{
        SpirvBufferFloat32AtomicRmwPolicy::NATIVE_IF_SUPPORTED};

    [[nodiscard]] constexpr SpirvTargetFeatureMask enabled_mask() const noexcept {
        SpirvTargetFeatureMask mask{};
        auto enable = [&mask](bool enabled,
                              SpirvTargetFeatureMask bit) noexcept {
            if (enabled) { mask |= bit; }
        };
        enable(sampled_image_array_dynamic_indexing,
               target_feature::sampled_image_array_dynamic_indexing);
        enable(sampled_image_array_non_uniform_indexing,
               target_feature::sampled_image_array_non_uniform_indexing);
        enable(shader_resource_min_lod,
               target_feature::shader_resource_min_lod);
        enable(shader_buffer_float32_atomics,
               target_feature::shader_buffer_float32_atomics);
        enable(shader_buffer_float32_atomic_add,
               target_feature::shader_buffer_float32_atomic_add);
        enable(shader_buffer_float32_atomic_min_max,
               target_feature::shader_buffer_float32_atomic_min_max);
        enable(shader_shared_float32_atomics,
               target_feature::shader_shared_float32_atomics);
        enable(shader_shared_float32_atomic_add,
               target_feature::shader_shared_float32_atomic_add);
        enable(shader_shared_float32_atomic_min_max,
               target_feature::shader_shared_float32_atomic_min_max);
        enable(subgroup_basic, target_feature::subgroup_basic);
        enable(subgroup_vote, target_feature::subgroup_vote);
        enable(subgroup_arithmetic, target_feature::subgroup_arithmetic);
        enable(subgroup_ballot, target_feature::subgroup_ballot);
        enable(subgroup_shuffle, target_feature::subgroup_shuffle);
        enable(subgroup_extended_types,
               target_feature::subgroup_extended_types);
        enable(storage_image_read_without_format,
               target_feature::storage_image_read_without_format);
        enable(storage_image_write_without_format,
               target_feature::storage_image_write_without_format);
        enable(shader_float8, target_feature::shader_float8);
        enable(shader_float16, target_feature::shader_float16);
        enable(shader_float64, target_feature::shader_float64);
        enable(shader_int8, target_feature::shader_int8);
        enable(shader_int16, target_feature::shader_int16);
        enable(shader_int64, target_feature::shader_int64);
        enable(storage_buffer_8bit_access,
               target_feature::storage_buffer_8bit_access);
        enable(uniform_storage_buffer_8bit_access,
               target_feature::uniform_storage_buffer_8bit_access);
        enable(storage_buffer_16bit_access,
               target_feature::storage_buffer_16bit_access);
        enable(uniform_storage_buffer_16bit_access,
               target_feature::uniform_storage_buffer_16bit_access);
        enable(ray_query, target_feature::ray_query);
        enable(sampler_anisotropy, target_feature::sampler_anisotropy);
        enable(shader_buffer_int64_atomics,
               target_feature::shader_buffer_int64_atomics);
        enable(shader_shared_int64_atomics,
               target_feature::shader_shared_int64_atomics);
        enable(descriptor_indexing, target_feature::descriptor_indexing);
        enable(runtime_descriptor_array,
               target_feature::runtime_descriptor_array);
        enable(descriptor_binding_partially_bound,
               target_feature::descriptor_binding_partially_bound);
        enable(storage_buffer_array_non_uniform_indexing,
               target_feature::storage_buffer_array_non_uniform_indexing);
        enable(descriptor_binding_sampled_image_update_after_bind,
               target_feature::descriptor_binding_sampled_image_update_after_bind);
        enable(descriptor_binding_storage_buffer_update_after_bind,
               target_feature::descriptor_binding_storage_buffer_update_after_bind);
        enable(storage_buffer_array_dynamic_indexing,
               target_feature::storage_buffer_array_dynamic_indexing);
        return mask;
    }

    [[nodiscard]] static constexpr SpirvTargetFeatures from_enabled_mask(
        SpirvTargetFeatureMask mask) noexcept {
        auto has = [mask](SpirvTargetFeatureMask bit) noexcept {
            return (mask & bit) != 0u;
        };
        return {
            .sampled_image_array_dynamic_indexing =
                has(target_feature::sampled_image_array_dynamic_indexing),
            .sampled_image_array_non_uniform_indexing =
                has(target_feature::sampled_image_array_non_uniform_indexing),
            .sampler_anisotropy = has(target_feature::sampler_anisotropy),
            .shader_resource_min_lod =
                has(target_feature::shader_resource_min_lod),
            .storage_image_read_without_format =
                has(target_feature::storage_image_read_without_format),
            .storage_image_write_without_format =
                has(target_feature::storage_image_write_without_format),
            .ray_query = has(target_feature::ray_query),
            .shader_float8 = has(target_feature::shader_float8),
            .shader_float16 = has(target_feature::shader_float16),
            .shader_float64 = has(target_feature::shader_float64),
            .shader_int8 = has(target_feature::shader_int8),
            .shader_int16 = has(target_feature::shader_int16),
            .shader_int64 = has(target_feature::shader_int64),
            .storage_buffer_8bit_access =
                has(target_feature::storage_buffer_8bit_access),
            .uniform_storage_buffer_8bit_access =
                has(target_feature::uniform_storage_buffer_8bit_access),
            .storage_buffer_16bit_access =
                has(target_feature::storage_buffer_16bit_access),
            .uniform_storage_buffer_16bit_access =
                has(target_feature::uniform_storage_buffer_16bit_access),
            .shader_buffer_float32_atomics =
                has(target_feature::shader_buffer_float32_atomics),
            .shader_buffer_float32_atomic_add =
                has(target_feature::shader_buffer_float32_atomic_add),
            .shader_buffer_float32_atomic_min_max =
                has(target_feature::shader_buffer_float32_atomic_min_max),
            .shader_shared_float32_atomics =
                has(target_feature::shader_shared_float32_atomics),
            .shader_shared_float32_atomic_add =
                has(target_feature::shader_shared_float32_atomic_add),
            .shader_shared_float32_atomic_min_max =
                has(target_feature::shader_shared_float32_atomic_min_max),
            .subgroup_basic = has(target_feature::subgroup_basic),
            .subgroup_vote = has(target_feature::subgroup_vote),
            .subgroup_arithmetic = has(target_feature::subgroup_arithmetic),
            .subgroup_ballot = has(target_feature::subgroup_ballot),
            .subgroup_shuffle = has(target_feature::subgroup_shuffle),
            .subgroup_extended_types =
                has(target_feature::subgroup_extended_types),
            .shader_buffer_int64_atomics =
                has(target_feature::shader_buffer_int64_atomics),
            .shader_shared_int64_atomics =
                has(target_feature::shader_shared_int64_atomics),
            .descriptor_indexing = has(target_feature::descriptor_indexing),
            .runtime_descriptor_array =
                has(target_feature::runtime_descriptor_array),
            .descriptor_binding_partially_bound =
                has(target_feature::descriptor_binding_partially_bound),
            .storage_buffer_array_non_uniform_indexing =
                has(target_feature::storage_buffer_array_non_uniform_indexing),
            .descriptor_binding_sampled_image_update_after_bind =
                has(target_feature::descriptor_binding_sampled_image_update_after_bind),
            .descriptor_binding_storage_buffer_update_after_bind =
                has(target_feature::descriptor_binding_storage_buffer_update_after_bind),
            .storage_buffer_array_dynamic_indexing =
                has(target_feature::storage_buffer_array_dynamic_indexing)};
    }
};

enum class SpirvFloatAtomicStorage : uint8_t {
    BUFFER,
    SHARED,
};

enum class SpirvFloatAtomicImplementation : uint8_t {
    WORD_EXCHANGE,
    WORD_COMPARE_EXCHANGE,
    WORD_CAS,
    NATIVE_EXCHANGE,
    NATIVE_ADD,
    NATIVE_MIN_MAX,
    UNSUPPORTED_WIDTH,
    UNSUPPORTED_OPERATION,
    UNSUPPORTED_REPRESENTATION,
    UNSUPPORTED_FEATURE,
};

[[nodiscard]] constexpr SpirvFloatAtomicImplementation plan_spirv_float_atomic(
    luisa::compute::xir::AtomicOp op, uint32_t bit_width,
    SpirvFloatAtomicStorage storage,
    SpirvTargetFeatures features) noexcept {
    namespace xir = luisa::compute::xir;
    if (bit_width != 32u) {
        return SpirvFloatAtomicImplementation::UNSUPPORTED_WIDTH;
    }
    if (storage == SpirvFloatAtomicStorage::BUFFER) {
        switch (op) {
            case xir::AtomicOp::EXCHANGE:
                return features.shader_buffer_float32_atomics ?
                           SpirvFloatAtomicImplementation::NATIVE_EXCHANGE :
                           SpirvFloatAtomicImplementation::WORD_EXCHANGE;
            case xir::AtomicOp::COMPARE_EXCHANGE:
                return SpirvFloatAtomicImplementation::WORD_COMPARE_EXCHANGE;
            case xir::AtomicOp::FETCH_ADD:
            case xir::AtomicOp::FETCH_SUB:
                return features.buffer_float32_atomic_rmw_policy ==
                               SpirvBufferFloat32AtomicRmwPolicy::
                                   NATIVE_IF_SUPPORTED &&
                           features.shader_buffer_float32_atomic_add ?
                           SpirvFloatAtomicImplementation::NATIVE_ADD :
                           SpirvFloatAtomicImplementation::WORD_CAS;
            case xir::AtomicOp::FETCH_MIN:
            case xir::AtomicOp::FETCH_MAX:
                return features.buffer_float32_atomic_rmw_policy ==
                               SpirvBufferFloat32AtomicRmwPolicy::
                                   NATIVE_IF_SUPPORTED &&
                           features.shader_buffer_float32_atomic_min_max ?
                           SpirvFloatAtomicImplementation::NATIVE_MIN_MAX :
                           SpirvFloatAtomicImplementation::WORD_CAS;
            default:
                return SpirvFloatAtomicImplementation::UNSUPPORTED_OPERATION;
        }
    }
    switch (op) {
        case xir::AtomicOp::EXCHANGE:
            return features.shader_shared_float32_atomics ?
                       SpirvFloatAtomicImplementation::NATIVE_EXCHANGE :
                       SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE;
        case xir::AtomicOp::FETCH_ADD:
        case xir::AtomicOp::FETCH_SUB:
            return features.shader_shared_float32_atomic_add ?
                       SpirvFloatAtomicImplementation::NATIVE_ADD :
                       SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE;
        case xir::AtomicOp::FETCH_MIN:
        case xir::AtomicOp::FETCH_MAX:
            return features.shader_shared_float32_atomic_min_max ?
                       SpirvFloatAtomicImplementation::NATIVE_MIN_MAX :
                       SpirvFloatAtomicImplementation::UNSUPPORTED_FEATURE;
        case xir::AtomicOp::COMPARE_EXCHANGE:
            return SpirvFloatAtomicImplementation::UNSUPPORTED_REPRESENTATION;
        default:
            return SpirvFloatAtomicImplementation::UNSUPPORTED_OPERATION;
    }
}

// Plans solely from the enabled Vulkan feature bits. Atomic-buffer planning
// uses this to distinguish a required word fallback from a vendor preference:
// a typed int64 atomic in the same Buffer<T> must win over a word-CAS
// preference, while a genuinely unavailable native operation still conflicts.
[[nodiscard]] constexpr SpirvFloatAtomicImplementation
plan_spirv_float_atomic_capability_driven(
    luisa::compute::xir::AtomicOp op, uint32_t bit_width,
    SpirvFloatAtomicStorage storage,
    SpirvTargetFeatures features) noexcept {
    features.buffer_float32_atomic_rmw_policy =
        SpirvBufferFloat32AtomicRmwPolicy::NATIVE_IF_SUPPORTED;
    return plan_spirv_float_atomic(
        op, bit_width, storage, features);
}

[[nodiscard]] constexpr bool spirv_float_atomic_implementation_is_native(
    SpirvFloatAtomicImplementation implementation) noexcept {
    return implementation == SpirvFloatAtomicImplementation::NATIVE_EXCHANGE ||
           implementation == SpirvFloatAtomicImplementation::NATIVE_ADD ||
           implementation == SpirvFloatAtomicImplementation::NATIVE_MIN_MAX;
}

enum class SpirvAtomicBufferStoragePlan : uint8_t {
    TYPED,
    WORD,
    CONFLICT,
};

struct SpirvAtomicBufferStorageRequirements {
    bool contains_bool{false};
    bool has_float32_word_fallback{false};
    bool prefers_float32_word_fallback{false};
    bool has_int64_atomic{false};
};

[[nodiscard]] constexpr SpirvAtomicBufferStoragePlan
plan_spirv_atomic_buffer_storage(
    SpirvAtomicBufferStorageRequirements requirements) noexcept {
    auto requires_word = requirements.contains_bool ||
                         requirements.has_float32_word_fallback;
    auto requires_typed = requirements.has_int64_atomic;
    if (requires_word && requires_typed) {
        return SpirvAtomicBufferStoragePlan::CONFLICT;
    }
    if (requires_typed) { return SpirvAtomicBufferStoragePlan::TYPED; }
    if (requires_word) { return SpirvAtomicBufferStoragePlan::WORD; }
    if (requirements.prefers_float32_word_fallback) {
        return SpirvAtomicBufferStoragePlan::WORD;
    }
    return SpirvAtomicBufferStoragePlan::TYPED;
}

}// namespace lc::spirv
