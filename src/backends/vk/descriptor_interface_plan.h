#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/stl/memory.h>

#include "../common/hlsl/shader_property.h"

namespace lc::vk::detail {

// Property tables are a persisted Vulkan ABI, not merely codegen metadata.
// Keep the accepted form deliberately small and deterministic: dense local
// bindings in set 0, the immutable sampler heap in set 1, then the enabled
// update-after-bind heaps in buffer/2D/3D order.
inline constexpr uint32_t descriptor_interface_max_local_bindings = 256u;
inline constexpr uint32_t descriptor_interface_sampler_count = 16u;
inline constexpr uint32_t descriptor_interface_max_set_count = 5u;
inline constexpr uint32_t descriptor_interface_max_property_count =
    descriptor_interface_max_local_bindings + 1u + 3u + 1u;

enum class DescriptorInterfaceStageMask : uint8_t {
    COMPUTE = 1u,
    RASTER = 2u | 4u,
    RAY_TRACING = 8u
};

enum class DescriptorInterfaceError : uint8_t {
    NONE,
    INVALID_STAGE_MASK,
    TOO_MANY_PROPERTIES,
    TOO_MANY_DESCRIPTOR_SETS,
    BINDLESS_UNAVAILABLE,
    INVALID_PROPERTY_TYPE,
    INVALID_ARRAY_SIZE,
    NONCANONICAL_PUSH_CONSTANT,
    DUPLICATE_PUSH_CONSTANT,
    NONCANONICAL_SAMPLER,
    MISSING_OR_DUPLICATE_SAMPLER,
    NONCANONICAL_LOCAL_BINDING,
    DUPLICATE_LOCAL_BINDING,
    NONDENSE_LOCAL_BINDINGS,
    NONCANONICAL_GLOBAL_HEAP,
    MISSING_OR_DUPLICATE_GLOBAL_HEAP,
    CONSTANT_UBO_MISMATCH,
    INDIRECT_DISPATCH_NOT_SUPPORTED,
    DUPLICATE_INDIRECT_DISPATCH,
    ACCELERATION_STRUCTURE_NOT_SUPPORTED,
    MAX_PER_STAGE_SAMPLERS,
    MAX_PER_STAGE_UNIFORM_BUFFERS,
    MAX_PER_STAGE_STORAGE_BUFFERS,
    MAX_PER_STAGE_SAMPLED_IMAGES,
    MAX_PER_STAGE_STORAGE_IMAGES,
    MAX_PER_STAGE_RESOURCES,
    MAX_PER_STAGE_ACCELERATION_STRUCTURES,
    MAX_DESCRIPTOR_SET_SAMPLERS,
    MAX_DESCRIPTOR_SET_UNIFORM_BUFFERS,
    MAX_DESCRIPTOR_SET_STORAGE_BUFFERS,
    MAX_DESCRIPTOR_SET_SAMPLED_IMAGES,
    MAX_DESCRIPTOR_SET_STORAGE_IMAGES,
    MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES,
    MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLERS,
    MAX_PER_STAGE_UPDATE_AFTER_BIND_STORAGE_BUFFERS,
    MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLED_IMAGES,
    MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES,
    MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS,
    MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_BUFFERS,
    MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLED_IMAGES,
    MAX_PER_STAGE_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES,
    MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES
};

[[nodiscard]] constexpr const char *descriptor_interface_error_name(
    DescriptorInterfaceError error) noexcept {
    switch (error) {
        case DescriptorInterfaceError::NONE: return "none";
        case DescriptorInterfaceError::INVALID_STAGE_MASK:
            return "invalid stage mask";
        case DescriptorInterfaceError::TOO_MANY_PROPERTIES:
            return "too many properties";
        case DescriptorInterfaceError::TOO_MANY_DESCRIPTOR_SETS:
            return "maxBoundDescriptorSets exceeded";
        case DescriptorInterfaceError::BINDLESS_UNAVAILABLE:
            return "bindless heap unavailable";
        case DescriptorInterfaceError::INVALID_PROPERTY_TYPE:
            return "invalid property type";
        case DescriptorInterfaceError::INVALID_ARRAY_SIZE:
            return "invalid descriptor array size";
        case DescriptorInterfaceError::NONCANONICAL_PUSH_CONSTANT:
            return "noncanonical push-constant pseudo-property";
        case DescriptorInterfaceError::DUPLICATE_PUSH_CONSTANT:
            return "duplicate push-constant pseudo-property";
        case DescriptorInterfaceError::NONCANONICAL_SAMPLER:
            return "noncanonical sampler heap";
        case DescriptorInterfaceError::MISSING_OR_DUPLICATE_SAMPLER:
            return "missing or duplicate sampler heap";
        case DescriptorInterfaceError::NONCANONICAL_LOCAL_BINDING:
            return "noncanonical local descriptor binding";
        case DescriptorInterfaceError::DUPLICATE_LOCAL_BINDING:
            return "duplicate local descriptor binding";
        case DescriptorInterfaceError::NONDENSE_LOCAL_BINDINGS:
            return "nondense local descriptor bindings";
        case DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP:
            return "noncanonical global descriptor heap";
        case DescriptorInterfaceError::MISSING_OR_DUPLICATE_GLOBAL_HEAP:
            return "missing or duplicate global descriptor heap";
        case DescriptorInterfaceError::CONSTANT_UBO_MISMATCH:
            return "constant UBO payload/property mismatch";
        case DescriptorInterfaceError::INDIRECT_DISPATCH_NOT_SUPPORTED:
            return "indirect-dispatch descriptor is not supported by this stage";
        case DescriptorInterfaceError::DUPLICATE_INDIRECT_DISPATCH:
            return "duplicate indirect-dispatch descriptor";
        case DescriptorInterfaceError::ACCELERATION_STRUCTURE_NOT_SUPPORTED:
            return "acceleration-structure descriptor is not supported";
        case DescriptorInterfaceError::MAX_PER_STAGE_SAMPLERS:
            return "maxPerStageDescriptorSamplers exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_UNIFORM_BUFFERS:
            return "maxPerStageDescriptorUniformBuffers exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_STORAGE_BUFFERS:
            return "maxPerStageDescriptorStorageBuffers exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_SAMPLED_IMAGES:
            return "maxPerStageDescriptorSampledImages exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_STORAGE_IMAGES:
            return "maxPerStageDescriptorStorageImages exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_RESOURCES:
            return "maxPerStageResources exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_ACCELERATION_STRUCTURES:
            return "maxPerStageDescriptorAccelerationStructures exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_SAMPLERS:
            return "maxDescriptorSetSamplers exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UNIFORM_BUFFERS:
            return "maxDescriptorSetUniformBuffers exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_STORAGE_BUFFERS:
            return "maxDescriptorSetStorageBuffers exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_SAMPLED_IMAGES:
            return "maxDescriptorSetSampledImages exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_STORAGE_IMAGES:
            return "maxDescriptorSetStorageImages exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES:
            return "maxDescriptorSetAccelerationStructures exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLERS:
            return "maxPerStageDescriptorUpdateAfterBindSamplers exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_STORAGE_BUFFERS:
            return "maxPerStageDescriptorUpdateAfterBindStorageBuffers exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLED_IMAGES:
            return "maxPerStageDescriptorUpdateAfterBindSampledImages exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES:
            return "maxPerStageUpdateAfterBindResources exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS:
            return "maxDescriptorSetUpdateAfterBindSamplers exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_BUFFERS:
            return "maxDescriptorSetUpdateAfterBindStorageBuffers exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLED_IMAGES:
            return "maxDescriptorSetUpdateAfterBindSampledImages exceeded";
        case DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES:
            return "maxPerStageDescriptorUpdateAfterBindAccelerationStructures exceeded";
        case DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES:
            return "maxDescriptorSetUpdateAfterBindAccelerationStructures exceeded";
    }
    return "unknown descriptor-interface error";
}

struct DescriptorInterfaceCounts {
    uint64_t samplers{};
    uint64_t uniform_buffers{};
    uint64_t storage_buffers{};
    uint64_t sampled_images{};
    uint64_t storage_images{};
    uint64_t acceleration_structures{};

    [[nodiscard]] constexpr uint64_t resources() const noexcept {
        // Vulkan's maxPerStageResources domains count buffer/image resources,
        // not standalone samplers or acceleration-structure descriptors.
        return uniform_buffers + storage_buffers + sampled_images +
               storage_images;
    }

    constexpr DescriptorInterfaceCounts &operator+=(
        DescriptorInterfaceCounts rhs) noexcept {
        samplers += rhs.samplers;
        uniform_buffers += rhs.uniform_buffers;
        storage_buffers += rhs.storage_buffers;
        sampled_images += rhs.sampled_images;
        storage_images += rhs.storage_images;
        acceleration_structures += rhs.acceleration_structures;
        return *this;
    }
};

struct DescriptorInterfaceLimits {
    uint32_t max_bound_descriptor_sets{};
    uint32_t max_per_stage_descriptor_samplers{};
    uint32_t max_per_stage_descriptor_uniform_buffers{};
    uint32_t max_per_stage_descriptor_storage_buffers{};
    uint32_t max_per_stage_descriptor_sampled_images{};
    uint32_t max_per_stage_descriptor_storage_images{};
    uint32_t max_per_stage_resources{};
    uint32_t max_descriptor_set_samplers{};
    uint32_t max_descriptor_set_uniform_buffers{};
    uint32_t max_descriptor_set_storage_buffers{};
    uint32_t max_descriptor_set_sampled_images{};
    uint32_t max_descriptor_set_storage_images{};
    uint32_t max_per_stage_descriptor_acceleration_structures{};
    uint32_t max_descriptor_set_acceleration_structures{};
    uint32_t max_per_stage_descriptor_update_after_bind_samplers{};
    uint32_t max_per_stage_descriptor_update_after_bind_storage_buffers{};
    uint32_t max_per_stage_descriptor_update_after_bind_sampled_images{};
    uint32_t max_per_stage_update_after_bind_resources{};
    uint32_t max_descriptor_set_update_after_bind_samplers{};
    uint32_t max_descriptor_set_update_after_bind_storage_buffers{};
    uint32_t max_descriptor_set_update_after_bind_sampled_images{};
    uint32_t max_per_stage_descriptor_update_after_bind_acceleration_structures{};
    uint32_t max_descriptor_set_update_after_bind_acceleration_structures{};
};

// Kept as a structural adapter so this contract remains usable in pure unit
// tests without pulling Vulkan headers into the test translation unit.
template<typename T, typename U, typename A>
[[nodiscard]] constexpr DescriptorInterfaceLimits
descriptor_interface_limits_from(
    const T &limits, const U &update_after_bind_limits,
    const A &acceleration_structure_limits) noexcept {
    return DescriptorInterfaceLimits{
        .max_bound_descriptor_sets = limits.maxBoundDescriptorSets,
        .max_per_stage_descriptor_samplers =
            limits.maxPerStageDescriptorSamplers,
        .max_per_stage_descriptor_uniform_buffers =
            limits.maxPerStageDescriptorUniformBuffers,
        .max_per_stage_descriptor_storage_buffers =
            limits.maxPerStageDescriptorStorageBuffers,
        .max_per_stage_descriptor_sampled_images =
            limits.maxPerStageDescriptorSampledImages,
        .max_per_stage_descriptor_storage_images =
            limits.maxPerStageDescriptorStorageImages,
        .max_per_stage_resources = limits.maxPerStageResources,
        .max_descriptor_set_samplers = limits.maxDescriptorSetSamplers,
        .max_descriptor_set_uniform_buffers =
            limits.maxDescriptorSetUniformBuffers,
        .max_descriptor_set_storage_buffers =
            limits.maxDescriptorSetStorageBuffers,
        .max_descriptor_set_sampled_images =
            limits.maxDescriptorSetSampledImages,
        .max_descriptor_set_storage_images =
            limits.maxDescriptorSetStorageImages,
        .max_per_stage_descriptor_acceleration_structures =
            acceleration_structure_limits.maxPerStageDescriptorAccelerationStructures,
        .max_descriptor_set_acceleration_structures =
            acceleration_structure_limits.maxDescriptorSetAccelerationStructures,
        .max_per_stage_descriptor_update_after_bind_samplers =
            update_after_bind_limits.maxPerStageDescriptorUpdateAfterBindSamplers,
        .max_per_stage_descriptor_update_after_bind_storage_buffers =
            update_after_bind_limits.maxPerStageDescriptorUpdateAfterBindStorageBuffers,
        .max_per_stage_descriptor_update_after_bind_sampled_images =
            update_after_bind_limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
        .max_per_stage_update_after_bind_resources =
            update_after_bind_limits.maxPerStageUpdateAfterBindResources,
        .max_descriptor_set_update_after_bind_samplers =
            update_after_bind_limits.maxDescriptorSetUpdateAfterBindSamplers,
        .max_descriptor_set_update_after_bind_storage_buffers =
            update_after_bind_limits.maxDescriptorSetUpdateAfterBindStorageBuffers,
        .max_descriptor_set_update_after_bind_sampled_images =
            update_after_bind_limits.maxDescriptorSetUpdateAfterBindSampledImages,
        .max_per_stage_descriptor_update_after_bind_acceleration_structures =
            acceleration_structure_limits.maxPerStageDescriptorUpdateAfterBindAccelerationStructures,
        .max_descriptor_set_update_after_bind_acceleration_structures =
            acceleration_structure_limits.maxDescriptorSetUpdateAfterBindAccelerationStructures};
}

struct DescriptorInterfaceRequest {
    luisa::span<const hlsl::Property> properties{};
    DescriptorInterfaceStageMask stage_mask{};
    uint32_t bindless_heap_capacity{};
    bool use_buffer_bindless{};
    bool use_tex2d_bindless{};
    bool use_tex3d_bindless{};
    bool has_constant_ubo_payload{};
    bool acceleration_structure_available{};
    bool sampled_image_update_after_bind_enabled{};
    bool storage_buffer_update_after_bind_enabled{};
};

struct DescriptorInterfacePlan {
    DescriptorInterfaceError error{DescriptorInterfaceError::NONE};
    DescriptorInterfaceStageMask stage_mask{};
    std::array<DescriptorInterfaceCounts,
               descriptor_interface_max_set_count>
        set_counts{};
    uint32_t descriptor_set_count{};
    uint32_t local_binding_count{};
    uint32_t update_after_bind_set_count{};
    uint32_t indirect_dispatch_binding_count{};
    uint32_t acceleration_structure_binding_count{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return error == DescriptorInterfaceError::NONE;
    }
};

// ConstantValue describes push-constant bytes and never owns a Vulkan
// descriptor. Keep local descriptor lookup on the same property
// interpretation as layout planning, independent of persisted table order.
[[nodiscard]] inline const hlsl::Property *find_local_descriptor_property(
    luisa::span<const hlsl::Property> properties,
    uint32_t binding) noexcept {
    for (auto i = properties.size(); i != 0u; --i) {
        auto &property = properties[i - 1u];
        if (property.type != hlsl::ShaderVariableType::ConstantValue &&
            property.space_index == 0u &&
            property.register_index == binding) {
            return &property;
        }
    }
    return nullptr;
}

namespace descriptor_interface_detail {

enum class GlobalHeapRole : uint8_t { NONE,
                                      BUFFER,
                                      TEXTURE_2D,
                                      TEXTURE_3D };

[[nodiscard]] constexpr bool valid_stage_mask(
    DescriptorInterfaceStageMask stage_mask) noexcept {
    return stage_mask == DescriptorInterfaceStageMask::COMPUTE ||
           stage_mask == DescriptorInterfaceStageMask::RASTER ||
           stage_mask == DescriptorInterfaceStageMask::RAY_TRACING;
}

[[nodiscard]] constexpr GlobalHeapRole global_heap_role(
    const DescriptorInterfaceRequest &request,
    uint32_t set) noexcept {
    auto next_set = 2u;
    if (request.use_buffer_bindless) {
        if (set == next_set) { return GlobalHeapRole::BUFFER; }
        ++next_set;
    }
    if (request.use_tex2d_bindless) {
        if (set == next_set) { return GlobalHeapRole::TEXTURE_2D; }
        ++next_set;
    }
    if (request.use_tex3d_bindless) {
        if (set == next_set) { return GlobalHeapRole::TEXTURE_3D; }
    }
    return GlobalHeapRole::NONE;
}

[[nodiscard]] constexpr DescriptorInterfaceCounts descriptor_counts(
    hlsl::ShaderVariableType type, uint32_t count) noexcept {
    auto result = DescriptorInterfaceCounts{};
    switch (type) {
        case hlsl::ShaderVariableType::ConstantBuffer:
        case hlsl::ShaderVariableType::CBVBufferHeap:
            result.uniform_buffers = count;
            break;
        case hlsl::ShaderVariableType::SRVTextureHeap:
            result.sampled_images = count;
            break;
        case hlsl::ShaderVariableType::UAVTextureHeap:
            result.storage_images = count;
            break;
        case hlsl::ShaderVariableType::StructuredBuffer:
        case hlsl::ShaderVariableType::RWStructuredBuffer:
        case hlsl::ShaderVariableType::SRVBufferHeap:
        case hlsl::ShaderVariableType::UAVBufferHeap:
        case hlsl::ShaderVariableType::SPIRVAccelInstance:
        case hlsl::ShaderVariableType::SPIRVAccelInstanceRW:
        case hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata:
        case hlsl::ShaderVariableType::SPIRVIndirectDispatch:
            result.storage_buffers = count;
            break;
        case hlsl::ShaderVariableType::SamplerHeap:
            result.samplers = count;
            break;
        case hlsl::ShaderVariableType::SPIRVAccel:
            result.acceleration_structures = count;
            break;
        case hlsl::ShaderVariableType::ConstantValue: break;
    }
    return result;
}

[[nodiscard]] constexpr DescriptorInterfaceError check_per_set_limits(
    DescriptorInterfaceCounts counts,
    const DescriptorInterfaceLimits &limits) noexcept {
    if (counts.samplers > limits.max_descriptor_set_samplers) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_SAMPLERS;
    }
    if (counts.uniform_buffers >
        limits.max_descriptor_set_uniform_buffers) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UNIFORM_BUFFERS;
    }
    if (counts.storage_buffers >
        limits.max_descriptor_set_storage_buffers) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_STORAGE_BUFFERS;
    }
    if (counts.sampled_images >
        limits.max_descriptor_set_sampled_images) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_SAMPLED_IMAGES;
    }
    if (counts.storage_images >
        limits.max_descriptor_set_storage_images) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_STORAGE_IMAGES;
    }
    if (counts.acceleration_structures >
        limits.max_descriptor_set_acceleration_structures) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_ACCELERATION_STRUCTURES;
    }
    return DescriptorInterfaceError::NONE;
}

[[nodiscard]] constexpr DescriptorInterfaceError check_per_stage_limits(
    DescriptorInterfaceCounts counts,
    const DescriptorInterfaceLimits &limits) noexcept {
    if (counts.samplers > limits.max_per_stage_descriptor_samplers) {
        return DescriptorInterfaceError::MAX_PER_STAGE_SAMPLERS;
    }
    if (counts.uniform_buffers >
        limits.max_per_stage_descriptor_uniform_buffers) {
        return DescriptorInterfaceError::MAX_PER_STAGE_UNIFORM_BUFFERS;
    }
    if (counts.storage_buffers >
        limits.max_per_stage_descriptor_storage_buffers) {
        return DescriptorInterfaceError::MAX_PER_STAGE_STORAGE_BUFFERS;
    }
    if (counts.sampled_images >
        limits.max_per_stage_descriptor_sampled_images) {
        return DescriptorInterfaceError::MAX_PER_STAGE_SAMPLED_IMAGES;
    }
    if (counts.storage_images >
        limits.max_per_stage_descriptor_storage_images) {
        return DescriptorInterfaceError::MAX_PER_STAGE_STORAGE_IMAGES;
    }
    if (counts.acceleration_structures >
        limits.max_per_stage_descriptor_acceleration_structures) {
        return DescriptorInterfaceError::MAX_PER_STAGE_ACCELERATION_STRUCTURES;
    }
    if (counts.resources() > limits.max_per_stage_resources) {
        return DescriptorInterfaceError::MAX_PER_STAGE_RESOURCES;
    }
    return DescriptorInterfaceError::NONE;
}

[[nodiscard]] constexpr DescriptorInterfaceError
check_update_after_bind_limits(
    DescriptorInterfaceCounts counts,
    const DescriptorInterfaceLimits &limits,
    bool sampled_image_update_after_bind_enabled,
    bool storage_buffer_update_after_bind_enabled) noexcept {
    if (sampled_image_update_after_bind_enabled) {
        if (counts.samplers >
            limits.max_per_stage_descriptor_update_after_bind_samplers) {
            return DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLERS;
        }
        if (counts.sampled_images >
            limits.max_per_stage_descriptor_update_after_bind_sampled_images) {
            return DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_SAMPLED_IMAGES;
        }
        if (counts.samplers >
            limits.max_descriptor_set_update_after_bind_samplers) {
            return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLERS;
        }
        if (counts.sampled_images >
            limits.max_descriptor_set_update_after_bind_sampled_images) {
            return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_SAMPLED_IMAGES;
        }
    }
    if (storage_buffer_update_after_bind_enabled) {
        if (counts.storage_buffers >
            limits.max_per_stage_descriptor_update_after_bind_storage_buffers) {
            return DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_STORAGE_BUFFERS;
        }
        if (counts.storage_buffers >
            limits.max_descriptor_set_update_after_bind_storage_buffers) {
            return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_STORAGE_BUFFERS;
        }
    }
    if ((sampled_image_update_after_bind_enabled ||
         storage_buffer_update_after_bind_enabled) &&
        counts.resources() >
            limits.max_per_stage_update_after_bind_resources) {
        return DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_RESOURCES;
    }
    // The acceleration-structure update-after-bind limits are unconditional
    // pipeline-layout limits, even though this backend does not put AS
    // bindings in update-after-bind layouts.
    if (counts.acceleration_structures >
        limits.max_per_stage_descriptor_update_after_bind_acceleration_structures) {
        return DescriptorInterfaceError::MAX_PER_STAGE_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES;
    }
    if (counts.acceleration_structures >
        limits.max_descriptor_set_update_after_bind_acceleration_structures) {
        return DescriptorInterfaceError::MAX_DESCRIPTOR_SET_UPDATE_AFTER_BIND_ACCELERATION_STRUCTURES;
    }
    return DescriptorInterfaceError::NONE;
}

}// namespace descriptor_interface_detail

[[nodiscard]] constexpr DescriptorInterfacePlan plan_descriptor_interface(
    const DescriptorInterfaceRequest &request,
    const DescriptorInterfaceLimits &limits) noexcept {
    using namespace descriptor_interface_detail;
    auto plan = DescriptorInterfacePlan{.stage_mask = request.stage_mask};
    auto fail = [&](DescriptorInterfaceError error) noexcept {
        plan.error = error;
        return plan;
    };
    if (!valid_stage_mask(request.stage_mask)) {
        return fail(DescriptorInterfaceError::INVALID_STAGE_MASK);
    }
    if (request.properties.size() >
        descriptor_interface_max_property_count) {
        return fail(DescriptorInterfaceError::TOO_MANY_PROPERTIES);
    }

    auto enabled_global_heap_count =
        static_cast<uint32_t>(request.use_buffer_bindless) +
        static_cast<uint32_t>(request.use_tex2d_bindless) +
        static_cast<uint32_t>(request.use_tex3d_bindless);
    auto descriptor_set_count_wide = uint64_t{2u} +
                                     enabled_global_heap_count;
    if (descriptor_set_count_wide > limits.max_bound_descriptor_sets ||
        descriptor_set_count_wide > descriptor_interface_max_set_count) {
        return fail(DescriptorInterfaceError::TOO_MANY_DESCRIPTOR_SETS);
    }
    plan.descriptor_set_count =
        static_cast<uint32_t>(descriptor_set_count_wide);
    plan.update_after_bind_set_count = enabled_global_heap_count;
    if (enabled_global_heap_count != 0u &&
        request.bindless_heap_capacity == 0u) {
        return fail(DescriptorInterfaceError::BINDLESS_UNAVAILABLE);
    }
    if ((request.use_buffer_bindless &&
         !request.storage_buffer_update_after_bind_enabled) ||
        ((request.use_tex2d_bindless || request.use_tex3d_bindless) &&
         !request.sampled_image_update_after_bind_enabled)) {
        return fail(DescriptorInterfaceError::BINDLESS_UNAVAILABLE);
    }

    std::array<uint8_t, descriptor_interface_max_local_bindings>
        local_bindings{};
    std::array<uint8_t, 3u> global_heaps{};
    auto sampler_count = 0u;
    auto push_constant_count = 0u;
    auto constant_ubo_count = 0u;
    auto highest_local_binding = 0u;
    auto has_local_binding = false;

    for (auto property : request.properties) {
        if (luisa::to_underlying(property.type) >
            luisa::to_underlying(
                hlsl::ShaderVariableType::SPIRVIndirectDispatch)) {
            return fail(DescriptorInterfaceError::INVALID_PROPERTY_TYPE);
        }
        if (property.type == hlsl::ShaderVariableType::ConstantValue) {
            if (property.space_index != 0u ||
                property.register_index != 0u || property.array_size != 1u) {
                return fail(
                    DescriptorInterfaceError::NONCANONICAL_PUSH_CONSTANT);
            }
            if (++push_constant_count > 1u) {
                return fail(
                    DescriptorInterfaceError::DUPLICATE_PUSH_CONSTANT);
            }
            continue;
        }
        if (property.type == hlsl::ShaderVariableType::SamplerHeap) {
            if (property.space_index != 1u ||
                property.register_index != 0u ||
                property.array_size != descriptor_interface_sampler_count) {
                return fail(DescriptorInterfaceError::NONCANONICAL_SAMPLER);
            }
            if (++sampler_count > 1u) {
                return fail(
                    DescriptorInterfaceError::MISSING_OR_DUPLICATE_SAMPLER);
            }
            plan.set_counts[1u] += descriptor_counts(
                property.type, property.array_size);
            continue;
        }

        auto unbounded = property.array_size ==
                         std::numeric_limits<uint32_t>::max();
        if (!unbounded && property.array_size != 1u) {
            return fail(DescriptorInterfaceError::INVALID_ARRAY_SIZE);
        }
        if (unbounded) {
            if (property.register_index != 0u) {
                return fail(
                    DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP);
            }
            auto role = global_heap_role(request, property.space_index);
            auto role_index = uint32_t{};
            switch (role) {
                case GlobalHeapRole::BUFFER:
                    if (property.type !=
                        hlsl::ShaderVariableType::SRVBufferHeap) {
                        return fail(
                            DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP);
                    }
                    role_index = 0u;
                    break;
                case GlobalHeapRole::TEXTURE_2D:
                    if (property.type !=
                        hlsl::ShaderVariableType::SRVTextureHeap) {
                        return fail(
                            DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP);
                    }
                    role_index = 1u;
                    break;
                case GlobalHeapRole::TEXTURE_3D:
                    if (property.type !=
                        hlsl::ShaderVariableType::SRVTextureHeap) {
                        return fail(
                            DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP);
                    }
                    role_index = 2u;
                    break;
                case GlobalHeapRole::NONE:
                    return fail(
                        DescriptorInterfaceError::NONCANONICAL_GLOBAL_HEAP);
            }
            if (++global_heaps[role_index] > 1u) {
                return fail(
                    DescriptorInterfaceError::MISSING_OR_DUPLICATE_GLOBAL_HEAP);
            }
            plan.set_counts[property.space_index] += descriptor_counts(
                property.type, request.bindless_heap_capacity);
            continue;
        }

        if (property.space_index != 0u ||
            property.register_index >=
                descriptor_interface_max_local_bindings) {
            return fail(
                DescriptorInterfaceError::NONCANONICAL_LOCAL_BINDING);
        }
        if (local_bindings[property.register_index] != 0u) {
            return fail(DescriptorInterfaceError::DUPLICATE_LOCAL_BINDING);
        }
        local_bindings[property.register_index] = 1u;
        highest_local_binding =
            has_local_binding ?
                (property.register_index > highest_local_binding ?
                     property.register_index :
                     highest_local_binding) :
                property.register_index;
        has_local_binding = true;

        if (property.type == hlsl::ShaderVariableType::ConstantBuffer) {
            ++constant_ubo_count;
        }
        if (property.type ==
            hlsl::ShaderVariableType::SPIRVIndirectDispatch) {
            if (request.stage_mask !=
                DescriptorInterfaceStageMask::COMPUTE) {
                return fail(
                    DescriptorInterfaceError::INDIRECT_DISPATCH_NOT_SUPPORTED);
            }
            if (++plan.indirect_dispatch_binding_count > 1u) {
                return fail(
                    DescriptorInterfaceError::DUPLICATE_INDIRECT_DISPATCH);
            }
        }
        if (property.type == hlsl::ShaderVariableType::SPIRVAccel) {
            if (!request.acceleration_structure_available) {
                return fail(
                    DescriptorInterfaceError::ACCELERATION_STRUCTURE_NOT_SUPPORTED);
            }
            ++plan.acceleration_structure_binding_count;
        }
        plan.set_counts[0u] += descriptor_counts(property.type, 1u);
    }

    if (sampler_count != 1u) {
        return fail(
            DescriptorInterfaceError::MISSING_OR_DUPLICATE_SAMPLER);
    }
    if ((request.has_constant_ubo_payload ? 1u : 0u) !=
        constant_ubo_count) {
        return fail(DescriptorInterfaceError::CONSTANT_UBO_MISMATCH);
    }
    if ((request.use_buffer_bindless ? 1u : 0u) != global_heaps[0u] ||
        (request.use_tex2d_bindless ? 1u : 0u) != global_heaps[1u] ||
        (request.use_tex3d_bindless ? 1u : 0u) != global_heaps[2u]) {
        return fail(
            DescriptorInterfaceError::MISSING_OR_DUPLICATE_GLOBAL_HEAP);
    }

    if (has_local_binding) {
        auto local_binding_count_wide = uint64_t{highest_local_binding} + 1u;
        if (local_binding_count_wide >
            descriptor_interface_max_local_bindings) {
            return fail(
                DescriptorInterfaceError::NONCANONICAL_LOCAL_BINDING);
        }
        plan.local_binding_count =
            static_cast<uint32_t>(local_binding_count_wide);
        for (auto i = 0u; i < plan.local_binding_count; ++i) {
            if (local_bindings[i] == 0u) {
                return fail(
                    DescriptorInterfaceError::NONDENSE_LOCAL_BINDINGS);
            }
        }
    }

    // Sets 2+ carry UPDATE_AFTER_BIND_POOL. Core ordinary limits count only
    // sets 0 and 1, while the enabled update-after-bind limits below count all
    // layouts in the pipeline interface.
    for (auto set = 0u; set < 2u; ++set) {
        if (auto error = check_per_set_limits(plan.set_counts[set], limits);
            error != DescriptorInterfaceError::NONE) {
            return fail(error);
        }
    }
    auto per_stage_counts = plan.set_counts[0u];
    per_stage_counts += plan.set_counts[1u];
    if (auto error = check_per_stage_limits(per_stage_counts, limits);
        error != DescriptorInterfaceError::NONE) {
        return fail(error);
    }
    auto update_after_bind_counts = per_stage_counts;
    for (auto set = 2u; set < plan.descriptor_set_count; ++set) {
        update_after_bind_counts += plan.set_counts[set];
    }
    if (auto error = check_update_after_bind_limits(
            update_after_bind_counts, limits,
            request.sampled_image_update_after_bind_enabled,
            request.storage_buffer_update_after_bind_enabled);
        error != DescriptorInterfaceError::NONE) {
        return fail(error);
    }
    return plan;
}

// Writer-side structural validation has no VkDevice, but persisted descriptor
// tables must still be canonical before they are hashed and written. Use
// unconstraining numeric limits while retaining every ABI-shape check above.
[[nodiscard]] constexpr DescriptorInterfacePlan
plan_persisted_descriptor_interface(
    luisa::span<const hlsl::Property> properties,
    DescriptorInterfaceStageMask stage_mask,
    bool use_buffer_bindless, bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool has_constant_ubo_payload) noexcept {
    constexpr auto maximum = std::numeric_limits<uint32_t>::max();
    constexpr auto structural_limits = DescriptorInterfaceLimits{
        .max_bound_descriptor_sets = descriptor_interface_max_set_count,
        .max_per_stage_descriptor_samplers = maximum,
        .max_per_stage_descriptor_uniform_buffers = maximum,
        .max_per_stage_descriptor_storage_buffers = maximum,
        .max_per_stage_descriptor_sampled_images = maximum,
        .max_per_stage_descriptor_storage_images = maximum,
        .max_per_stage_resources = maximum,
        .max_descriptor_set_samplers = maximum,
        .max_descriptor_set_uniform_buffers = maximum,
        .max_descriptor_set_storage_buffers = maximum,
        .max_descriptor_set_sampled_images = maximum,
        .max_descriptor_set_storage_images = maximum,
        .max_per_stage_descriptor_acceleration_structures = maximum,
        .max_descriptor_set_acceleration_structures = maximum,
        .max_per_stage_descriptor_update_after_bind_samplers = maximum,
        .max_per_stage_descriptor_update_after_bind_storage_buffers = maximum,
        .max_per_stage_descriptor_update_after_bind_sampled_images = maximum,
        .max_per_stage_update_after_bind_resources = maximum,
        .max_descriptor_set_update_after_bind_samplers = maximum,
        .max_descriptor_set_update_after_bind_storage_buffers = maximum,
        .max_descriptor_set_update_after_bind_sampled_images = maximum,
        .max_per_stage_descriptor_update_after_bind_acceleration_structures =
            maximum,
        .max_descriptor_set_update_after_bind_acceleration_structures =
            maximum};
    auto uses_bindless = use_buffer_bindless || use_tex2d_bindless ||
                         use_tex3d_bindless;
    return plan_descriptor_interface(
        {.properties = properties,
         .stage_mask = stage_mask,
         .bindless_heap_capacity = uses_bindless ? 1u : 0u,
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .has_constant_ubo_payload = has_constant_ubo_payload,
         .acceleration_structure_available = true,
         .sampled_image_update_after_bind_enabled = uses_bindless,
         .storage_buffer_update_after_bind_enabled = uses_bindless},
        structural_limits);
}

[[nodiscard]] constexpr bool valid_push_constant_write(
    uint32_t offset, uint32_t size,
    uint32_t push_constant_range_size) noexcept {
    return size != 0u && offset % sizeof(uint32_t) == 0u &&
           size % sizeof(uint32_t) == 0u &&
           offset <= push_constant_range_size &&
           size <= push_constant_range_size - offset;
}

}// namespace lc::vk::detail
