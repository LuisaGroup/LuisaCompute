#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "../common/spirv/spirv_codegen/target_feature_mask.h"

namespace lc::vk {
class Device;
}// namespace lc::vk

namespace lc::vk::detail {

// Optional *shader capability* device extensions.
//
// Every entry round-trips through the same three layers:
//
//   1. negotiation - `OptionalExtension::SHADER_*` in vk_extension_registry.h
//      probes the extension and its VkPhysicalDevice*Features* bits, and
//      `VulkanDeviceConfigExt::enable_shader_*()` decides whether the backend
//      asks for it;
//   2. declaration - `shader_spirv_capability_declarations` names the exact
//      SPIR-V capability/extension an emitter has to declare before it may use
//      the instruction group, and `shader_capability_declarations` names the
//      persistent target-feature bit (spirv_codegen/target_feature_mask.h) the
//      compiled artifact then requires;
//   3. planning - `plan_shader_capabilities()` turns the negotiated extension
//      state into the set of capabilities a shader may use.
//
// None of this is a correctness requirement. A device that does not support an
// extension, a device where the switch is off, and an imported logical device
// (whose enabled extension/feature chain Vulkan cannot report) all plan zero
// capabilities, and the backend keeps its existing code path. A capability may
// never be reported merely because the physical device supports it.
enum class ShaderCapability : uint32_t {
    // VK_KHR_shader_fma: correctly-rounded OpFmaKHR.
    FMA,
    // VK_EXT_shader_64bit_indexing: 64-bit addressing calculations.
    INDEXING_64BIT,
    // VK_EXT_shader_uniform_buffer_unsized_array: OpTypeRuntimeArray as the last
    // member of a uniform block.
    UNIFORM_BUFFER_UNSIZED_ARRAY,
    // VK_EXT_shader_long_vector: vectors with more than four components.
    LONG_VECTOR,
    // VK_EXT_shader_subgroup_partitioned: OpGroupNonUniformPartitionEXT.
    SUBGROUP_PARTITIONED,
    // VK_EXT_shader_split_barrier: OpControlBarrierArriveEXT/WaitEXT.
    SPLIT_BARRIER,
    // VK_EXT_shader_ocp_microscaling_types: the OCP microscaling scalar types
    // (float4/float6/float8 unsigned E8M0/MX int8).
    OCP_MICROSCALING_TYPES,
    // VK_KHR_shader_constant_data: OpConstantDataKHR in the ConstantDataKHR
    // storage class.
    CONSTANT_DATA,
    // VK_KHR_shader_abort: OpAbortKHR.
    ABORT,
    // VK_VALVE_shader_mixed_float_dot_product: OpFDot2MixAcc32VALVE
    // (FP16 inputs, FP32 accumulator).
    MIXED_FLOAT_DOT_PRODUCT,
    COUNT
};

inline constexpr size_t shader_capability_count =
    static_cast<size_t>(ShaderCapability::COUNT);

// A set of ShaderCapability values, one bit per enumerator.
using ShaderCapabilityMask = uint32_t;

[[nodiscard]] constexpr ShaderCapabilityMask shader_capability_bit(
    ShaderCapability capability) noexcept {
    return ShaderCapabilityMask{1u} << static_cast<uint32_t>(capability);
}

// Vulkan extension name and persistent target-feature bit of one capability.
struct ShaderCapabilityDeclaration {
    ShaderCapability capability{ShaderCapability::COUNT};
    const char *vulkan_extension{nullptr};
    spirv::SpirvTargetFeatureMask target_feature{0u};
};

inline constexpr std::array shader_capability_declarations{
    ShaderCapabilityDeclaration{ShaderCapability::FMA,
                                "VK_KHR_shader_fma",
                                spirv::target_feature::shader_fma_float32},
    ShaderCapabilityDeclaration{ShaderCapability::INDEXING_64BIT,
                                "VK_EXT_shader_64bit_indexing",
                                spirv::target_feature::shader_64bit_indexing},
    ShaderCapabilityDeclaration{ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY,
                                "VK_EXT_shader_uniform_buffer_unsized_array",
                                spirv::target_feature::shader_uniform_buffer_unsized_array},
    ShaderCapabilityDeclaration{ShaderCapability::LONG_VECTOR,
                                "VK_EXT_shader_long_vector",
                                spirv::target_feature::shader_long_vector},
    ShaderCapabilityDeclaration{ShaderCapability::SUBGROUP_PARTITIONED,
                                "VK_EXT_shader_subgroup_partitioned",
                                spirv::target_feature::shader_subgroup_partitioned},
    ShaderCapabilityDeclaration{ShaderCapability::SPLIT_BARRIER,
                                "VK_EXT_shader_split_barrier",
                                spirv::target_feature::shader_split_barrier},
    ShaderCapabilityDeclaration{ShaderCapability::OCP_MICROSCALING_TYPES,
                                "VK_EXT_shader_ocp_microscaling_types",
                                spirv::target_feature::shader_ocp_microscaling_types},
    ShaderCapabilityDeclaration{ShaderCapability::CONSTANT_DATA,
                                "VK_KHR_shader_constant_data",
                                spirv::target_feature::shader_constant_data},
    ShaderCapabilityDeclaration{ShaderCapability::ABORT,
                                "VK_KHR_shader_abort",
                                spirv::target_feature::shader_abort},
    ShaderCapabilityDeclaration{ShaderCapability::MIXED_FLOAT_DOT_PRODUCT,
                                "VK_VALVE_shader_mixed_float_dot_product",
                                spirv::target_feature::shader_mixed_float_dot_product_float16_acc_float32}};

// One OpCapability an enabled capability may need.
//
// `spirv_extension` is the SPIR-V extension string to add to
// OpExtension/OpModuleProcessed, `spirv_capability` is the spv::Capability
// enumerant value, and `spirv_capability_name` is its diagnostic name. A row
// with a null extension and capability 0 documents an extension that reuses
// existing SPIR-V capabilities and therefore needs no new declaration.
struct ShaderSpirvCapabilityDeclaration {
    ShaderCapability capability{ShaderCapability::COUNT};
    const char *spirv_extension{nullptr};
    uint32_t spirv_capability{0u};
    const char *spirv_capability_name{nullptr};
};

inline constexpr std::array shader_spirv_capability_declarations{
    // SPV_KHR_fma: OpFmaKHR. The backend negotiates shaderFmaFloat32 only.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::FMA,
                                     "SPV_KHR_fma", 6030u, "FMAKHR"},
    // SPV_EXT_shader_64bit_indexing: OpCapability and
    // OpExecutionMode Shader64BitIndexingEXT, plus a pipeline/shader creation
    // flag in Vulkan.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::INDEXING_64BIT,
                                     "SPV_EXT_shader_64bit_indexing", 5426u,
                                     "Shader64BitIndexingEXT"},
    // VK_EXT_shader_uniform_buffer_unsized_array reuses OpTypeRuntimeArray with
    // an existing capability, so there is no new SPIR-V declaration. Vulkan
    // only requires the shaderUniformBufferUnsizedArray feature and forbids
    // OpArrayLength on such a member.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY,
                                     nullptr, 0u, nullptr},
    // SPV_EXT_long_vector: vectors with more than four components, limited by
    // VkPhysicalDeviceShaderLongVectorPropertiesEXT::maxVectorComponents.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::LONG_VECTOR,
                                     "SPV_EXT_long_vector", 5425u,
                                     "LongVectorEXT"},
    // SPV_EXT_shader_subgroup_partitioned: OpGroupNonUniformPartitionEXT and
    // the PartitionedReduce/InclusiveScan/ExclusiveScan group operations.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::SUBGROUP_PARTITIONED,
                                     "SPV_EXT_shader_subgroup_partitioned", 5297u,
                                     "GroupNonUniformPartitionedEXT"},
    // SPV_EXT_split_barrier: OpControlBarrierArriveEXT/WaitEXT.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::SPLIT_BARRIER,
                                     "SPV_EXT_split_barrier", 6141u,
                                     "SplitBarrierEXT"},
    // SPV_EXT_ocp_microscaling_types: one capability per scalar type plus the
    // bitcast/extract form. The negotiated feature struct enables whichever
    // types the physical device reports, so a shader declares only the ones it
    // actually uses.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::OCP_MICROSCALING_TYPES,
                                     "SPV_EXT_ocp_microscaling_types", 4229u,
                                     "Float4EXT"},
    ShaderSpirvCapabilityDeclaration{ShaderCapability::OCP_MICROSCALING_TYPES,
                                     "SPV_EXT_ocp_microscaling_types", 4228u,
                                     "Float6EXT"},
    ShaderSpirvCapabilityDeclaration{ShaderCapability::OCP_MICROSCALING_TYPES,
                                     "SPV_EXT_ocp_microscaling_types", 4230u,
                                     "Float8UnsignedE8M0EXT"},
    ShaderSpirvCapabilityDeclaration{ShaderCapability::OCP_MICROSCALING_TYPES,
                                     "SPV_EXT_ocp_microscaling_types", 4231u,
                                     "MXInt8EXT"},
    ShaderSpirvCapabilityDeclaration{ShaderCapability::OCP_MICROSCALING_TYPES,
                                     "SPV_EXT_ocp_microscaling_types", 4232u,
                                     "BitcastExtractEXT"},
    // SPV_KHR_constant_data: OpConstantDataKHR/OpSpecConstantDataKHR and the
    // ConstantDataKHR storage class.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::CONSTANT_DATA,
                                     "SPV_KHR_constant_data", 5146u,
                                     "ConstantDataKHR"},
    // SPV_KHR_abort: OpAbortKHR.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::ABORT,
                                     "SPV_KHR_abort", 5120u, "AbortKHR"},
    // SPV_VALVE_mixed_float_dot_product: OpFDot2MixAcc32VALVE. The backend
    // negotiates shaderMixedFloatDotProductFloat16AccFloat32 only.
    ShaderSpirvCapabilityDeclaration{ShaderCapability::MIXED_FLOAT_DOT_PRODUCT,
                                     "SPV_VALVE_mixed_float_dot_product", 6912u,
                                     "DotProductFloat16AccFloat32VALVE"}};

// Declaration of exactly one capability. `ShaderCapability::COUNT` is not a
// capability; it falls back to the first row instead of inventing an entry.
[[nodiscard]] constexpr ShaderCapabilityDeclaration const &
shader_capability_declaration(ShaderCapability capability) noexcept {
    for (auto const &declaration : shader_capability_declarations) {
        if (declaration.capability == capability) { return declaration; }
    }
    return shader_capability_declarations.front();
}

// Host-testable snapshot of the negotiated optional device extensions that
// decide the shader capability set. `true` means the extension is *enabled* on
// the device that compiles the shader: supported by the physical device, asked
// for by VulkanDeviceConfigExt, and its required feature bits negotiated. There
// is deliberately no "supported" field - a supported-but-not-enabled extension
// must behave exactly like an unsupported one.
struct ShaderExtensionAvailability {
    bool fma{false};
    bool indexing_64bit{false};
    bool uniform_buffer_unsized_array{false};
    bool long_vector{false};
    bool subgroup_partitioned{false};
    bool split_barrier{false};
    bool ocp_microscaling_types{false};
    bool constant_data{false};
    bool abort{false};
    bool mixed_float_dot_product{false};

    [[nodiscard]] constexpr bool enabled(
        ShaderCapability capability) const noexcept {
        switch (capability) {
            case ShaderCapability::FMA: return fma;
            case ShaderCapability::INDEXING_64BIT: return indexing_64bit;
            case ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY:
                return uniform_buffer_unsized_array;
            case ShaderCapability::LONG_VECTOR: return long_vector;
            case ShaderCapability::SUBGROUP_PARTITIONED:
                return subgroup_partitioned;
            case ShaderCapability::SPLIT_BARRIER: return split_barrier;
            case ShaderCapability::OCP_MICROSCALING_TYPES:
                return ocp_microscaling_types;
            case ShaderCapability::CONSTANT_DATA: return constant_data;
            case ShaderCapability::ABORT: return abort;
            case ShaderCapability::MIXED_FLOAT_DOT_PRODUCT:
                return mixed_float_dot_product;
            case ShaderCapability::COUNT: break;
        }
        return false;
    }
};

// What a shader compiled for this device may use.
//
// `available_target_features` is the union of the persistent target-feature bits
// of every available capability. It is a *capability* set, not a requirement:
// an artifact only records the bits of the capabilities it really emitted.
struct ShaderCapabilityPlan {
    ShaderCapabilityMask available{0u};
    spirv::SpirvTargetFeatureMask available_target_features{0u};

    [[nodiscard]] constexpr bool is_available(
        ShaderCapability capability) const noexcept {
        return (available & shader_capability_bit(capability)) != 0u;
    }
    [[nodiscard]] constexpr bool any() const noexcept { return available != 0u; }
    [[nodiscard]] constexpr size_t count() const noexcept {
        auto bits = available;
        auto result = size_t{0u};
        while (bits != 0u) {
            bits &= bits - 1u;
            result++;
        }
        return result;
    }
};

[[nodiscard]] constexpr ShaderCapabilityPlan plan_shader_capabilities(
    ShaderExtensionAvailability availability) noexcept {
    ShaderCapabilityPlan plan;
    for (auto const &declaration : shader_capability_declarations) {
        if (availability.enabled(declaration.capability)) {
            plan.available |= shader_capability_bit(declaration.capability);
            plan.available_target_features |= declaration.target_feature;
        }
    }
    return plan;
}

// ---- declaration path ----------------------------------------------------

// Target-feature bits of the capabilities named by `used`, independent of any
// device. Unknown bits are ignored.
[[nodiscard]] constexpr spirv::SpirvTargetFeatureMask
shader_capability_target_features(ShaderCapabilityMask used) noexcept {
    spirv::SpirvTargetFeatureMask features{0u};
    for (auto const &declaration : shader_capability_declarations) {
        if ((used & shader_capability_bit(declaration.capability)) != 0u) {
            features |= declaration.target_feature;
        }
    }
    return features;
}

// Capabilities an emitter asked for that this device cannot provide. A non-zero
// result is a hard error: the instruction would be emitted without its
// capability, so the shader must be rejected instead.
[[nodiscard]] constexpr ShaderCapabilityMask missing_shader_capabilities(
    ShaderCapabilityPlan plan, ShaderCapabilityMask used) noexcept {
    return used & ~plan.available;
}

// Exact target-feature requirement to record on an artifact that used
// `used`, after `missing_shader_capabilities()` proved every one of them is
// available.
[[nodiscard]] constexpr spirv::SpirvTargetFeatureMask
required_shader_capability_features(
    ShaderCapabilityPlan plan, ShaderCapabilityMask used) noexcept {
    return shader_capability_target_features(used & plan.available);
}

namespace vk_shader_capability_contract_detail {

// Every capability has exactly one declaration, its persistent bit is a single
// distinct bit of the known mask, and `plan_shader_capabilities` can therefore
// build the plan by union.
[[nodiscard]] constexpr bool valid_shader_capability_declarations() noexcept {
    ShaderCapabilityMask described{0u};
    spirv::SpirvTargetFeatureMask features{0u};
    for (auto const &declaration : shader_capability_declarations) {
        if (declaration.capability == ShaderCapability::COUNT ||
            declaration.vulkan_extension == nullptr ||
            declaration.target_feature == 0u ||
            (declaration.target_feature & (declaration.target_feature - 1u)) != 0u ||
            (features & declaration.target_feature) != 0u ||
            (described & shader_capability_bit(declaration.capability)) != 0u) {
            return false;
        }
        described |= shader_capability_bit(declaration.capability);
        features |= declaration.target_feature;
    }
    auto all = (ShaderCapabilityMask{1u} << static_cast<uint32_t>(ShaderCapability::COUNT)) - 1u;
    return described == all && features != 0u;
}

// Declared bits must be part of the persistent, forward-compatible mask, so an
// artifact that requires them can be validated and cached.
[[nodiscard]] constexpr bool known_shader_capability_features() noexcept {
    return (shader_capability_target_features(~ShaderCapabilityMask{0u}) &
            ~spirv::target_feature::known_mask) == 0u;
}

// A SPIR-V declaration row either names a capability together with its
// extension, or documents an extension that needs no new OpCapability.
[[nodiscard]] constexpr bool valid_shader_spirv_capability_declarations() noexcept {
    for (auto const &declaration : shader_spirv_capability_declarations) {
        auto has_capability = declaration.spirv_capability != 0u;
        auto has_name = declaration.spirv_capability_name != nullptr;
        if (declaration.capability == ShaderCapability::COUNT ||
            has_capability != has_name ||
            (has_capability && declaration.spirv_extension == nullptr)) {
            return false;
        }
    }
    return true;
}

}// namespace vk_shader_capability_contract_detail

static_assert(vk_shader_capability_contract_detail::valid_shader_capability_declarations());
static_assert(vk_shader_capability_contract_detail::known_shader_capability_features());
static_assert(vk_shader_capability_contract_detail::valid_shader_spirv_capability_declarations());

// ---- device-facing adapter (vk_shader_capabilities.cpp) ------------------
//
// The functions below are the only place where the negotiated device state is
// read. They live in the Vulkan backend translation unit so the planner above
// stays host-testable without a device.

// Negotiated extension state of `device`. A null device plans nothing.
[[nodiscard]] ShaderExtensionAvailability shader_extension_availability(
    const Device *device) noexcept;

// What a shader compiled for `device` may use.
[[nodiscard]] ShaderCapabilityPlan device_shader_capability_plan(
    const Device *device) noexcept;

// Union of the target-feature bits of every capability `device` enables. This
// is the capability set, never a per-artifact requirement.
[[nodiscard]] spirv::SpirvTargetFeatureMask
device_shader_capability_target_features(const Device *device) noexcept;

// Whether exactly one capability is enabled on `device`.
[[nodiscard]] bool shader_capability_available(
    const Device *device, ShaderCapability capability) noexcept;

// Records that the backend really routes work through this capability
// (`Device::mark_optional_extension_active`). Returns false when the capability
// is unavailable, in which case nothing is recorded.
bool mark_shader_capability_active(Device *device,
                                   ShaderCapability capability) noexcept;

// Fail-closed declaration helper for an emission site. Returns the persistent
// target-feature bit the artifact must record after marking the extension
// active, and rejects the shader when the device cannot provide the capability.
// `LUISA_ERROR` is fatal, so a rejected capability never returns a mask.
[[nodiscard]] spirv::SpirvTargetFeatureMask require_shader_capability(
    Device *device, ShaderCapability capability,
    luisa::string_view shader_name) noexcept;

}// namespace lc::vk::detail
