#include "vk_shader_capabilities.h"

#include <array>

#include <luisa/core/logging.h>

#include "device.h"
#include "vk_extension_registry.h"

namespace lc::vk::detail {

namespace vk_ext_shader_caps_detail {

// Maps one shader capability to the optional device extension that negotiates
// it. Both sides are stable identifiers, so the mapping is a plain table.
struct ExtensionRow {
    ShaderCapability capability;
    OptionalExtension extension;
};

inline constexpr std::array extension_rows{
    ExtensionRow{ShaderCapability::FMA, OptionalExtension::SHADER_FMA},
    ExtensionRow{ShaderCapability::INDEXING_64BIT, OptionalExtension::SHADER_64BIT_INDEXING},
    ExtensionRow{ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY, OptionalExtension::SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY},
    ExtensionRow{ShaderCapability::LONG_VECTOR, OptionalExtension::SHADER_LONG_VECTOR},
    ExtensionRow{ShaderCapability::SUBGROUP_PARTITIONED, OptionalExtension::SHADER_SUBGROUP_PARTITIONED},
    ExtensionRow{ShaderCapability::SPLIT_BARRIER, OptionalExtension::SHADER_SPLIT_BARRIER},
    ExtensionRow{ShaderCapability::OCP_MICROSCALING_TYPES, OptionalExtension::SHADER_OCP_MICROSCALING_TYPES},
    ExtensionRow{ShaderCapability::CONSTANT_DATA, OptionalExtension::SHADER_CONSTANT_DATA},
    ExtensionRow{ShaderCapability::ABORT, OptionalExtension::SHADER_ABORT}};

[[nodiscard]] constexpr OptionalExtension optional_extension_of(
    ShaderCapability capability) noexcept {
    for (auto const &row : extension_rows) {
        if (row.capability == capability) { return row.extension; }
    }
    return OptionalExtension::COUNT;
}

// `OptionalExtensionRegistry::enabled` already folds in the physical-device
// support probe and the negotiated VkPhysicalDevice*Features* bits, so this is
// the single gate a shader capability may depend on. A null device (host-side
// planning without a device) reports nothing.
[[nodiscard]] bool extension_enabled(const Device *device,
                                     OptionalExtension extension) noexcept {
    return device != nullptr && device->optional_extension_enabled(extension);
}

// `ShaderCapability::COUNT` maps to `OptionalExtension::COUNT`, which is not a
// negotiable extension.
[[nodiscard]] constexpr bool negotiable_extension(
    OptionalExtension extension) noexcept {
    return extension != OptionalExtension::COUNT;
}

}// namespace vk_ext_shader_caps_detail

ShaderExtensionAvailability shader_extension_availability(
    const Device *device) noexcept {
    ShaderExtensionAvailability availability;
    availability.fma =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_FMA);
    availability.indexing_64bit =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_64BIT_INDEXING);
    availability.uniform_buffer_unsized_array =
        vk_ext_shader_caps_detail::extension_enabled(
            device,
            OptionalExtension::SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY);
    availability.long_vector =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_LONG_VECTOR);
    availability.subgroup_partitioned =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_SUBGROUP_PARTITIONED);
    availability.split_barrier =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_SPLIT_BARRIER);
    availability.ocp_microscaling_types =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_OCP_MICROSCALING_TYPES);
    availability.constant_data =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_CONSTANT_DATA);
    availability.abort =
        vk_ext_shader_caps_detail::extension_enabled(
            device, OptionalExtension::SHADER_ABORT);
    return availability;
}

ShaderCapabilityPlan device_shader_capability_plan(
    const Device *device) noexcept {
    return plan_shader_capabilities(shader_extension_availability(device));
}

spirv::SpirvTargetFeatureMask device_shader_capability_target_features(
    const Device *device) noexcept {
    return device_shader_capability_plan(device).available_target_features;
}

bool shader_capability_available(const Device *device,
                                 ShaderCapability capability) noexcept {
    auto extension = vk_ext_shader_caps_detail::optional_extension_of(capability);
    return vk_ext_shader_caps_detail::negotiable_extension(extension) &&
           vk_ext_shader_caps_detail::extension_enabled(device, extension);
}

bool mark_shader_capability_active(Device *device,
                                   ShaderCapability capability) noexcept {
    auto extension = vk_ext_shader_caps_detail::optional_extension_of(capability);
    LUISA_ASSERT(extension != OptionalExtension::COUNT,
                 "Vulkan shader capability {} has no negotiated optional "
                 "extension.",
                 static_cast<uint32_t>(capability));
    if (!vk_ext_shader_caps_detail::extension_enabled(device, extension)) {
        return false;
    }
    device->mark_optional_extension_active(extension);
    return true;
}

spirv::SpirvTargetFeatureMask require_shader_capability(
    Device *device, ShaderCapability capability,
    luisa::string_view shader_name) noexcept {
    if (!shader_capability_available(device, capability)) {
        auto const &declaration = shader_capability_declaration(capability);
        LUISA_ERROR(
            "Vulkan shader '{}' requires {}, which is not enabled on this "
            "device. Enable it through VulkanDeviceConfigExt or compile the "
            "shader without it.",
            shader_name, declaration.vulkan_extension);
        return 0u;
    }
    static_cast<void>(mark_shader_capability_active(device, capability));
    return shader_capability_declaration(capability).target_feature;
}

}// namespace lc::vk::detail
