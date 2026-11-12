// Test for the Vulkan backend's optional extension negotiation.
// This test covers:
// - name/alias resolution of every optional extension the backend knows
// - the supported/requested/enabled matrix produced by
//   OptionalExtensionRegistry (unsupported, switched off, usable)
// - the feature-structure probe chain (only exposed extensions are chained)
// - the logical-device feature chain (only enabled extensions are chained)
// - imported logical devices never assume an optional extension
// - prerequisites (present extensions need the swapchain extension)

#include "ut/ut.hpp"

#include "vk_extension_registry.h"

#include <luisa/backends/ext/vk_config_ext.h>

#include <cstdio>
#include <cstring>
#include <luisa/core/logging.h>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace vk_extension_registry_test_detail {

using luisa::compute::VulkanDeviceConfigExt;

// Requests every optional extension.
class AllEnabledConfig final : public VulkanDeviceConfigExt {};

// Refuses every optional extension.
class AllDisabledConfig final : public VulkanDeviceConfigExt {
public:
#define LC_TEST_DISABLE(Name)                                     \
    [[nodiscard]] bool Name() const noexcept override {            \
        return false;                                             \
    }
    LC_TEST_DISABLE(enable_descriptor_heap)
    LC_TEST_DISABLE(enable_device_address_commands)
    LC_TEST_DISABLE(enable_unified_image_layouts)
    LC_TEST_DISABLE(enable_copy_memory_indirect)
    LC_TEST_DISABLE(enable_memory_decompression)
    LC_TEST_DISABLE(enable_zero_initialize_device_memory)
    LC_TEST_DISABLE(enable_internally_synchronized_queues)
    LC_TEST_DISABLE(enable_extended_flags)
    LC_TEST_DISABLE(enable_image_tiling_control)
    LC_TEST_DISABLE(enable_present_id2)
    LC_TEST_DISABLE(enable_present_wait2)
    LC_TEST_DISABLE(enable_present_mode_fifo_latest_ready)
    LC_TEST_DISABLE(enable_surface_maintenance1)
    LC_TEST_DISABLE(enable_swapchain_maintenance1)
    LC_TEST_DISABLE(enable_present_timing)
    LC_TEST_DISABLE(enable_multisampled_render_to_swapchain)
    LC_TEST_DISABLE(enable_shader_fma)
    LC_TEST_DISABLE(enable_shader_64bit_indexing)
    LC_TEST_DISABLE(enable_shader_uniform_buffer_unsized_array)
    LC_TEST_DISABLE(enable_shader_long_vector)
    LC_TEST_DISABLE(enable_shader_subgroup_partitioned)
    LC_TEST_DISABLE(enable_shader_split_barrier)
    LC_TEST_DISABLE(enable_shader_ocp_microscaling_types)
    LC_TEST_DISABLE(enable_shader_constant_data)
    LC_TEST_DISABLE(enable_shader_abort)
    LC_TEST_DISABLE(enable_ray_tracing_invocation_reorder)
    LC_TEST_DISABLE(enable_opacity_micromap)
    LC_TEST_DISABLE(enable_custom_resolve)
    LC_TEST_DISABLE(enable_fragment_density_map_offset)
    LC_TEST_DISABLE(enable_primitive_restart_index)
    LC_TEST_DISABLE(enable_texture_compression_astc_3d)
    LC_TEST_DISABLE(enable_pipeline_library_group_handles)
    LC_TEST_DISABLE(enable_external_memory_metal)
    LC_TEST_DISABLE(enable_depth_clamp_zero_one)
    LC_TEST_DISABLE(enable_maintenance8)
    LC_TEST_DISABLE(enable_maintenance9)
    LC_TEST_DISABLE(enable_maintenance10)
    LC_TEST_DISABLE(enable_maintenance11)
    LC_TEST_DISABLE(enable_maintenance12)
    LC_TEST_DISABLE(enable_robustness2)
    LC_TEST_DISABLE(enable_tensor_arm)
    LC_TEST_DISABLE(enable_data_graph_arm)
    LC_TEST_DISABLE(enable_tile_shading_qcom)
    LC_TEST_DISABLE(enable_tile_memory_heap_qcom)
    LC_TEST_DISABLE(enable_partitioned_acceleration_structure)
    LC_TEST_DISABLE(enable_ray_tracing_linear_swept_spheres)
    LC_TEST_DISABLE(enable_dense_geometry_format)
#undef LC_TEST_DISABLE
};

// Refuses one extension and keeps the rest enabled, mirroring a user that
// disables a single optimization.
class SelectiveConfig final : public VulkanDeviceConfigExt {
public:
    [[nodiscard]] bool enable_maintenance8() const noexcept override { return false; }
};

[[nodiscard]] luisa::vector<luisa::string> names(
    std::initializer_list<const char *> values) {
    luisa::vector<luisa::string> result;
    for (auto value : values) { result.emplace_back(value); }
    return result;
}

// Walks a VkBaseOutStructure chain and returns the node with `type`, or null.
[[nodiscard]] void *find_feature(void *chain, VkStructureType type) noexcept {
    for (auto node = static_cast<VkBaseOutStructure *>(chain);
         node != nullptr; node = node->pNext) {
        if (node->sType == type) { return node; }
    }
    return nullptr;
}

[[nodiscard]] size_t chain_length(void *chain) noexcept {
    auto count = size_t{0u};
    for (auto node = static_cast<VkBaseOutStructure *>(chain);
         node != nullptr; node = node->pNext) {
        ++count;
    }
    return count;
}

// VK_KHR_surface_maintenance1 is an *instance* extension, so a test that names
// it must pass it through the instance list; every other optional extension is a
// device extension. This helper splits the names so the individual cases stay
// readable.
void select_names(OptionalExtensionRegistry &registry,
                  const luisa::vector<luisa::string> &names,
                  const VulkanDeviceConfigExt &config,
                  bool imported_device = false) {
    luisa::vector<luisa::string> device_names;
    luisa::vector<luisa::string> instance_names;
    for (auto const &name : names) {
        if (name.find("surface_maintenance1") != luisa::string::npos) {
            instance_names.emplace_back(name);
        } else {
            device_names.emplace_back(name);
        }
    }
    registry.select(luisa::span<const luisa::string>{device_names},
                    luisa::span<const luisa::string>{instance_names},
                    &config, imported_device);
}

}// namespace vk_extension_registry_test_detail

using namespace vk_extension_registry_test_detail;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_optional_extension_name_round_trip"_test = [] {
        expect(luisa::string_view{optional_extension_name(OptionalExtension::PRESENT_ID2)} ==
               "VK_KHR_present_id2");
        expect(optional_extension_from_name("VK_KHR_present_id2") ==
               OptionalExtension::PRESENT_ID2);
        // Aliases resolve to the same identifier.
        expect(optional_extension_from_name("VK_EXT_present_mode_fifo_latest_ready") ==
               OptionalExtension::PRESENT_MODE_FIFO_LATEST_READY);
        expect(optional_extension_from_name("VK_EXT_surface_maintenance1") ==
               OptionalExtension::SURFACE_MAINTENANCE1);
        expect(optional_extension_from_name("VK_NV_ray_tracing_invocation_reorder") ==
               OptionalExtension::RAY_TRACING_INVOCATION_REORDER);
        expect(optional_extension_from_name("VK_EXT_robustness2") ==
               OptionalExtension::ROBUSTNESS2);
        expect(optional_extension_from_name("VK_this_is_not_an_extension") ==
               OptionalExtension::COUNT);
        expect(optional_extension_from_name("") == OptionalExtension::COUNT);
        // Every identifier must be reachable by its own name.
        // Keep the registry alive: `states()` returns a view into it.
        OptionalExtensionRegistry all;
        for (auto const &state : all.states()) {
            expect(state.id != OptionalExtension::COUNT);
            expect(optional_extension_from_name(optional_extension_name(state.id)) ==
                   state.id);
        }
    };

    "vk_optional_extension_no_support_enables_nothing"_test = [] {
        AllEnabledConfig config;
        OptionalExtensionRegistry registry;
        select_names(registry, {}, config, false);
        expect(registry.feature_query_chain() == nullptr);
        registry.resolve();
        expect(registry.enabled_extension_names().empty());
        for (auto const &state : registry.states()) {
            expect(!state.supported_by_device);
            expect(!state.requested_by_config);
            expect(!state.enabled);
        }
        expect(registry.status_string().empty())
            << "an empty negotiation must not produce log lines";
    };

    "vk_optional_extension_config_switch_disables_supported_extension"_test = [] {
        AllDisabledConfig config;
        auto supported = names({"VK_KHR_maintenance8"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        registry.resolve();
        expect(registry.supported(OptionalExtension::MAINTENANCE8));
        expect(!registry.requested(OptionalExtension::MAINTENANCE8));
        expect(!registry.enabled(OptionalExtension::MAINTENANCE8));
        expect(registry.enabled_extension_names().empty());
        // A switched-off extension must not be probed either: its feature
        // structure would otherwise reach the driver for nothing.
        expect(registry.feature_query_chain() == nullptr);
    };

    "vk_optional_extension_single_switch_does_not_affect_others"_test = [] {
        SelectiveConfig config;
        auto supported = names({"VK_KHR_maintenance8", "VK_KHR_surface_maintenance1"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        // A swapchain must exist for the instance-level surface maintenance
        // extension, and maintenance8's feature bit must be reported.
        registry.note_enabled_extension("VK_KHR_swapchain");
        expect(find_feature(registry.feature_query_chain(),
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR) == nullptr)
            << "a switched-off extension must not be probed at all";
        registry.resolve();
        expect(!registry.enabled(OptionalExtension::MAINTENANCE8))
            << "one switched-off extension must not affect the others";
        expect(registry.enabled(OptionalExtension::SURFACE_MAINTENANCE1))
            << "extension without a feature structure is enabled by name alone";
        // An instance-level extension is never part of the device extension list.
        for (auto name : registry.enabled_extension_names()) {
            expect(luisa::string_view{name} != "VK_KHR_surface_maintenance1");
            expect(luisa::string_view{name} != "VK_EXT_surface_maintenance1");
        }
    };

    "vk_optional_extension_alias_spelling_is_kept"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_EXT_surface_maintenance1"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        registry.note_enabled_extension("VK_KHR_swapchain");
        registry.resolve();
        auto const &state = registry.state(OptionalExtension::SURFACE_MAINTENANCE1);
        expect(luisa::string_view{state.name} == "VK_EXT_surface_maintenance1")
            << "the spelling the instance exposes must be the one enabled";
        expect(state.enabled);
        // Instance-level extensions go to vkCreateInstance, never to
        // VkDeviceCreateInfo: passing one to vkCreateDevice is an error.
        expect(registry.enabled_extension_names().empty());
    };

    "vk_optional_extension_feature_bit_decides_enablement"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_KHR_maintenance8"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        auto probe = find_feature(registry.feature_query_chain(),
                                  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR);
        expect(probe != nullptr) << "a requested, supported extension is probed";
        // Simulate a driver that exposes the name but reports the feature off.
        registry.resolve();
        expect(!registry.enabled(OptionalExtension::MAINTENANCE8));
        expect(registry.device_feature_chain(nullptr) == nullptr)
            << "a rejected extension must not extend the device chain";
        // Simulate the feature being available.
        static_cast<VkPhysicalDeviceMaintenance8FeaturesKHR *>(probe)->maintenance8 = VK_TRUE;
        registry.resolve();
        expect(registry.enabled(OptionalExtension::MAINTENANCE8));
        auto chain = registry.device_feature_chain(nullptr);
        expect(find_feature(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR) == probe);
        expect(static_cast<VkPhysicalDeviceMaintenance8FeaturesKHR *>(probe)->maintenance8 == VK_TRUE)
            << "the enabled feature bit is what reaches vkCreateDevice";
    };

    "vk_optional_extension_chain_only_carries_enabled_extensions"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_KHR_maintenance8",
                                "VK_KHR_shader_fma",
                                "VK_KHR_surface_maintenance1"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        // Only the two entries that own a feature structure are probed; the
        // featureless one is enabled by name.
        expect(chain_length(registry.feature_query_chain()) == 2u);
        auto maintenance = find_feature(registry.feature_query_chain(),
                                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR);
        auto fma = find_feature(registry.feature_query_chain(),
                                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR);
        expect(maintenance != nullptr);
        expect(fma != nullptr);
        static_cast<VkPhysicalDeviceMaintenance8FeaturesKHR *>(maintenance)->maintenance8 = VK_TRUE;
        static_cast<VkPhysicalDeviceShaderFmaFeaturesKHR *>(fma)->shaderFmaFloat32 = VK_TRUE;
        registry.note_enabled_extension("VK_KHR_swapchain");
        registry.resolve();
        expect(registry.enabled(OptionalExtension::MAINTENANCE8));
        expect(registry.enabled(OptionalExtension::SHADER_FMA));
        expect(registry.enabled(OptionalExtension::SURFACE_MAINTENANCE1));
        auto chain = registry.device_feature_chain(nullptr);
        expect(chain_length(chain) == 2u) << "featureless extension adds no node";
        expect(find_feature(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR) != nullptr);
        expect(find_feature(chain, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR) != nullptr);
    };

    "vk_optional_extension_partial_feature_support_rejects_extension"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_KHR_shader_fma"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        auto fma = static_cast<VkPhysicalDeviceShaderFmaFeaturesKHR *>(
            find_feature(registry.feature_query_chain(),
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR));
        expect(fma != nullptr);
        // FMA on FP16 only is not enough for the backend: it needs the FP32 form.
        fma->shaderFmaFloat16 = VK_TRUE;
        registry.resolve();
        expect(!registry.enabled(OptionalExtension::SHADER_FMA));
    };

    "vk_optional_extension_imported_device_is_never_assumed"_test = [] {
        AllEnabledConfig config;
        // Even a device that exposes everything cannot attest its enabled
        // feature chain when the logical device is imported.
        luisa::vector<luisa::string> supported;
        // Keep the registry alive: `states()` returns a view into it.
        OptionalExtensionRegistry all;
        for (auto const &state : all.states()) {
            supported.emplace_back(state.name == nullptr ? "" : state.name);
        }
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, true);
        registry.resolve();
        for (auto const &state : registry.states()) {
            expect(!state.supported_by_device);
            expect(!state.requested_by_config);
            expect(!state.enabled);
        }
        expect(registry.enabled_extension_names().empty());
    };

    "vk_optional_extension_prerequisite_blocks_present_extensions"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_KHR_present_id2", "VK_KHR_present_wait2"});
        {
            // Without a swapchain the present extensions are unusable.
            OptionalExtensionRegistry registry;
            select_names(registry, supported, config, false);
            registry.resolve();
            expect(registry.supported(OptionalExtension::PRESENT_ID2));
            expect(!registry.enabled(OptionalExtension::PRESENT_ID2));
            expect(!registry.enabled(OptionalExtension::PRESENT_WAIT2));
        }
        {
            OptionalExtensionRegistry registry;
            select_names(registry, supported, config, false);
            // Simulate a driver that exposes both present features.
            auto id2 = static_cast<VkPhysicalDevicePresentId2FeaturesKHR *>(
                find_feature(registry.feature_query_chain(),
                             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR));
            auto wait2 = static_cast<VkPhysicalDevicePresentWait2FeaturesKHR *>(
                find_feature(registry.feature_query_chain(),
                             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR));
            expect(id2 != nullptr);
            expect(wait2 != nullptr);
            id2->presentId2 = VK_TRUE;
            wait2->presentWait2 = VK_TRUE;
            registry.note_enabled_extension("VK_KHR_swapchain");
            registry.resolve();
            expect(registry.enabled(OptionalExtension::PRESENT_ID2));
            expect(registry.enabled(OptionalExtension::PRESENT_WAIT2));
        }
    };

    "vk_optional_extension_instance_level_extension_is_matched_separately"_test = [] {
        AllEnabledConfig config;
        // VK_KHR_surface_maintenance1 is an instance extension: it must never be
        // matched against the device extension list ...
        auto device_names = names({"VK_KHR_surface_maintenance1"});
        {
            OptionalExtensionRegistry registry;
            registry.select(luisa::span<const luisa::string>{device_names},
                            &config, false);
            expect(!registry.supported(OptionalExtension::SURFACE_MAINTENANCE1))
                << "an instance extension must not be matched against device "
                   "extensions";
        }
        // ... and only when the instance enabled it *and* a swapchain exists.
        {
            OptionalExtensionRegistry registry;
            registry.select(luisa::span<const luisa::string>{},
                            luisa::span<const luisa::string>{device_names},
                            &config, false);
            registry.resolve();
            expect(registry.supported(OptionalExtension::SURFACE_MAINTENANCE1));
            expect(!registry.enabled(OptionalExtension::SURFACE_MAINTENANCE1))
                << "no swapchain means no surface maintenance1";
            registry.note_enabled_extension("VK_KHR_swapchain");
            registry.resolve();
            expect(registry.enabled(OptionalExtension::SURFACE_MAINTENANCE1));
        }
        // An imported instance cannot attest its enabled extensions.
        {
            OptionalExtensionRegistry registry;
            registry.select(luisa::span<const luisa::string>{},
                            luisa::span<const luisa::string>{device_names},
                            &config, true);
            registry.note_enabled_extension("VK_KHR_swapchain");
            registry.resolve();
            expect(!registry.enabled(OptionalExtension::SURFACE_MAINTENANCE1));
        }
    };

    "vk_optional_extension_enabled_names_are_unique"_test = [] {
        AllEnabledConfig config;
        // A repeated device extension name must be negotiated once. The
        // instance-level surface maintenance extension is covered separately.
        auto supported = names({"VK_KHR_maintenance8",
                                "VK_KHR_maintenance8",
                                "VK_KHR_maintenance8"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        auto maintenance = static_cast<VkPhysicalDeviceMaintenance8FeaturesKHR *>(
            find_feature(registry.feature_query_chain(),
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR));
        expect(maintenance != nullptr);
        maintenance->maintenance8 = VK_TRUE;
        registry.resolve();
        auto enabled = registry.enabled_extension_names();
        expect(enabled.size() == 1u);
        // force_enable must not duplicate a name either.
        expect(registry.force_enable(OptionalExtension::MAINTENANCE8));
        expect(registry.enabled_extension_names().size() == 1u);
    };

    "vk_optional_extension_force_enable_requires_support"_test = [] {
        AllEnabledConfig config;
        OptionalExtensionRegistry registry;
        select_names(registry, {}, config, false);
        registry.resolve();
        expect(!registry.force_enable(OptionalExtension::SURFACE_MAINTENANCE1));
        expect(!registry.enabled(OptionalExtension::SURFACE_MAINTENANCE1));
    };

    // `select()` chains the feature structure of an extension only when
    // VulkanDeviceConfigExt asks for it. force_enable() therefore has no probed
    // feature state to work with for a switched-off extension: it must refuse,
    // because enabling the name without its VkPhysicalDevice*Features* bits would
    // create a logical device that reports the extension but not the feature it
    // exists for (and chaining the structure afterwards is impossible - the
    // vkGetPhysicalDeviceFeatures2 probe already ran).
    "vk_optional_extension_force_enable_never_enables_unprobed_features"_test = [] {
        AllDisabledConfig config;
        auto supported = names({"VK_KHR_maintenance8"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        expect(registry.supported(OptionalExtension::MAINTENANCE8));
        expect(!registry.requested(OptionalExtension::MAINTENANCE8));
        expect(find_feature(registry.feature_query_chain(),
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR) ==
               nullptr)
            << "a switched-off extension is never probed";
        expect(!registry.force_enable(OptionalExtension::MAINTENANCE8))
            << "force_enable must fail closed without probed feature bits";
        registry.resolve();
        expect(!registry.enabled(OptionalExtension::MAINTENANCE8));
        expect(registry.enabled_extension_names().empty())
            << "no name may reach VkDeviceCreateInfo without its feature struct";
    };

    "vk_optional_extension_status_string_reports_state"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_KHR_maintenance8", "VK_KHR_surface_maintenance1"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        auto maintenance = static_cast<VkPhysicalDeviceMaintenance8FeaturesKHR *>(
            find_feature(registry.feature_query_chain(),
                         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR));
        expect(maintenance != nullptr);
        maintenance->maintenance8 = VK_TRUE;
        registry.note_enabled_extension("VK_KHR_swapchain");
        registry.resolve();
        auto status = registry.status_string();
        expect(status.find("VK_KHR_maintenance8 [enabled]") != luisa::string::npos);
        expect(status.find("VK_KHR_surface_maintenance1 [enabled]") != luisa::string::npos);
        expect(status.find("VK_EXT_descriptor_heap") == luisa::string::npos)
            << "extensions the device never exposed stay out of the log";
    };

    "vk_optional_extension_debug_only_extensions_follow_ndebug"_test = [] {
        AllEnabledConfig config;
        auto supported = names({"VK_KHR_robustness2"});
        OptionalExtensionRegistry registry;
        select_names(registry, supported, config, false);
        registry.resolve();
#ifdef NDEBUG
        expect(!registry.requested(OptionalExtension::ROBUSTNESS2))
            << "robustness2 is a debug-only aid";
        expect(!registry.enabled(OptionalExtension::ROBUSTNESS2));
#else
        expect(registry.requested(OptionalExtension::ROBUSTNESS2));
#endif
    };
}
