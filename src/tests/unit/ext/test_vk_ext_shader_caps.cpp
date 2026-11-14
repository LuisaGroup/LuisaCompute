// Test for Vulkan shader capability extension planning.
// This test covers:
// - the persistent target-feature bits of every VK_KHR/EXT_shader_* capability
//   (stable values, no collisions, all inside `known_mask`)
// - the SPIR-V declaration (capability id and extension string) each enabled
//   extension unlocks, checked against the SPIR-V specification
// - the full on/off capability matrix of `plan_shader_capabilities`
// - the fail-closed declaration helpers (`missing_shader_capabilities`,
//   `required_shader_capability_features`)

#include "ut/ut.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "vk_shader_capabilities.h"

using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_vk_ext_shader_caps_detail {

using namespace lc::vk::detail;
using lc::spirv::SpirvTargetFeatureMask;
namespace target_feature = lc::spirv::target_feature;

// One row per capability: the persistent bit, the negotiated Vulkan extension,
// and the SPIR-V declaration it unlocks.
struct CapabilityExpectation {
    ShaderCapability capability;
    SpirvTargetFeatureMask target_feature;
    luisa::string_view vulkan_extension;
    luisa::string_view spirv_extension;
    uint32_t spirv_capability;
    luisa::string_view spirv_capability_name;
};

inline constexpr std::array capability_expectations{
    CapabilityExpectation{ShaderCapability::FMA,
                          0x0000080000000000ull,
                          "VK_KHR_shader_fma",
                          "SPV_KHR_fma",
                          6030u,
                          "FMAKHR"},
    CapabilityExpectation{ShaderCapability::INDEXING_64BIT,
                          0x0000100000000000ull,
                          "VK_EXT_shader_64bit_indexing",
                          "SPV_EXT_shader_64bit_indexing",
                          5426u,
                          "Shader64BitIndexingEXT"},
    CapabilityExpectation{ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY,
                          0x0000200000000000ull,
                          "VK_EXT_shader_uniform_buffer_unsized_array",
                          {},
                          0u,
                          {}},
    CapabilityExpectation{ShaderCapability::LONG_VECTOR,
                          0x0000400000000000ull,
                          "VK_EXT_shader_long_vector",
                          "SPV_EXT_long_vector",
                          5425u,
                          "LongVectorEXT"},
    CapabilityExpectation{ShaderCapability::SUBGROUP_PARTITIONED,
                          0x0000800000000000ull,
                          "VK_EXT_shader_subgroup_partitioned",
                          "SPV_EXT_shader_subgroup_partitioned",
                          5297u,
                          "GroupNonUniformPartitionedEXT"},
    CapabilityExpectation{ShaderCapability::SPLIT_BARRIER,
                          0x0001000000000000ull,
                          "VK_EXT_shader_split_barrier",
                          "SPV_EXT_split_barrier",
                          6141u,
                          "SplitBarrierEXT"},
    CapabilityExpectation{ShaderCapability::OCP_MICROSCALING_TYPES,
                          0x0002000000000000ull,
                          "VK_EXT_shader_ocp_microscaling_types",
                          "SPV_EXT_ocp_microscaling_types",
                          4229u,
                          "Float4EXT"},
    CapabilityExpectation{ShaderCapability::CONSTANT_DATA,
                          0x0004000000000000ull,
                          "VK_KHR_shader_constant_data",
                          "SPV_KHR_constant_data",
                          5146u,
                          "ConstantDataKHR"},
    CapabilityExpectation{ShaderCapability::ABORT,
                          0x0008000000000000ull,
                          "VK_KHR_shader_abort",
                          "SPV_KHR_abort",
                          5120u,
                          "AbortKHR"},
    CapabilityExpectation{ShaderCapability::MIXED_FLOAT_DOT_PRODUCT,
                          0x0020000000000000ull,
                          "VK_VALVE_shader_mixed_float_dot_product",
                          "SPV_VALVE_mixed_float_dot_product",
                          6912u,
                          "DotProductFloat16AccFloat32VALVE"}};

// Every capability switched off except `capability`.
[[nodiscard]] ShaderExtensionAvailability only(
    ShaderCapability capability) noexcept {
    ShaderExtensionAvailability availability;
    switch (capability) {
        case ShaderCapability::FMA: availability.fma = true; break;
        case ShaderCapability::INDEXING_64BIT:
            availability.indexing_64bit = true;
            break;
        case ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY:
            availability.uniform_buffer_unsized_array = true;
            break;
        case ShaderCapability::LONG_VECTOR:
            availability.long_vector = true;
            break;
        case ShaderCapability::SUBGROUP_PARTITIONED:
            availability.subgroup_partitioned = true;
            break;
        case ShaderCapability::SPLIT_BARRIER:
            availability.split_barrier = true;
            break;
        case ShaderCapability::OCP_MICROSCALING_TYPES:
            availability.ocp_microscaling_types = true;
            break;
        case ShaderCapability::CONSTANT_DATA:
            availability.constant_data = true;
            break;
        case ShaderCapability::ABORT: availability.abort = true; break;
        case ShaderCapability::MIXED_FLOAT_DOT_PRODUCT:
            availability.mixed_float_dot_product = true;
            break;
        case ShaderCapability::COUNT: break;
    }
    return availability;
}

[[nodiscard]] constexpr ShaderExtensionAvailability all_enabled() noexcept {
    ShaderExtensionAvailability availability;
    availability.fma = true;
    availability.indexing_64bit = true;
    availability.uniform_buffer_unsized_array = true;
    availability.long_vector = true;
    availability.subgroup_partitioned = true;
    availability.split_barrier = true;
    availability.ocp_microscaling_types = true;
    availability.constant_data = true;
    availability.abort = true;
    availability.mixed_float_dot_product = true;
    return availability;
}

[[nodiscard]] constexpr ShaderCapabilityMask all_capabilities() noexcept {
    return (ShaderCapabilityMask{1u} << static_cast<uint32_t>(ShaderCapability::COUNT)) - 1u;
}

// Number of SPIR-V capability rows declared for one capability.
[[nodiscard]] constexpr size_t spirv_declaration_count(
    ShaderCapability capability) noexcept {
    auto count = size_t{0u};
    for (auto const &declaration : shader_spirv_capability_declarations) {
        if (declaration.capability == capability) { count++; }
    }
    return count;
}

}// namespace test_vk_ext_shader_caps_detail

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using namespace lc::vk::detail;
    using namespace test_vk_ext_shader_caps_detail;

    "vk_shader_capabilities_cover_every_extension_of_the_work_package"_test = [] {
        expect(eq(shader_capability_count, 10u));
        expect(eq(shader_capability_declarations.size(), 10u));
        expect(eq(all_capabilities(), 0x000003ffu));
        // One bit per enumerator, in declaration order.
        for (uint32_t i = 0u; i < shader_capability_count; i++) {
            expect(eq(shader_capability_bit(static_cast<ShaderCapability>(i)),
                      ShaderCapabilityMask{1u} << i));
        }
        expect(eq(shader_capability_bit(ShaderCapability::COUNT), 1024u));
    };

    "vk_shader_capability_bits_are_stable_and_unique"_test = [] {
        SpirvTargetFeatureMask seen{0u};
        for (auto const &expectation : capability_expectations) {
            auto const &declaration =
                shader_capability_declaration(expectation.capability);
            // Exact literal value: a future reorder or reuse of an ABI bit
            // must fail here, not in a shader cache.
            expect(eq(declaration.target_feature, expectation.target_feature))
                << "persistent shader capability bit changed for "
                << declaration.vulkan_extension;
            // Single bit, and never shared with another capability.
            expect(eq(declaration.target_feature &
                          (declaration.target_feature - 1u),
                      0u));
            expect(declaration.target_feature != 0u);
            expect(eq(seen & declaration.target_feature, 0u));
            seen |= declaration.target_feature;
            // Part of the persistent mask, and named for diagnostics.
            expect(eq(lc::spirv::target_feature::known_mask &
                          declaration.target_feature,
                      declaration.target_feature));
            expect(!lc::spirv::spirv_target_feature_name(declaration.target_feature)
                        .empty());
        }
        // The nine shader-capability bits are the contiguous run above
        // shaderBFloat16Type; the VALVE mixed-float-dot-product bit follows the
        // ray tracing opacity micromap bit.
        expect(eq(seen, 0x002ff80000000000ull));
        expect(eq(seen, lc::spirv::target_feature::shader_fma_float32 |
                            lc::spirv::target_feature::shader_64bit_indexing |
                            lc::spirv::target_feature::shader_uniform_buffer_unsized_array |
                            lc::spirv::target_feature::shader_long_vector |
                            lc::spirv::target_feature::shader_subgroup_partitioned |
                            lc::spirv::target_feature::shader_split_barrier |
                            lc::spirv::target_feature::shader_ocp_microscaling_types |
                            lc::spirv::target_feature::shader_constant_data |
                            lc::spirv::target_feature::shader_abort |
                            lc::spirv::target_feature::shader_mixed_float_dot_product_float16_acc_float32));
        expect(eq(lc::spirv::target_feature::shader_fma_float32, 0x0000080000000000ull));
        expect(eq(lc::spirv::target_feature::shader_64bit_indexing, 0x0000100000000000ull));
        expect(eq(lc::spirv::target_feature::shader_uniform_buffer_unsized_array, 0x0000200000000000ull));
        expect(eq(lc::spirv::target_feature::shader_long_vector, 0x0000400000000000ull));
        expect(eq(lc::spirv::target_feature::shader_subgroup_partitioned, 0x0000800000000000ull));
        expect(eq(lc::spirv::target_feature::shader_split_barrier, 0x0001000000000000ull));
        expect(eq(lc::spirv::target_feature::shader_ocp_microscaling_types, 0x0002000000000000ull));
        expect(eq(lc::spirv::target_feature::shader_constant_data, 0x0004000000000000ull));
        expect(eq(lc::spirv::target_feature::shader_abort, 0x0008000000000000ull));
        expect(eq(lc::spirv::target_feature::shader_mixed_float_dot_product_float16_acc_float32, 0x0020000000000000ull));
        // `known_mask` covers those ten bits plus the ray tracing opacity
        // micromap bit, which belongs to the ray tracing capability family
        // rather than to the shader capability table iterated above.
        expect(eq(lc::spirv::target_feature::ray_tracing_opacity_micromap, 0x0010000000000000ull));
        expect(eq(lc::spirv::target_feature::known_mask, 0x003fffffffffffffull));
        expect(eq(lc::spirv::target_feature::known_mask & seen, seen));
    };

    "vk_shader_capability_declarations_map_to_the_negotiated_extensions"_test = [] {
        for (auto const &expectation : capability_expectations) {
            auto const &declaration =
                shader_capability_declaration(expectation.capability);
            expect(declaration.capability == expectation.capability);
            expect(luisa::string_view{declaration.vulkan_extension} ==
                   expectation.vulkan_extension);
            expect(luisa::string_view{declaration.vulkan_extension}.starts_with(
                "VK_"));
        }
    };

    "vk_shader_spirv_declarations_match_the_spirv_specification"_test = [] {
        // Numeric values are from the SPIR-V headers (spirv.h); the strings are
        // the extension names the module has to declare.
        expect(eq(shader_spirv_capability_declarations.size(), 14u));
        for (auto const &expectation : capability_expectations) {
            auto found{false};
            for (auto const &declaration :
                 shader_spirv_capability_declarations) {
                if (declaration.capability != expectation.capability) {
                    continue;
                }
                if (!found) {
                    // Every capability names exactly one SPIR-V extension (or
                    // no extension at all when it reuses existing capability
                    // support), and the first row carries the expectation.
                    found = true;
                    if (expectation.spirv_capability == 0u) {
                        expect(declaration.spirv_extension == nullptr);
                        expect(declaration.spirv_capability_name == nullptr);
                        expect(eq(spirv_declaration_count(expectation.capability), 1u));
                        continue;
                    }
                    expect(luisa::string_view{declaration.spirv_extension} ==
                           expectation.spirv_extension);
                    expect(eq(declaration.spirv_capability,
                              expectation.spirv_capability));
                    expect(luisa::string_view{declaration.spirv_capability_name} ==
                           expectation.spirv_capability_name);
                } else {
                    // Additional scalar types of the OCP microscaling group
                    // share one SPIR-V extension.
                    expect(luisa::string_view{declaration.spirv_extension} ==
                           expectation.spirv_extension);
                }
            }
            expect(found) << "no SPIR-V declaration for capability "
                          << static_cast<uint32_t>(expectation.capability);
        }
        // The OCP microscaling group needs one capability per scalar type.
        expect(eq(spirv_declaration_count(ShaderCapability::OCP_MICROSCALING_TYPES), 5u));
        expect(eq(spirv_declaration_count(ShaderCapability::ABORT), 1u));
        expect(eq(spirv_declaration_count(
                      ShaderCapability::UNIFORM_BUFFER_UNSIZED_ARRAY),
                  1u));
    };

    "vk_shader_capabilities_plan_nothing_when_every_switch_is_off"_test = [] {
        // A physical device that supports every extension but whose switches are
        // off (or an imported device, which can never report them) plans the
        // empty set: the existing code path stays the fallback.
        constexpr auto plan =
            plan_shader_capabilities(ShaderExtensionAvailability{});
        static_assert(plan.available == 0u);
        static_assert(plan.available_target_features == 0u);
        expect(eq(plan.available, 0u));
        expect(eq(plan.available_target_features, 0u));
        expect(!plan.any());
        expect(eq(plan.count(), 0u));
        for (auto const &expectation : capability_expectations) {
            expect(!plan.is_available(expectation.capability));
            expect(eq(missing_shader_capabilities(
                          plan, shader_capability_bit(expectation.capability)),
                      shader_capability_bit(expectation.capability)));
        }
        expect(eq(missing_shader_capabilities(plan, all_capabilities()),
                  all_capabilities()));
        expect(eq(required_shader_capability_features(
                      plan, all_capabilities()),
                  0u));
    };

    "vk_shader_capabilities_plan_each_extension_alone"_test = [] {
        for (auto const &expectation : capability_expectations) {
            auto plan = plan_shader_capabilities(only(expectation.capability));
            expect(eq(plan.available,
                      shader_capability_bit(expectation.capability)))
                << "capability " << static_cast<uint32_t>(expectation.capability)
                << " is not planned alone";
            expect(eq(plan.available_target_features,
                      expectation.target_feature));
            expect(plan.any());
            expect(eq(plan.count(), 1u));
            expect(plan.is_available(expectation.capability));
            expect(eq(missing_shader_capabilities(
                          plan, shader_capability_bit(expectation.capability)),
                      0u));
            expect(eq(required_shader_capability_features(
                          plan, all_capabilities()),
                      expectation.target_feature));
            // No other capability is reported.
            for (auto const &other : capability_expectations) {
                if (other.capability == expectation.capability) { continue; }
                expect(!plan.is_available(other.capability));
                expect(eq(missing_shader_capabilities(
                              plan, shader_capability_bit(other.capability)),
                          shader_capability_bit(other.capability)));
            }
        }
    };

    "vk_shader_capabilities_plan_every_extension_when_all_are_enabled"_test = [] {
        constexpr auto plan = plan_shader_capabilities(all_enabled());
        static_assert(plan.available == all_capabilities());
        static_assert(plan.available_target_features == 0x002ff80000000000ull);
        expect(eq(plan.available, all_capabilities()));
        expect(eq(plan.available_target_features, 0x002ff80000000000ull));
        expect(plan.any());
        expect(eq(plan.count(), shader_capability_count));
        expect(eq(missing_shader_capabilities(plan, all_capabilities()), 0u));
        expect(eq(required_shader_capability_features(plan, all_capabilities()),
                  plan.available_target_features));
        for (auto const &expectation : capability_expectations) {
            expect(plan.is_available(expectation.capability));
        }
    };

    "vk_shader_capabilities_plan_is_the_union_of_the_enabled_switches"_test = [] {
        // Every single switch is independent: enabling one never drags in
        // another capability or bit.
        auto availability = ShaderExtensionAvailability{};
        auto expected_mask = ShaderCapabilityMask{0u};
        auto expected_features = SpirvTargetFeatureMask{0u};
        for (auto const &expectation : capability_expectations) {
            availability = only(expectation.capability);
            auto plan = plan_shader_capabilities(availability);
            expect(eq(plan.available, shader_capability_bit(expectation.capability)));
            expect(eq(plan.available & ~shader_capability_bit(expectation.capability), 0u));
            expected_mask |= shader_capability_bit(expectation.capability);
            expected_features |= expectation.target_feature;
        }
        // Two arbitrary proper subsets.
        auto partial = ShaderExtensionAvailability{};
        partial.abort = true;
        partial.long_vector = true;
        auto plan = plan_shader_capabilities(partial);
        expect(eq(plan.available, shader_capability_bit(ShaderCapability::ABORT) |
                                      shader_capability_bit(ShaderCapability::LONG_VECTOR)));
        expect(eq(plan.available_target_features,
                  lc::spirv::target_feature::shader_abort |
                      lc::spirv::target_feature::shader_long_vector));
        expect(eq(plan.count(), 2u));
        expect(eq(expected_mask, all_capabilities()));
        expect(eq(expected_features, 0x002ff80000000000ull));
        // Enabling one capability leaves every other one exactly as planned.
        auto only_split = plan_shader_capabilities(only(ShaderCapability::SPLIT_BARRIER));
        expect(eq(missing_shader_capabilities(
                      only_split, shader_capability_bit(ShaderCapability::ABORT)),
                  shader_capability_bit(ShaderCapability::ABORT)));
        expect(eq(required_shader_capability_features(
                      only_split, all_capabilities()),
                  lc::spirv::target_feature::shader_split_barrier));
    };

    "vk_shader_capability_requirement_helpers_ignore_unknown_bits"_test = [] {
        constexpr auto plan = plan_shader_capabilities(all_enabled());
        // Bits above ShaderCapability::COUNT are not capabilities.
        constexpr auto unknown = 1u << 20u;
        expect(eq(shader_capability_target_features(unknown), 0u));
        expect(eq(required_shader_capability_features(plan, unknown), 0u));
        expect(eq(missing_shader_capabilities(plan, unknown), unknown));
        // Capability bits outside the plan stay missing even when the unknown
        // bits are asked for together with known ones.
        constexpr auto off = plan_shader_capabilities(ShaderExtensionAvailability{});
        expect(eq(missing_shader_capabilities(off, unknown | all_capabilities()),
                  unknown | all_capabilities()));
    };

    "vk_shader_capability_target_features_cover_the_mask"_test = [] {
        // The unconditional mapping must stay inside the persistent mask, so a
        // required bit can never be rejected as unknown.
        constexpr auto features =
            shader_capability_target_features(all_capabilities());
        static_assert((features & ~lc::spirv::target_feature::known_mask) == 0u);
        expect(eq(features & ~lc::spirv::target_feature::known_mask, 0u));
        expect(eq(features, 0x002ff80000000000ull));
    };
}
