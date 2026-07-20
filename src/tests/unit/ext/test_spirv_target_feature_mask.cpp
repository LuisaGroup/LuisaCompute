// Test for the persistent SPIR-V target-feature mask contract.
// This test covers:
// - stable, named bit values and diagnostic names
// - exact and superset requirement checks
// - fail-closed reporting of missing and unknown required bits

#include "ut/ut.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "spirv_codegen/target_feature_mask.h"
#include "spirv_codegen/target_features.h"

using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_target_feature_mask_has_stable_named_values"_test = [] {
        constexpr std::array<uint64_t, 40u> expected_bits{
            0x0000000000000001ull,
            0x0000000000000002ull,
            0x0000000000000004ull,
            0x0000000000000008ull,
            0x0000000000000010ull,
            0x0000000000000020ull,
            0x0000000000000040ull,
            0x0000000000000080ull,
            0x0000000000000100ull,
            0x0000000000000200ull,
            0x0000000000000400ull,
            0x0000000000000800ull,
            0x0000000000001000ull,
            0x0000000000002000ull,
            0x0000000000004000ull,
            0x0000000000008000ull,
            0x0000000000010000ull,
            0x0000000000020000ull,
            0x0000000000040000ull,
            0x0000000000080000ull,
            0x0000000000100000ull,
            0x0000000000200000ull,
            0x0000000000400000ull,
            0x0000000000800000ull,
            0x0000000001000000ull,
            0x0000000002000000ull,
            0x0000000004000000ull,
            0x0000000008000000ull,
            0x0000000010000000ull,
            0x0000000020000000ull,
            0x0000000040000000ull,
            0x0000000080000000ull,
            0x0000000100000000ull,
            0x0000000200000000ull,
            0x0000000400000000ull,
            0x0000000800000000ull,
            0x0000001000000000ull,
            0x0000002000000000ull,
            0x0000004000000000ull,
            0x0000008000000000ull};
        constexpr std::array<std::string_view, 40u> expected_names{
            "shaderSampledImageArrayDynamicIndexing",
            "shaderSampledImageArrayNonUniformIndexing",
            "shaderResourceMinLod",
            "shaderBufferFloat32Atomics",
            "shaderBufferFloat32AtomicAdd",
            "shaderBufferFloat32AtomicMinMax",
            "shaderSharedFloat32Atomics",
            "shaderSharedFloat32AtomicAdd",
            "shaderSharedFloat32AtomicMinMax",
            "subgroupBasic",
            "subgroupVote",
            "subgroupArithmetic",
            "subgroupBallot",
            "subgroupShuffle",
            "shaderSubgroupExtendedTypes",
            "shaderStorageImageReadWithoutFormat",
            "shaderStorageImageWriteWithoutFormat",
            "shaderFloat8",
            "shaderFloat16",
            "shaderFloat64",
            "shaderInt8",
            "shaderInt16",
            "shaderInt64",
            "storageBuffer8BitAccess",
            "uniformAndStorageBuffer8BitAccess",
            "storageBuffer16BitAccess",
            "uniformAndStorageBuffer16BitAccess",
            "rayQuery",
            "samplerAnisotropy",
            "shaderBufferInt64Atomics",
            "shaderSharedInt64Atomics",
            "descriptorIndexing",
            "runtimeDescriptorArray",
            "descriptorBindingPartiallyBound",
            "shaderStorageBufferArrayNonUniformIndexing",
            "descriptorBindingSampledImageUpdateAfterBind",
            "descriptorBindingStorageBufferUpdateAfterBind",
            "shaderStorageBufferArrayDynamicIndexing",
            "shaderDeviceClock",
            "bufferDeviceAddress"};

        constexpr auto &features =
            lc::spirv::spirv_target_feature_descriptions;
        static_assert(features.size() == expected_bits.size());
        for (std::size_t i = 0u; i < features.size(); i++) {
            expect(eq(features[i].bit, expected_bits[i]))
                << "persistent SPIR-V feature bit changed at index " << i;
            expect(features[i].name == expected_names[i])
                << "SPIR-V feature diagnostic name changed at index " << i;
            expect(lc::spirv::spirv_target_feature_name(expected_bits[i]) ==
                   expected_names[i]);
        }
        expect(eq(lc::spirv::target_feature::known_mask,
                  0x000000ffffffffffull));
        expect(lc::spirv::spirv_target_feature_name(0u).empty());
        expect(lc::spirv::spirv_target_feature_name(
                   expected_bits[0] | expected_bits[1])
                   .empty());
        expect(lc::spirv::spirv_target_feature_name(1ull << 63u).empty());
    };

    "spirv_target_feature_list_reports_known_and_unknown_bits"_test = [] {
        using namespace lc::spirv;
        constexpr auto unknown = 1ull << 63u;
        constexpr auto mask =
            target_feature::shader_resource_min_lod |
            target_feature::ray_query |
            target_feature::descriptor_indexing |
            unknown;
        constexpr auto list = list_spirv_target_features(mask);
        static_assert(list.count == 3u);
        static_assert(list.unknown_bits == unknown);
        expect(eq(list.count, 3u));
        expect(eq(list.unknown_bits, unknown));
        expect(list.features[0].name == "shaderResourceMinLod");
        expect(list.features[1].name == "rayQuery");
        expect(list.features[2].name == "descriptorIndexing");
        std::size_t iterated{};
        for ([[maybe_unused]] auto feature : list) { iterated++; }
        expect(eq(iterated, list.count));
    };

    "spirv_target_feature_requirements_accept_zero_exact_and_superset"_test = [] {
        using namespace lc::spirv;
        constexpr auto zero =
            check_spirv_target_feature_requirements(0u, 0u);
        static_assert(static_cast<bool>(zero));
        expect(static_cast<bool>(zero));
        expect(eq(zero.unknown_required_bits, 0u));
        expect(eq(zero.missing_required_bits, 0u));

        constexpr auto exact = check_spirv_target_feature_requirements(
            target_feature::known_mask,
            target_feature::known_mask);
        static_assert(static_cast<bool>(exact));
        expect(static_cast<bool>(exact));

        constexpr auto required =
            target_feature::shader_float16 |
            target_feature::ray_query;
        constexpr auto superset = check_spirv_target_feature_requirements(
            required, target_feature::known_mask);
        static_assert(static_cast<bool>(superset));
        expect(static_cast<bool>(superset));
    };

    "spirv_target_features_round_trip_every_persistent_bit"_test = [] {
        using namespace lc::spirv;
        constexpr auto all = SpirvTargetFeatures::from_enabled_mask(
            target_feature::known_mask);
        static_assert(all.enabled_mask() == target_feature::known_mask);
        expect(eq(all.enabled_mask(), target_feature::known_mask));

        for (auto feature : spirv_target_feature_descriptions) {
            auto round_trip = SpirvTargetFeatures::from_enabled_mask(
                                  feature.bit)
                                  .enabled_mask();
            expect(eq(round_trip, feature.bit))
                << "target feature did not round-trip exactly: "
                << feature.name;
        }
    };

    "spirv_target_feature_requirements_report_each_missing_bit"_test = [] {
        using namespace lc::spirv;
        for (auto feature : spirv_target_feature_descriptions) {
            auto check = check_spirv_target_feature_requirements(
                target_feature::known_mask,
                target_feature::known_mask & ~feature.bit);
            expect(!static_cast<bool>(check))
                << "missing feature accepted: " << feature.name;
            expect(eq(check.unknown_required_bits, 0u));
            expect(eq(check.missing_required_bits, feature.bit))
                << "missing feature was not reported exactly: "
                << feature.name;
        }
    };

    "spirv_target_feature_requirements_report_multiple_missing_bits"_test = [] {
        using namespace lc::spirv;
        constexpr auto missing =
            target_feature::shader_buffer_int64_atomics |
            target_feature::runtime_descriptor_array |
            target_feature::descriptor_binding_storage_buffer_update_after_bind;
        constexpr auto check = check_spirv_target_feature_requirements(
            target_feature::known_mask,
            target_feature::known_mask & ~missing);
        static_assert(!static_cast<bool>(check));
        static_assert(check.unknown_required_bits == 0u);
        static_assert(check.missing_required_bits == missing);
        expect(!static_cast<bool>(check));
        expect(eq(check.unknown_required_bits, 0u));
        expect(eq(check.missing_required_bits, missing));
    };

    "spirv_target_feature_requirements_reject_unknown_even_if_enabled"_test = [] {
        using namespace lc::spirv;
        constexpr auto unknown = 1ull << 63u;
        constexpr auto check = check_spirv_target_feature_requirements(
            target_feature::ray_query | unknown,
            target_feature::ray_query | unknown);
        static_assert(!static_cast<bool>(check));
        static_assert(check.unknown_required_bits == unknown);
        static_assert(check.missing_required_bits == 0u);
        expect(!static_cast<bool>(check));
        expect(eq(check.unknown_required_bits, unknown));
        expect(eq(check.missing_required_bits, 0u));

        constexpr auto unknown_and_missing =
            check_spirv_target_feature_requirements(
                target_feature::shader_int64 | unknown,
                unknown);
        static_assert(!static_cast<bool>(unknown_and_missing));
        expect(eq(unknown_and_missing.unknown_required_bits, unknown));
        expect(eq(unknown_and_missing.missing_required_bits,
                  target_feature::shader_int64));
    };

    return 0;
}
