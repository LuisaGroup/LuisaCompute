// Test for the Vulkan backend's optional device-memory policy extensions.
// This test covers the host-visible planning helpers of:
// - VK_EXT_zero_initialize_device_memory (allocate flag decision + memory type
//   selection)
// - VK_EXT_memory_decompression (direct and indirect decompression validation)
// - VK_EXT_image_tiling_control (tiling-policy create-info planning)
// - VK_EXT_texture_compression_astc_3d (2D ASTC footprint -> 3D ASTC format)
// - VK_KHR_extended_flags (capability-only invariant over the backend's flag
//   constants)
//
// Every planner is exercised for the enabled and for the
// disabled/unsupported/fallback case: the backend must never make correctness
// depend on an optional extension, so the "extension not enabled" answer has to
// select the pre-existing path.

#include "ut/ut.hpp"
#include "memory_policy_plan.h"
#include <bit>
#include <cstdint>
#include <iterator>
#include <string_view>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;
using luisa::compute::PixelFormat;
namespace vk_ext_memory_test_detail {

using luisa::compute::PixelFormat;

// A representative VkPhysicalDeviceMemoryProperties shape: two heaps, the first
// with one device-local and one device-local + host-visible type, the second
// host-only, plus a protected and a lazily allocated device-local type.
inline constexpr MemoryTypeCandidate memory_types[] = {
    // 0: device local
    {.property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .heap_flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT},
    // 1: device local + host visible + coherent (resizable BAR)
    {.property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
     .heap_flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT},
    // 2: host visible + coherent (system memory)
    {.property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
     .heap_flags = 0u},
    // 3: protected device local
    {.property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_PROTECTED_BIT,
     .heap_flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT},
    // 4: lazily allocated device local
    {.property_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
     .heap_flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT},
    // 5: host cached readback memory
    {.property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
     .heap_flags = 0u},
};
inline constexpr uint32_t memory_type_count =
    static_cast<uint32_t>(std::size(memory_types));

[[nodiscard]] constexpr uint32_t memory_type_flags(uint32_t index) noexcept {
    return index < memory_type_count ? memory_types[index].property_flags : 0u;
}

// Feeds one memory type index through the zero-initialize planner.
[[nodiscard]] constexpr ZeroInitializeAllocationPlan zero_init_for_type(
    bool extension_enabled, bool requested, uint32_t index) noexcept {
    return plan_zero_initialize_allocation(
        extension_enabled, requested, false, index, memory_type_count,
        memory_type_flags(index));
}

}// namespace vk_ext_memory_test_detail

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    // -----------------------------------------------------------------------
    // VK_EXT_zero_initialize_device_memory
    // -----------------------------------------------------------------------
    "zero_initialize_request_matrix"_test = [] {
        using enum ZeroInitializeAllocationDecision;
        // Not requested: the pre-existing undefined-content allocation, even on
        // a device that could zero the memory.
        expect(plan_zero_initialize_allocation(true, false, false, 0u, 6u, 0u).decision == kNotRequested);
        expect(plan_zero_initialize_allocation(false, false, false, 0u, 6u, 0u).decision == kNotRequested);
        expect(!plan_zero_initialize_allocation(true, false, false, 0u, 6u, 0u).use_flag());
        // Requested on an enabled device with a plain device-local type.
        auto enabled = plan_zero_initialize_allocation(true, true, false, 0u, 6u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        expect(enabled.decision == kDriverZeroInitialize);
        expect(enabled.use_flag());
        expect(!enabled.is_fallback());
        expect(static_cast<bool>(enabled));
        // Requested, but the extension is unsupported, switched off through
        // VulkanDeviceConfigExt, or forced off for an imported logical device:
        // the fallback is exactly the pre-existing allocation.
        auto disabled = plan_zero_initialize_allocation(false, true, false, 0u, 6u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        expect(disabled.decision == kDisabledByConfig);
        expect(disabled.is_fallback());
        expect(!static_cast<bool>(disabled));
        // A memory import operation may never request the flag
        // (VUID-VkMemoryAllocateFlagsInfo-flags-10760).
        expect(plan_zero_initialize_allocation(true, true, true, 0u, 6u, 0u).decision == kMemoryImportForbidden);
        // Out-of-range memory type index.
        expect(plan_zero_initialize_allocation(true, true, false, 6u, 6u, 0u).decision == kInvalidMemoryType);
        expect(plan_zero_initialize_allocation(true, true, false, 99u, 6u, 0u).decision == kInvalidMemoryType);
        // Protected memory (VUID-VkMemoryAllocateFlagsInfo-flags-10761).
        expect(plan_zero_initialize_allocation(true, true, false, 3u, 6u, vk_ext_memory_test_detail::memory_type_flags(3u)).decision == kProtectedMemoryType);
        // Lazily allocated memory is not backed by memory before it is bound to
        // a framebuffer, so it can never be zeroed up front.
        expect(plan_zero_initialize_allocation(true, true, false, 4u, 6u, vk_ext_memory_test_detail::memory_type_flags(4u)).decision == kLazilyAllocatedMemoryType);
        // Host-visible and cached types are perfectly zero-initializable.
        expect(vk_ext_memory_test_detail::zero_init_for_type(true, true, 2u).use_flag());
        expect(vk_ext_memory_test_detail::zero_init_for_type(true, true, 5u).use_flag());
        // Only the "driver zero initialize" answer ever sets the flag.
        for (auto i = 0u; i < vk_ext_memory_test_detail::memory_type_count; i++) {
            auto plan = vk_ext_memory_test_detail::zero_init_for_type(true, true, i);
            expect(plan.use_flag() == (plan.decision == kDriverZeroInitialize));
            expect(plan.use_flag() == (i != 3u && i != 4u));
            // Disabled always falls back, whatever the memory type is.
            expect(!vk_ext_memory_test_detail::zero_init_for_type(false, true, i).use_flag());
        }
        // The decision names are stable because they are logged.
        expect(std::string_view{zero_initialize_decision_name(kDriverZeroInitialize)} == "driver-zero-initialize");
        expect(std::string_view{zero_initialize_decision_name(kDisabledByConfig)} == "extension-disabled");
    };

    "memory_type_selection"_test = [] {
        namespace d = vk_ext_memory_test_detail;
        using enum MemoryTypeSelectionStatus;
        constexpr auto all_types = 0x3fu;
        constexpr auto forbidden = static_cast<uint32_t>(
            VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
        // Device-local preference picks the first device-local type.
        auto device_local = select_memory_type(
            all_types, d::memory_type_count, d::memory_types,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0u, forbidden);
        expect(static_cast<bool>(device_local));
        expect(device_local.memory_type_index == 0u);
        // A preferred host-visible property skips the plain device-local type.
        auto host_visible = select_memory_type(
            all_types, d::memory_type_count, d::memory_types,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            forbidden);
        expect(static_cast<bool>(host_visible));
        expect(host_visible.memory_type_index == 1u);
        // The resource's memoryTypeBits mask is honoured.
        auto masked = select_memory_type(
            0x4u, d::memory_type_count, d::memory_types, 0u, 0u, forbidden);
        expect(static_cast<bool>(masked));
        expect(masked.memory_type_index == 2u);
        // A mask that names no existing memory type.
        expect(select_memory_type(0x0u, d::memory_type_count, d::memory_types, 0u, 0u, forbidden).status == kNoMemoryTypeInBits);
        expect(select_memory_type(0x40u, d::memory_type_count, d::memory_types, 0u, 0u, forbidden).status == kNoMemoryTypeInBits);
        // Only forbidden candidates (protected + lazily allocated).
        expect(select_memory_type(0x18u, d::memory_type_count, d::memory_types, 0u, 0u, forbidden).status == kAllCandidatesForbidden);
        // Required host-cached properties that no device-local type provides.
        expect(select_memory_type(0x3u, d::memory_type_count, d::memory_types, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0u, forbidden).status == kMissingRequiredProperties);
        // Degenerate inputs never return a usable type.
        expect(select_memory_type(all_types, d::memory_type_count, nullptr, 0u, 0u, forbidden).status == kNoMemoryTypeInBits);
        expect(select_memory_type(all_types, 0u, d::memory_types, 0u, 0u, forbidden).status == kNoMemoryTypeInBits);
        expect(select_memory_type(all_types, 33u, d::memory_types, 0u, 0u, forbidden).status == kTooManyMemoryTypes);
        expect(!select_memory_type(0x0u, d::memory_type_count, d::memory_types, 0u, 0u, forbidden));
        // Status names are stable because they are reported through LUISA_ERROR.
        expect(std::string_view{memory_type_selection_status_name(kTooManyMemoryTypes)} == "too-many-memory-types");
    };

    // -----------------------------------------------------------------------
    // VK_EXT_memory_decompression
    // -----------------------------------------------------------------------
    "decompression_region_validation"_test = [] {
        using enum DecompressionValidationStatus;
        constexpr auto gdeflate = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT;
        constexpr DecompressionRegion good{
            .src_address = 0x1000u,
            .dst_address = 0x2000u,
            .compressed_size = 0x100u,
            .decompressed_size = 0x1000u};
        expect(validate_decompression_region(good, gdeflate) == kOk);
        // 4-byte alignment of both addresses.
        expect(validate_decompression_region({.src_address = 0x1001u, .dst_address = 0x2000u, .compressed_size = 4u, .decompressed_size = 4u}, gdeflate) == kSourceNotAligned);
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x2002u, .compressed_size = 4u, .decompressed_size = 4u}, gdeflate) == kDestinationNotAligned);
        // Empty ranges are invalid.
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 0u, .decompressed_size = 4u}, gdeflate) == kEmptySource);
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 4u, .decompressed_size = 0u}, gdeflate) == kEmptyDestination);
        // GDeflate decompresses at most 64 KiB per region.
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 4u, .decompressed_size = kGDeflateMaxDecompressedSize}, gdeflate) == kOk);
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 4u, .decompressed_size = kGDeflateMaxDecompressedSize + 4u}, gdeflate) == kGDeflateRegionTooLarge);
        // The same range is rejected before the generic overlap test.
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x1000u, .compressed_size = 4u, .decompressed_size = 4u}, gdeflate) == kSourceEqualsDestination);
        // Overlapping source and destination ranges.
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x1004u, .compressed_size = 0x10u, .decompressed_size = 0x10u}, gdeflate) == kRangesOverlap);
        // Address arithmetic overflow.
        expect(validate_decompression_region({.src_address = UINT64_MAX - 3u, .dst_address = 0x2000u, .compressed_size = 0x10u, .decompressed_size = 4u}, gdeflate) == kAddressOverflow);
        // A non-GDeflate method is not subject to the 64 KiB cap.
        expect(validate_decompression_region({.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 4u, .decompressed_size = 0x20000u}, 0u) == kOk);
    };

    "decompression_request_planning"_test = [] {
        using enum DecompressionValidationStatus;
        constexpr auto gdeflate = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT;
        constexpr auto supported = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT;
        constexpr DecompressionRegion regions[] = {
            {.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
            {.src_address = 0x3000u, .dst_address = 0x4000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
        };
        // Enabled: two independent regions are accepted.
        auto ok = plan_decompression(true, gdeflate, supported, regions, 2u);
        expect(static_cast<bool>(ok));
        expect(ok.status == kOk);
        // Disabled/unsupported extension: fail closed, never silently no-op.
        auto disabled = plan_decompression(false, gdeflate, supported, regions, 2u);
        expect(!static_cast<bool>(disabled));
        expect(disabled.status == kExtensionDisabled);
        // Exactly one method bit (VUID-...-decompressionMethod-07690).
        expect(plan_decompression(true, 0u, supported, regions, 1u).status == kMethodNotSingleBit);
        expect(plan_decompression(true, supported | 0x2u, supported | 0x2u, regions, 1u).status == kMethodNotSingleBit);
        // Method bit the device does not report.
        expect(plan_decompression(true, 0x2u, supported, regions, 1u).status == kMethodUnsupported);
        // No regions at all.
        expect(plan_decompression(true, gdeflate, supported, nullptr, 0u).status == kNoRegions);
        expect(plan_decompression(true, gdeflate, supported, regions, 0u).status == kNoRegions);
        expect(plan_decompression(true, gdeflate, supported, nullptr, 2u).status == kNoRegions);
        // Region-local failure reports its index.
        constexpr DecompressionRegion misaligned[] = {
            regions[0],
            {.src_address = 0x3002u, .dst_address = 0x4000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
        };
        auto region_failure = plan_decompression(true, gdeflate, supported, misaligned, 2u);
        expect(region_failure.status == kSourceNotAligned);
        expect(region_failure.region_index == 1u);
        // Overlapping destinations across regions.
        constexpr DecompressionRegion overlapping_dst[] = {
            {.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
            {.src_address = 0x3000u, .dst_address = 0x2800u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
        };
        expect(plan_decompression(true, gdeflate, supported, overlapping_dst, 2u).status == kRangesOverlap);
        // Region 1's source inside region 0's destination.
        constexpr DecompressionRegion source_in_other_dst[] = {
            {.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 0x100u, .decompressed_size = 0x2000u},
            {.src_address = 0x2400u, .dst_address = 0x4000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
        };
        expect(plan_decompression(true, gdeflate, supported, source_in_other_dst, 2u).status == kRangesOverlap);
        // Two regions may share the same source (read-only aliasing).
        constexpr DecompressionRegion shared_source[] = {
            {.src_address = 0x1000u, .dst_address = 0x2000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
            {.src_address = 0x1000u, .dst_address = 0x4000u, .compressed_size = 0x100u, .decompressed_size = 0x1000u},
        };
        expect(static_cast<bool>(plan_decompression(true, gdeflate, supported, shared_source, 2u)));
    };

    "decompression_indirect_planning"_test = [] {
        using enum DecompressionValidationStatus;
        constexpr auto gdeflate = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT;
        constexpr auto supported = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT;
        // The region stride is the packed VkDecompressMemoryRegionEXT size.
        expect(kDecompressionIndirectStride == sizeof(VkDecompressMemoryRegionEXT));
        expect(static_cast<bool>(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride)));
        // Disabled extension fails closed.
        expect(plan_decompression_indirect(false, gdeflate, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride).status == kExtensionDisabled);
        // Method checks are shared with the direct path.
        expect(plan_decompression_indirect(true, 0u, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride).status == kMethodNotSingleBit);
        expect(plan_decompression_indirect(true, 0x2u, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride).status == kMethodUnsupported);
        // Both indirect addresses must be 4-byte aligned
        // (VUID-vkCmdDecompressMemoryIndirectCountEXT-indirectCommandsAddress-07695).
        expect(plan_decompression_indirect(true, gdeflate, supported, 0x1001u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride).status == kSourceNotAligned);
        expect(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2002u, 16u, 1024u, kDecompressionIndirectStride).status == kDestinationNotAligned);
        // VUID-...-stride-11767: multiple of 4 and at least the region size.
        expect(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride - 4u).status == kIndirectStrideInvalid);
        expect(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride + 2u).status == kIndirectStrideInvalid);
        expect(static_cast<bool>(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2000u, 16u, 1024u, kDecompressionIndirectStride + 4u)));
        // VUID-...-maxDecompressionCount-11768 and a non-zero count.
        expect(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2000u, 1025u, 1024u, kDecompressionIndirectStride).status == kIndirectCountTooLarge);
        expect(plan_decompression_indirect(true, gdeflate, supported, 0x1000u, 0x2000u, 0u, 1024u, kDecompressionIndirectStride).status == kNoRegions);
    };

    "decompression_method_helpers"_test = [] {
        expect(decompression_method_bit_count(0u) == 0u);
        expect(decompression_method_bit_count(VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT) == 1u);
        expect(decompression_method_bit_count(static_cast<VkMemoryDecompressionMethodFlagsEXT>(
                   VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT | 0x2u)) == 2u);
        expect(std::string_view{decompression_validation_status_name(DecompressionValidationStatus::kExtensionDisabled)} == "extension-disabled");
    };

    // -----------------------------------------------------------------------
    // VK_EXT_image_tiling_control
    // -----------------------------------------------------------------------
    "image_tiling_control_planning"_test = [] {
        // The default policy is the pre-existing behaviour: the create info is
        // never chained, exactly like omitting the structure.
        for (auto enabled : {false, true}) {
            auto plan = plan_image_tiling_control(enabled, ImageTilingPolicy::kDefault, VK_IMAGE_TILING_OPTIMAL);
            expect(!plan.chain_create_info);
            expect(!plan.applied());
            expect(plan.tiling_control == VK_IMAGE_TILING_CONTROL_DEFAULT_EXT);
        }
        // Disabled/unsupported extension: the alternative policies degrade to
        // the pre-existing default tiling instead of failing.
        for (auto policy : {ImageTilingPolicy::kMinimizeMemory, ImageTilingPolicy::kMaximizePerformance}) {
            auto plan = plan_image_tiling_control(false, policy, VK_IMAGE_TILING_OPTIMAL);
            expect(!plan.chain_create_info);
            expect(plan.tiling_control == VK_IMAGE_TILING_CONTROL_DEFAULT_EXT);
            // VUID-VkImageTilingControlCreateInfoEXT-tilingControl-12479: a
            // linear image can never carry a non-default policy.
            auto linear = plan_image_tiling_control(true, policy, VK_IMAGE_TILING_LINEAR);
            expect(!linear.chain_create_info);
            expect(linear.tiling_control == VK_IMAGE_TILING_CONTROL_DEFAULT_EXT);
        }
        // Enabled: the two selectable policies map onto the spec enumerants.
        auto min_size = plan_image_tiling_control(true, ImageTilingPolicy::kMinimizeMemory, VK_IMAGE_TILING_OPTIMAL);
        expect(min_size.chain_create_info);
        expect(min_size.applied());
        expect(min_size.tiling_control == VK_IMAGE_TILING_CONTROL_MIN_SIZE_EXT);
        auto max_perf = plan_image_tiling_control(true, ImageTilingPolicy::kMaximizePerformance, VK_IMAGE_TILING_OPTIMAL);
        expect(max_perf.chain_create_info);
        expect(max_perf.tiling_control == VK_IMAGE_TILING_CONTROL_MAX_PERFORMANCE_EXT);
        // Policy names are stable because they are logged.
        expect(std::string_view{image_tiling_policy_name(ImageTilingPolicy::kMinimizeMemory)} == "minimize-memory");
        expect(std::string_view{image_tiling_policy_name(ImageTilingPolicy::kMaximizePerformance)} == "maximize-performance");
    };

    // -----------------------------------------------------------------------
    // VK_EXT_texture_compression_astc_3d
    // -----------------------------------------------------------------------
    "astc_block_footprints"_test = [] {
        using enum AstcBlockFootprint;
        // Every 2D ASTC PixelFormat carries its block footprint.
        expect(astc_block_footprint(PixelFormat::ASTC_4x4) == k4x4);
        expect(astc_block_footprint(PixelFormat::ASTC_4x4_SRGB) == k4x4);
        expect(astc_block_footprint(PixelFormat::ASTC_5x5) == k5x5);
        expect(astc_block_footprint(PixelFormat::ASTC_5x5_SRGB) == k5x5);
        expect(astc_block_footprint(PixelFormat::ASTC_6x6) == k6x6);
        expect(astc_block_footprint(PixelFormat::ASTC_6x6_SRGB) == k6x6);
        expect(astc_block_footprint(PixelFormat::ASTC_8x8) == k8x8);
        expect(astc_block_footprint(PixelFormat::ASTC_10x10_SRGB) == k10x10);
        expect(astc_block_footprint(PixelFormat::ASTC_12x12) == k12x12);
        expect(astc_block_footprint(PixelFormat::ASTC_12x12_SRGB) == k12x12);
        // Non-ASTC formats never take the 3D ASTC path.
        for (auto format : {PixelFormat::RGBA8UNorm, PixelFormat::BC1UNorm, PixelFormat::R32F, PixelFormat::RGBA8SRGB}) {
            expect(astc_block_footprint(format) == kNone);
        }
    };

    "astc_3d_format_planning"_test = [] {
        using enum AstcBlockFootprint;
        using enum Astc3DFormatStatus;
        // A non-ASTC format keeps the plain 2D mapping.
        expect(plan_astc_3d_format(true, kNone, false, 3u).status == kNotAstcBlockFormat);
        expect(!plan_astc_3d_format(true, kNone, false, 3u).is_astc_block_format());
        // ASTC 2D images are unaffected by the 3D extension.
        for (auto footprint : {k4x4, k5x5, k6x6, k8x8, k10x10, k12x12}) {
            for (auto dimension : {1u, 2u}) {
                auto plan = plan_astc_3d_format(true, footprint, false, dimension);
                expect(plan.status == kNotAstcBlockFormat);
                expect(plan.format == VK_FORMAT_UNDEFINED);
            }
        }
        // Extension disabled: a 3D image requested with a 2D ASTC block format
        // keeps the pre-existing "unsupported format" failure.
        for (auto footprint : {k4x4, k5x5, k6x6, k8x8, k10x10, k12x12}) {
            for (auto srgb : {false, true}) {
                auto plan = plan_astc_3d_format(false, footprint, srgb, 3u);
                expect(plan.is_astc_block_format());
                expect(plan.status == kExtensionDisabled);
                expect(plan.format == VK_FORMAT_UNDEFINED);
                expect(!static_cast<bool>(plan));
            }
        }
        // Extension enabled: the three block footprints with a 3D twin map to
        // the equally sized 3D-compressed ASTC format.
        auto astc4 = plan_astc_3d_format(true, k4x4, false, 3u);
        expect(astc4.status == kAstc3DFormat);
        expect(astc4.format == VK_FORMAT_ASTC_4x4x4_UNORM_BLOCK_EXT);
        expect(static_cast<bool>(astc4));
        expect(plan_astc_3d_format(true, k4x4, true, 3u).format == VK_FORMAT_ASTC_4x4x4_SRGB_BLOCK_EXT);
        expect(plan_astc_3d_format(true, k5x5, false, 3u).format == VK_FORMAT_ASTC_5x5x5_UNORM_BLOCK_EXT);
        expect(plan_astc_3d_format(true, k5x5, true, 3u).format == VK_FORMAT_ASTC_5x5x5_SRGB_BLOCK_EXT);
        expect(plan_astc_3d_format(true, k6x6, false, 3u).format == VK_FORMAT_ASTC_6x6x6_UNORM_BLOCK_EXT);
        expect(plan_astc_3d_format(true, k6x6, true, 3u).format == VK_FORMAT_ASTC_6x6x6_SRGB_BLOCK_EXT);
        // 8x8, 10x10 and 12x12 have no 3D compressed counterpart, so they keep
        // failing even when the extension is enabled.
        for (auto footprint : {k8x8, k10x10, k12x12}) {
            for (auto srgb : {false, true}) {
                auto plan = plan_astc_3d_format(true, footprint, srgb, 3u);
                expect(plan.status == kNoAstc3DCounterpart);
                expect(plan.format == VK_FORMAT_UNDEFINED);
                expect(!static_cast<bool>(plan));
            }
        }
        // Status names are stable because they are reported through LUISA_ERROR.
        expect(std::string_view{astc_3d_format_status_name(kNoAstc3DCounterpart)} == "no-astc-3d-counterpart");
    };

    // -----------------------------------------------------------------------
    // VK_KHR_extended_flags (capability-only)
    // -----------------------------------------------------------------------
    "extended_flags_invariant"_test = [] {
        // VK_KHR_extended_flags only widens flag *types* and defines no new
        // bits. The backend therefore keeps 32-bit flag plumbing, which is only
        // valid while every bit it composes fits into 32 bits.
        constexpr auto fits_32_bits = [](uint64_t value) constexpr noexcept {
            return std::bit_width(value) <= 32;
        };
        // Image create flags the backend composes.
        for (auto flag : {uint64_t{VK_IMAGE_CREATE_SPARSE_BINDING_BIT},
                          uint64_t{VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT},
                          uint64_t{VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
                          uint64_t{VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT},
                          uint64_t{VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT},
                          uint64_t{VK_IMAGE_CREATE_ALIAS_BIT}}) {
            expect(fits_32_bits(flag));
        }
        // Image usage flags the backend composes.
        for (auto flag : {uint64_t{VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
                          uint64_t{VK_IMAGE_USAGE_TRANSFER_DST_BIT},
                          uint64_t{VK_IMAGE_USAGE_SAMPLED_BIT},
                          uint64_t{VK_IMAGE_USAGE_STORAGE_BIT},
                          uint64_t{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
                          uint64_t{VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT},
                          uint64_t{VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT}}) {
            expect(fits_32_bits(flag));
        }
        // Buffer usage flags the backend composes, and the memory allocation
        // flag the zero-initialize path adds.
        for (auto flag : {uint64_t{VK_BUFFER_USAGE_TRANSFER_SRC_BIT},
                          uint64_t{VK_BUFFER_USAGE_TRANSFER_DST_BIT},
                          uint64_t{VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT},
                          uint64_t{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
                          uint64_t{VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
                          uint64_t{VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT},
                          uint64_t{VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT},
                          uint64_t{VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT}}) {
            expect(fits_32_bits(flag));
        }
        // The single exception the backend needs: the memory decompression
        // buffer usage bit lives above bit 31, so it must travel through
        // VkBufferUsageFlags2CreateInfoKHR instead of widening
        // VkBufferUsageFlags. Reading the flag must therefore require either
        // Vulkan 1.4 or VK_KHR_maintenance5.
        expect(!fits_32_bits(uint64_t{VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT}));
        expect(uint64_t{VK_BUFFER_USAGE_2_MEMORY_DECOMPRESSION_BIT_EXT} == (1ull << 32u));
        expect(!plan_buffer_usage_flags2(VK_API_VERSION_1_3, false));
        expect(plan_buffer_usage_flags2(VK_API_VERSION_1_3, true));
        expect(plan_buffer_usage_flags2(VK_API_VERSION_1_4, false));
        expect(plan_buffer_usage_flags2(VK_API_VERSION_1_4, true));
        // Pipeline stage/access flags are already 64-bit types, so the
        // decompression stage and access bits need no widening either.
        expect(std::bit_width(uint64_t{VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT}) > 32);
        expect(std::bit_width(uint64_t{VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT}) > 32);
        expect(std::bit_width(uint64_t{VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT}) > 32);
        static_assert(sizeof(VkPipelineStageFlags2) == sizeof(uint64_t));
        static_assert(sizeof(VkAccessFlags2) == sizeof(uint64_t));
        static_assert(sizeof(VkBufferUsageFlags) == sizeof(uint32_t));
    };

    // -----------------------------------------------------------------------
    // Everything above must be usable at compile time, which keeps the planners
    // free of device state (and of any dependency on the extension being
    // present at runtime).
    // -----------------------------------------------------------------------
    static_assert(plan_zero_initialize_allocation(true, true, false, 0u, 1u, 0u).use_flag());
    static_assert(!plan_zero_initialize_allocation(false, true, false, 0u, 1u, 0u).use_flag());
    static_assert(!plan_zero_initialize_allocation(true, true, true, 0u, 1u, 0u).use_flag());
    static_assert(plan_image_tiling_control(true, ImageTilingPolicy::kMinimizeMemory, VK_IMAGE_TILING_OPTIMAL).chain_create_info);
    static_assert(!plan_image_tiling_control(false, ImageTilingPolicy::kMinimizeMemory, VK_IMAGE_TILING_OPTIMAL).chain_create_info);
    static_assert(plan_astc_3d_format(true, AstcBlockFootprint::k4x4, false, 3u).format == VK_FORMAT_ASTC_4x4x4_UNORM_BLOCK_EXT);
    static_assert(!plan_astc_3d_format(false, AstcBlockFootprint::k4x4, false, 3u));
    static_assert(plan_decompression_indirect(true, VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT,
                                              VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT,
                                              0x1000u, 0x2000u, 4u, 8u,
                                              kDecompressionIndirectStride));
    static_assert(!plan_decompression(false, VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT,
                                      VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT,
                                      nullptr, 0u));

    return 0;
}
