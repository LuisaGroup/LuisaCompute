// Pure planning tests for the Vulkan backend's Android-compatibility work:
// descriptor-pool sizing on mobile limits, timeline-semaphore "0 = no limit"
// normalization, legacy (Vulkan 1.0/1.1/1.2) barrier/copy mask conversion,
// and ASTC block pixel-storage sizing. These are device-independent and run
// without a Vulkan instance.
#include "ut/ut.hpp"
#include "device_feature_plan.h"
#include "timeline_semaphore_plan.h"
#include "command_buffer_sync.h"
#include <luisa/runtime/rhi/pixel.h>
#include <limits>
using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;
namespace vk_detail = lc::vk::detail;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_descriptor_pool_plan_preserves_desktop_defaults"_test = [] {
        // Desktop-class limits: huge per-stage/per-set budgets. The plan must
        // keep the historical defaults (262144 sets, 65536 per type).
        constexpr auto plan = vk_detail::plan_descriptor_pool_sizes(
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(),
            false);
        expect(eq(plan.max_sets, 262144u));
        expect(eq(plan.storage_buffers, 65536u));
        expect(eq(plan.storage_images, 65536u));
        expect(eq(plan.sampled_images, 65536u));
        expect(eq(plan.samplers, 65536u));
        expect(eq(plan.uniform_buffers, 65536u));
    };

    "vk_descriptor_pool_plan_clamps_to_mobile_limits"_test = [] {
        // Low-end mobile: 16 per-stage and 32 per-set descriptors. Every
        // per-type count and maxSets must be clamped to those budgets instead
        // of reserving the desktop 262144-set pool.
        constexpr auto plan = vk_detail::plan_descriptor_pool_sizes(
            16u, 16u, 16u, 16u, 16u,  // per-stage
            32u, 32u, 32u, 32u, 32u,  // per-set
            false);
        expect(eq(plan.storage_buffers, 32u));
        expect(eq(plan.storage_images, 32u));
        expect(eq(plan.sampled_images, 32u));
        expect(eq(plan.samplers, 32u));
        expect(eq(plan.uniform_buffers, 32u));
        // maxSets is bounded by the total descriptor capacity of the pool.
        expect(plan.max_sets <= 32u * 5u);
        expect(plan.max_sets >= 64u);
        constexpr auto rt_plan = vk_detail::plan_descriptor_pool_sizes(
            16u, 16u, 16u, 16u, 16u,
            32u, 32u, 32u, 32u, 32u,
            true);
        // Ray tracing adds an acceleration-structure pool entry.
        expect(rt_plan.acceleration_structures >= 1u);
        expect(rt_plan.max_sets <= 32u * 6u);
    };

    "vk_descriptor_pool_plan_never_requests_zero_counts"_test = [] {
        // Even a degenerate device that reports 0 per-stage/per-set budgets
        // must produce a usable pool (>= 1 per type, >= 64 sets).
        constexpr auto plan = vk_detail::plan_descriptor_pool_sizes(
            0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u,
            false);
        expect(plan.storage_buffers >= 1u);
        expect(plan.storage_images >= 1u);
        expect(plan.sampled_images >= 1u);
        expect(plan.samplers >= 1u);
        expect(plan.uniform_buffers >= 1u);
        expect(plan.max_sets >= 64u);
    };

    "vk_timeline_plan_treats_zero_max_value_difference_as_no_limit"_test = [] {
        using Status = vk_detail::TimelineSemaphoreValueStatus;
        constexpr auto unlimited = std::numeric_limits<uint64_t>::max();
        // signal: current == tracked, increment stays within the window.
        auto signal = vk_detail::plan_timeline_semaphore_signal(
            10u, 10u, 20u, 0u/*reported limit*/);
        expect(signal.status == Status::SUCCESS);
        // A zero reported limit must behave like an unbounded window: a huge
        // jump is legal.
        auto big_signal = vk_detail::plan_timeline_semaphore_signal(
            0u, 0u, unlimited, 0u);
        expect(big_signal.status == Status::SUCCESS);
        // wait: current (10) < wait (12) <= tracked (15); the zero reported
        // limit means the whole window is legal.
        auto wait = vk_detail::plan_timeline_semaphore_wait(
            10u, 15u, 12u, 0u);
        expect(wait.status == Status::SUCCESS);
        expect(!wait.already_satisfied);
        // A zero limit must also allow an unbounded tracked interval.
        auto big_wait = vk_detail::plan_timeline_semaphore_wait(
            0u, unlimited, unlimited, 0u);
        expect(big_wait.status == Status::SUCCESS);
    };

    "vk_legacy_mask_conversion_round_trips_common_bits"_test = [] {
        constexpr auto access =
            VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT |
            VK_ACCESS_2_TRANSFER_READ_BIT |
            VK_ACCESS_2_HOST_WRITE_BIT |
            VK_ACCESS_2_MEMORY_READ_BIT |
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        auto legacy_access = vk_detail::legacy_access_mask(access);
        expect((legacy_access & VK_ACCESS_SHADER_READ_BIT) != 0u);
        expect((legacy_access & VK_ACCESS_SHADER_WRITE_BIT) != 0u);
        expect((legacy_access & VK_ACCESS_TRANSFER_READ_BIT) != 0u);
        expect((legacy_access & VK_ACCESS_HOST_WRITE_BIT) != 0u);
        expect((legacy_access & VK_ACCESS_MEMORY_READ_BIT) != 0u);
#ifdef VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
        auto rt_access = vk_detail::legacy_access_mask(
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
        expect((rt_access & VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR) != 0u);
        expect((rt_access & VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR) != 0u);
#endif
        constexpr auto stages =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_2_TRANSFER_BIT |
            VK_PIPELINE_STAGE_2_HOST_BIT |
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        auto legacy_stages = vk_detail::legacy_stage_mask(stages);
        expect((legacy_stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0u);
        expect((legacy_stages & VK_PIPELINE_STAGE_TRANSFER_BIT) != 0u);
        expect((legacy_stages & VK_PIPELINE_STAGE_HOST_BIT) != 0u);
        expect((legacy_stages & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) != 0u);
        expect((legacy_stages & VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT) != 0u);
#ifdef VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
        auto rt_stages = vk_detail::legacy_stage_mask(
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
        expect((rt_stages & VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR) != 0u);
        expect((rt_stages & VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR) != 0u);
#endif
    };

    "vk_astc_pixel_storage_uses_format_specific_block_dimensions"_test = [] {
        // ASTC blocks are always 16 bytes but cover different pixel extents.
        // 6x6 image with ASTC 5x5 blocks -> 2x2 blocks -> 4 * 16 = 64 bytes.
        auto astc_5x5 = checked_pixel_storage_size(
            PixelStorage::ASTC_5x5, uint3{6u, 6u, 1u});
        expect(astc_5x5.status == PixelStorageSizeStatus::SUCCESS);
        expect(eq(astc_5x5.size, size_t{64u}));
        // 12x12 image with ASTC 12x12 blocks -> 1 block -> 16 bytes.
        auto astc_12x12 = checked_pixel_storage_size(
            PixelStorage::ASTC_12x12, uint3{12u, 12u, 1u});
        expect(astc_12x12.status == PixelStorageSizeStatus::SUCCESS);
        expect(eq(astc_12x12.size, size_t{16u}));
        // 13x13 image with ASTC 12x12 blocks -> 2x2 blocks -> 64 bytes.
        auto astc_12x12_padded = checked_pixel_storage_size(
            PixelStorage::ASTC_12x12, uint3{13u, 13u, 1u});
        expect(astc_12x12_padded.status == PixelStorageSizeStatus::SUCCESS);
        expect(eq(astc_12x12_padded.size, size_t{64u}));
        // SRGB variant shares the same block layout as its UNORM sibling.
        auto astc_4x4_srgb = checked_pixel_storage_size(
            PixelStorage::ASTC_4x4_SRGB, uint3{4u, 4u, 1u});
        expect(astc_4x4_srgb.status == PixelStorageSizeStatus::SUCCESS);
        expect(eq(astc_4x4_srgb.size, size_t{16u}));
        // Block-compressed formats must be flagged as such and never request
        // STORAGE usage (block_compressed formats are not STORAGE images).
        expect(is_block_compressed(PixelFormat::ASTC_8x8));
        expect(is_block_compressed(PixelStorage::ASTC_8x8_SRGB));
        expect(!is_srgb(PixelFormat::ASTC_8x8));
        expect(is_srgb(PixelFormat::ASTC_8x8_SRGB));
    };
}
