#include "ut/ut.hpp"

#include "indirect_dispatch_layout.h"
#include "indirect_prepare_shader.h"

#include <limits>
#include <utility>

using namespace boost::ut;
using namespace boost::ut::literals;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_indirect_source_layout_is_explicit_and_tightly_bounded"_test = [] {
        using lc::IndirectDispatchLayout;
        expect(eq(IndirectDispatchLayout::header_size, 4u));
        expect(eq(IndirectDispatchLayout::record_size, 28u));
        expect(eq(IndirectDispatchLayout::vulkan_command_size, 12u));
        expect(eq(IndirectDispatchLayout::prepare_block_size, 64u));
        expect(eq(IndirectDispatchLayout::record_word_offset(0u), 1u));
        expect(eq(IndirectDispatchLayout::record_word_offset(15u), 106u));

        size_t size = 0u;
        expect(IndirectDispatchLayout::try_total_size(16u, size));
        expect(eq(size, 452u));
        expect(!IndirectDispatchLayout::try_total_size(
            std::numeric_limits<size_t>::max(), size));
    };

    "vk_indirect_prepare_hlsl_layout_tokens_are_decimal"_test = [] {
        auto definitions =
            lc::vk::indirect_prepare_hlsl_layout_definitions();
        constexpr auto expected =
            "#define LC_INDIRECT_HEADER_WORDS 1u\n"
            "#define LC_INDIRECT_RECORD_WORDS 7u\n"
            "#define LC_INDIRECT_LOGICAL_WORD 0u\n"
            "#define LC_INDIRECT_GROUP_WORD 4u\n"
            "#define LC_INDIRECT_COMMAND_WORDS 3u\n"
            "#define LC_INDIRECT_PREPARE_BLOCK_SIZE 64u\n";
        expect(definitions.view() == expected);
    };

    "vk_indirect_host_plan_clamps_maximum_to_capacity"_test = [] {
        auto plan = lc::plan_indirect_dispatch(
            16u, 5u, std::numeric_limits<uint32_t>::max());
        expect(static_cast<bool>(plan));
        expect(eq(plan.plan.source_record_offset, 5u));
        expect(eq(plan.plan.command_count, 11u));
        expect(eq(plan.plan.scratch_size_bytes, 132u));

        auto tail = lc::plan_indirect_dispatch(16u, 16u, 7u);
        expect(static_cast<bool>(tail));
        expect(eq(tail.plan.command_count, 0u));

        auto invalid = lc::plan_indirect_dispatch(16u, 17u, 1u);
        expect(!static_cast<bool>(invalid));
        expect(invalid.error ==
               lc::IndirectDispatchPlanError::OFFSET_OUT_OF_RANGE);
    };

    "vk_indirect_host_plan_distinguishes_empty_tail_and_invalid_ranges"_test = [] {
        auto zero_maximum = lc::plan_indirect_dispatch(4u, 0u, 0u);
        expect(static_cast<bool>(zero_maximum));
        expect(eq(zero_maximum.plan.command_count, 0u));

        auto empty_tail = lc::plan_indirect_dispatch(4u, 4u, 1u);
        expect(static_cast<bool>(empty_tail));
        expect(eq(empty_tail.plan.command_count, 0u));

        auto clamped_tail = lc::plan_indirect_dispatch(4u, 3u, 8u);
        expect(static_cast<bool>(clamped_tail));
        expect(eq(clamped_tail.plan.command_count, 1u));

        auto invalid_offset = lc::plan_indirect_dispatch(4u, 5u, 0u);
        expect(!static_cast<bool>(invalid_offset));
        expect(invalid_offset.error ==
               lc::IndirectDispatchPlanError::OFFSET_OUT_OF_RANGE);
        if constexpr (sizeof(size_t) > sizeof(uint32_t)) {
            auto oversized_capacity = lc::plan_indirect_dispatch(
                static_cast<size_t>(
                    std::numeric_limits<uint32_t>::max()) +
                    1u,
                0u, 1u);
            expect(!static_cast<bool>(oversized_capacity));
            expect(oversized_capacity.error ==
                   lc::IndirectDispatchPlanError::CAPACITY_EXCEEDS_UINT32);
        }
    };

    "vk_indirect_group_arithmetic_is_overflow_safe"_test = [] {
        constexpr auto max = std::numeric_limits<uint32_t>::max();
        constexpr auto rounded =
            lc::indirect_dispatch_group_count(max, 2u);
        static_assert(rounded.valid_block_size);
        expect(eq(rounded.value, 0x80000000u));

        constexpr auto exact =
            lc::indirect_dispatch_group_count(max, max);
        expect(exact.valid_block_size);
        expect(eq(exact.value, 1u));

        constexpr auto empty =
            lc::indirect_dispatch_group_count(0u, 64u);
        expect(empty.valid_block_size);
        expect(eq(empty.value, 0u));

        constexpr auto invalid =
            lc::indirect_dispatch_group_count(17u, 0u);
        expect(!invalid.valid_block_size);
        expect(eq(invalid.value, 0u));
    };

    "vk_indirect_uint32_global_id_limit_is_exact"_test = [] {
        expect(eq(
            lc::indirect_dispatch_max_group_count_for_uint32_global_id(64u),
            67'108'864u));
        expect(eq(
            lc::indirect_dispatch_max_group_count_for_uint32_global_id(1u),
            std::numeric_limits<uint32_t>::max()));
        expect(eq(
            lc::indirect_dispatch_max_group_count_for_uint32_global_id(0u),
            0u));
    };
}
