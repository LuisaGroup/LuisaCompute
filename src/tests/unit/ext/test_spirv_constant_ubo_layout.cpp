// Tests for the portable XIR-to-SPIR-V constant-UBO layout planner.
// Covers exact range admission, one-stride overflow fallback, std140 padding,
// and checked host-size arithmetic without requiring a Vulkan device.

#include "ut/ut.hpp"

#include <limits>

#include <luisa/core/basic_types.h>

#include "spirv_codegen/constant_ubo_layout.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_constant_ubo_accepts_exact_portable_limit"_test = [] {
        using namespace lc::spirv;
        constexpr auto layout = plan_constant_ubo_member(
            0u, 4u, 4u, 1024u);
        expect(static_cast<bool>(layout));
        expect(layout.status == ConstantUBOLayoutStatus::SUCCESS);
        expect(eq(layout.member_offset, 0u));
        expect(eq(layout.array_stride, 16u));
        expect(eq(layout.end_offset,
                  portable_constant_ubo_max_range));
    };

    "spirv_constant_ubo_rejects_one_stride_over_limit"_test = [] {
        using namespace lc::spirv;
        ConstantUBOLayoutPlanner planner;
        auto over_limit = planner.try_append(4u, 4u, 1025u);
        expect(!static_cast<bool>(over_limit));
        expect(over_limit.status ==
               ConstantUBOLayoutStatus::RANGE_EXCEEDED);
        expect(eq(planner.size_bytes(), 0u));

        // A rejected member does not modify planner state. The caller can
        // leave it as OpConstant and still admit a later, smaller member.
        auto later_member = planner.try_append(4u, 4u, 1u);
        expect(static_cast<bool>(later_member));
        expect(eq(later_member.member_offset, 0u));
        expect(eq(later_member.end_offset, 16u));
        expect(eq(planner.size_bytes(), 16u));
    };

    "spirv_constant_ubo_checks_std140_prefix_and_stride"_test = [] {
        using namespace lc::spirv;
        constexpr auto first = plan_constant_ubo_member(
            0u, 4u, 4u, 1u, 128u);
        constexpr auto aligned = plan_constant_ubo_member(
            first.end_offset, 32u, 24u, 2u, 128u);
        expect(static_cast<bool>(first));
        expect(static_cast<bool>(aligned));
        expect(eq(aligned.member_offset, 32u));
        expect(eq(aligned.array_stride, 32u));
        expect(eq(aligned.end_offset, 96u));
    };

    "spirv_constant_ubo_rejects_invalid_and_overflowed_layouts"_test = [] {
        using namespace lc::spirv;
        constexpr auto invalid_alignment = plan_constant_ubo_member(
            0u, 3u, 4u, 1u);
        expect(invalid_alignment.status ==
               ConstantUBOLayoutStatus::INVALID_LAYOUT);

        constexpr auto max_size =
            std::numeric_limits<size_t>::max();
        constexpr auto offset_overflow = plan_constant_ubo_member(
            max_size - 7u, 16u, 16u, 1u, max_size);
        expect(offset_overflow.status ==
               ConstantUBOLayoutStatus::ARITHMETIC_OVERFLOW);

        constexpr auto stride_overflow = plan_constant_ubo_member(
            0u, 16u, max_size - 7u, 1u, max_size);
        expect(stride_overflow.status ==
               ConstantUBOLayoutStatus::ARITHMETIC_OVERFLOW);

        constexpr auto product_overflow = plan_constant_ubo_member(
            0u, 16u, 16u, max_size / 16u + 1u, max_size);
        expect(product_overflow.status ==
               ConstantUBOLayoutStatus::ARITHMETIC_OVERFLOW);

        constexpr auto end_overflow = plan_constant_ubo_member(
            max_size - 15u, 16u, 16u, 1u, max_size);
        expect(end_overflow.status ==
               ConstantUBOLayoutStatus::ARITHMETIC_OVERFLOW);
    };
}
