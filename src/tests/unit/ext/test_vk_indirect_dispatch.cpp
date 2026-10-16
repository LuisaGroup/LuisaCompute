#include "ut/ut.hpp"

#include "indirect_dispatch_layout.h"
#include "vulkan_builtin_contract.h"

#include <cstddef>
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

    "vk_builtin_pipeline_contract_is_explicit"_test = [] {
        using lc::hlsl::ShaderVariableType;
        using lc::vk::detail::VulkanBuiltinKernel;
        using lc::vk::detail::vulkan_builtin_buffer_properties;
        using lc::vk::detail::vulkan_builtin_kernel_contract;

        expect(eq(vulkan_builtin_buffer_properties.size(), 3u));
        expect(vulkan_builtin_buffer_properties[0].type ==
               ShaderVariableType::StructuredBuffer);
        expect(eq(vulkan_builtin_buffer_properties[0].space_index, 0u));
        expect(eq(vulkan_builtin_buffer_properties[0].register_index, 0u));
        expect(vulkan_builtin_buffer_properties[1].type ==
               ShaderVariableType::RWStructuredBuffer);
        expect(eq(vulkan_builtin_buffer_properties[1].space_index, 0u));
        expect(eq(vulkan_builtin_buffer_properties[1].register_index, 1u));
        expect(vulkan_builtin_buffer_properties[2].type ==
               ShaderVariableType::SamplerHeap);
        expect(eq(vulkan_builtin_buffer_properties[2].space_index, 1u));
        expect(eq(vulkan_builtin_buffer_properties[2].register_index, 0u));
        expect(eq(vulkan_builtin_buffer_properties[2].array_size, 16u));

        constexpr auto indirect = vulkan_builtin_kernel_contract(
            VulkanBuiltinKernel::INDIRECT_PREPARE);
        constexpr auto accel = vulkan_builtin_kernel_contract(
            VulkanBuiltinKernel::ACCEL_PROCESS);
        constexpr auto bindless = vulkan_builtin_kernel_contract(
            VulkanBuiltinKernel::BINDLESS_UPLOAD);
        expect(eq(indirect.block_size_x, 64u));
        expect(eq(indirect.push_constant_size, 48u));
        expect(eq(accel.block_size_x, 256u));
        expect(eq(accel.push_constant_size, 8u));
        expect(eq(bindless.block_size_x, 256u));
        expect(eq(bindless.push_constant_size, 4u));
    };

    "vk_accel_update_input_has_portable_packed_layout"_test = [] {
        using lc::vk::detail::VulkanAccelUpdateLayout;
        using lc::vk::detail::VulkanAccelUpdateInput;
        expect(eq(sizeof(VulkanAccelUpdateInput), 64u));
        expect(eq(alignof(VulkanAccelUpdateInput), 16u));
        expect(eq(offsetof(VulkanAccelUpdateInput, mesh), 48u));
        expect(eq(
            offsetof(VulkanAccelUpdateInput, index_visibility), 56u));
        expect(eq(offsetof(VulkanAccelUpdateInput, user_id_flags), 60u));
        expect(eq(
            VulkanAccelUpdateInput::pack_index_visibility(
                0x89abcdefu, 0x123u),
            0x23abcdefu));
        expect(eq(
            VulkanAccelUpdateInput::pack_user_id_flags(
                0x76543210u, 0x1abu),
            0xab543210u));
        constexpr auto address =
            VulkanAccelUpdateInput::device_address_words(
                0x0123456789abcdefull);
        expect(eq(address[0], 0x89abcdefu));
        expect(eq(address[1], 0x01234567u));
        expect(eq(
            VulkanAccelUpdateLayout::updated_contribution_offset_flags(
                0x5a123456u,
                VulkanAccelUpdateLayout::flag_transform),
            0x5a000000u));
        expect(eq(
            VulkanAccelUpdateLayout::updated_contribution_offset_flags(
                0x5a123456u,
                VulkanAccelUpdateLayout::flag_opaque_on),
            0x04000000u));
        expect(eq(
            VulkanAccelUpdateLayout::updated_contribution_offset_flags(
                0x5a123456u,
                VulkanAccelUpdateLayout::flag_opaque_off),
            0x08000000u));
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
