#include "ut/ut.hpp"

#include "resource_barrier_contract.h"

#include <type_traits>

using namespace boost::ut;
using namespace boost::ut::literals;

template<typename Handle>
[[nodiscard]] Handle fake_native_handle(void *address) noexcept {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(address);
    } else {
        return static_cast<Handle>(reinterpret_cast<uintptr_t>(address));
    }
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_compute_queue_keeps_indirect_command_stage"_test = [] {
        constexpr auto input =
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        constexpr auto filtered =
            lc::vk::detail::filter_compute_queue_stages(input);
        expect((filtered & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) != 0u);
        expect((filtered & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) != 0u)
            << "compute dispatch-indirect dependencies use DRAW_INDIRECT";
        expect((filtered & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT) == 0u);
    };

    "vk_queue_scope_normalization_preserves_supported_pairs"_test = [] {
        constexpr auto compute = lc::vk::detail::normalize_queue_scope(
            lc::vk::detail::QueueType::COMPUTE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        expect(compute.stages ==
               (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT));
        expect(compute.access ==
               (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT));
    };

    "vk_queue_scope_normalization_conservatively_repairs_foreign_scope"_test = [] {
        constexpr auto copy = lc::vk::detail::normalize_queue_scope(
            lc::vk::detail::QueueType::COPY,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        expect(copy.stages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        expect(copy.access ==
               (VK_ACCESS_2_MEMORY_READ_BIT |
                VK_ACCESS_2_MEMORY_WRITE_BIT));

        constexpr auto execution_only =
            lc::vk::detail::normalize_queue_scope(
                lc::vk::detail::QueueType::COMPUTE,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_NONE);
        expect(execution_only.stages ==
               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        expect(execution_only.access == VK_ACCESS_2_NONE);

        constexpr auto missing_graphics_stage =
            lc::vk::detail::normalize_queue_scope(
                lc::vk::detail::QueueType::GRAPHICS,
                VK_PIPELINE_STAGE_2_NONE,
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        expect(missing_graphics_stage.stages ==
               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        expect(missing_graphics_stage.access ==
               (VK_ACCESS_2_MEMORY_READ_BIT |
                VK_ACCESS_2_MEMORY_WRITE_BIT));
    };

    "vk_texture_layout_policy_separates_generic_and_native_contracts"_test = [] {
        using lc::vk::detail::resolve_texture_barrier_layout;
        using lc::vk::detail::TextureLayoutContract;

        expect(resolve_texture_barrier_layout(
                   true, VK_ACCESS_2_TRANSFER_READ_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   TextureLayoutContract::GENERIC_USAGE) ==
               VK_IMAGE_LAYOUT_GENERAL)
            << "backend-owned commands may query and use the tracker's "
               "simultaneous-access layout";
        expect(resolve_texture_barrier_layout(
                   true, VK_ACCESS_2_TRANSFER_READ_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   TextureLayoutContract::EXPLICIT_NATIVE) ==
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            << "a native command's declared VkImageLayout must remain exact";
        expect(resolve_texture_barrier_layout(
                   true, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   TextureLayoutContract::GENERIC_USAGE) ==
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
            << "attachment layouts remain exact even for generic usage";
        expect(resolve_texture_barrier_layout(
                   false, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   TextureLayoutContract::GENERIC_USAGE) ==
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            << "non-simultaneous images retain the requested layout";
    };

    "vk_texture_access_layout_combination_has_an_empty_identity"_test = [] {
        using lc::vk::detail::combine_texture_access_layout;
        using lc::vk::detail::TextureAccessLayout;

        constexpr auto transfer_read = TextureAccessLayout{
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL};
        constexpr auto empty = TextureAccessLayout{
            VK_ACCESS_2_NONE,
            VK_IMAGE_LAYOUT_GENERAL};
        constexpr auto left_identity =
            combine_texture_access_layout(empty, transfer_read);
        constexpr auto right_identity =
            combine_texture_access_layout(transfer_read, empty);
        expect(left_identity.access == transfer_read.access);
        expect(left_identity.layout == transfer_read.layout);
        expect(right_identity.access == transfer_read.access);
        expect(right_identity.layout == transfer_read.layout);

        constexpr auto repeated = combine_texture_access_layout(
            transfer_read, transfer_read);
        expect(repeated.access == transfer_read.access);
        expect(repeated.layout == transfer_read.layout)
            << "equal read-only native layouts must not collapse to GENERAL";

        constexpr auto incompatible = combine_texture_access_layout(
            transfer_read,
            TextureAccessLayout{
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        expect(incompatible.access ==
               (VK_ACCESS_2_TRANSFER_READ_BIT |
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
        expect(incompatible.layout == VK_IMAGE_LAYOUT_GENERAL)
            << "different abstract read layouts require a common layout";

        constexpr auto read_write = combine_texture_access_layout(
            TextureAccessLayout{
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            TextureAccessLayout{
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL});
        expect(read_write.access ==
               (VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT))
            << "a write must not discard an aliased sampled-read scope";
        expect(read_write.layout == VK_IMAGE_LAYOUT_GENERAL);
    };

    "vk_sampled_descriptor_view_layout_is_uniform"_test = [] {
        using lc::vk::detail::combine_texture_descriptor_view_layout;

        constexpr auto first = combine_texture_descriptor_view_layout(
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        constexpr auto repeated = combine_texture_descriptor_view_layout(
            first, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        constexpr auto mixed = combine_texture_descriptor_view_layout(
            repeated, VK_IMAGE_LAYOUT_GENERAL);
        expect(first == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        expect(repeated == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        expect(mixed == VK_IMAGE_LAYOUT_GENERAL)
            << "one descriptor cannot declare different layouts for its mips";
        expect(combine_texture_descriptor_view_layout(
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ==
               VK_IMAGE_LAYOUT_GENERAL);
    };

    "vk_buffer_identity_preserves_native_aliases"_test = [] {
        uint64_t storage_a = 0u;
        uint64_t storage_b = 0u;
        auto buffer_a = fake_native_handle<VkBuffer>(&storage_a);
        auto alias_a = buffer_a;
        auto buffer_b = fake_native_handle<VkBuffer>(&storage_b);

        auto identity_a = lc::vk::detail::native_buffer_identity(buffer_a);
        expect(identity_a == lc::vk::detail::native_buffer_identity(alias_a));
        expect(identity_a != lc::vk::detail::native_buffer_identity(buffer_b));
    };

    "vk_image_identity_preserves_native_aliases"_test = [] {
        uint64_t storage_a = 0u;
        uint64_t storage_b = 0u;
        auto image_a = fake_native_handle<VkImage>(&storage_a);
        auto alias_a = image_a;
        auto image_b = fake_native_handle<VkImage>(&storage_b);

        auto identity_a = lc::vk::detail::native_image_identity(image_a);
        expect(identity_a == lc::vk::detail::native_image_identity(alias_a));
        expect(identity_a != lc::vk::detail::native_image_identity(image_b));
    };

    "vk_direct_storage_buffer_descriptor_checks_hlsl_abi_boundary"_test = [] {
        using lc::vk::detail::DirectStorageBufferDescriptorStatus;
        using lc::vk::detail::validate_direct_storage_buffer_descriptor;

        expect(validate_direct_storage_buffer_descriptor(
                   256u, 512u, 1024u, 16u, 256u, 512u) ==
               DirectStorageBufferDescriptorStatus::SUCCESS);
        expect(validate_direct_storage_buffer_descriptor(
                   0u, 0u, 1024u, 0u, 256u, 1024u) ==
               DirectStorageBufferDescriptorStatus::EMPTY_RANGE);
        expect(validate_direct_storage_buffer_descriptor(
                   900u, 125u, 1024u, 0u, 4u, 1024u) ==
               DirectStorageBufferDescriptorStatus::VIEW_OUT_OF_BOUNDS);
        expect(validate_direct_storage_buffer_descriptor(
                   4u, 64u, 1024u, 0u, 256u, 1024u) ==
               DirectStorageBufferDescriptorStatus::MISALIGNED_DESCRIPTOR_OFFSET);
        expect(validate_direct_storage_buffer_descriptor(
                   256u, 63u, 1024u, 16u, 256u, 1024u) ==
               DirectStorageBufferDescriptorStatus::INCOMPLETE_TYPED_ELEMENT);
        expect(validate_direct_storage_buffer_descriptor(
                   256u, 768u, 1024u, 0u, 256u, 512u) ==
               DirectStorageBufferDescriptorStatus::RANGE_LIMIT_EXCEEDED);
        expect(validate_direct_storage_buffer_descriptor(
                   0u, 16u, 16u, 0u, 3u, 16u) ==
               DirectStorageBufferDescriptorStatus::INVALID_DESCRIPTOR_ALIGNMENT);

        // Byte-addressed views deliberately use stride zero. They still obey
        // Vulkan descriptor alignment/range rules, but do not pretend to have
        // a typed-element divisibility constraint.
        expect(validate_direct_storage_buffer_descriptor(
                   256u, 7u, 1024u, 0u, 256u, 1024u) ==
               DirectStorageBufferDescriptorStatus::SUCCESS);
    };
}
