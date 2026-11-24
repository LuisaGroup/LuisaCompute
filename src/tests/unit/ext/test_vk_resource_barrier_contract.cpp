#include "ut/ut.hpp"

#include "resource_barrier_contract.h"

#include <array>
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

namespace {

// One read-only tracked mip barrier, i.e. the exact struct
// `ResourceBarrier::_update_state(TextureStates &)` records for a sampled mip.
[[nodiscard]] VkImageMemoryBarrier2 make_image_barrier(
    VkImage image, uint32_t base_mip, uint32_t level_count) noexcept {
    return VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = base_mip,
            .levelCount = level_count,
            .baseArrayLayer = 0u,
            .layerCount = 1u}};
}

}// namespace

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

    // -----------------------------------------------------------------------
    // Conservative barrier elision (read/read only)
    // -----------------------------------------------------------------------
    "vk_write_access_mask_is_the_single_write_definition"_test = [] {
        using lc::vk::detail::has_write_access;

        // Every access the tracker can record as a *write* of a resource.
        constexpr auto declared_writes = std::array{
            VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_HOST_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR};
        for (auto write : declared_writes) {
            expect(has_write_access(write))
                << "a declared write bit must be recognized as a write";
        }
        // Read-only scopes must stay read-only, otherwise no barrier could ever
        // be elided.
        constexpr auto declared_reads = std::array{
            VK_ACCESS_2_SHADER_READ_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_ACCESS_2_HOST_READ_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT,
            VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
            VK_ACCESS_2_INDEX_READ_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR};
        for (auto read : declared_reads) {
            expect(!has_write_access(read))
                << "a read-only access bit must not block elision";
        }
        expect(!has_write_access(VK_ACCESS_2_NONE))
            << "an empty scope is not a write";
        expect(has_write_access(
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT))
            << "one write bit inside a mixed scope is enough to keep the barrier";
    };

    "vk_barrier_elision_requires_a_read_only_same_layout_pair"_test = [] {
        using lc::vk::detail::barrier_elision_blocker;
        using lc::vk::detail::BarrierElisionBlocker;
        using lc::vk::detail::BarrierElisionInput;
        using lc::vk::detail::can_elide_resource_barrier;
        constexpr auto read_only = BarrierElisionInput{
            .src_access = VK_ACCESS_2_SHADER_READ_BIT,
            .dst_access = VK_ACCESS_2_SHADER_READ_BIT,
            .layout_unchanged = true,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = false};
        expect(can_elide_resource_barrier(read_only))
            << "two reads of one subresource with an unchanged layout and no "
               "external before-state never need a barrier";
        expect(barrier_elision_blocker(read_only) ==
               BarrierElisionBlocker::NONE);

        // The steady-state shape the tracker emits under the unified layout
        // policy: same layout, but a *write* is being published.
        constexpr auto publishes_a_write = BarrierElisionInput{
            .src_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .dst_access = VK_ACCESS_2_SHADER_READ_BIT,
            .layout_unchanged = true,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = false};
        expect(!can_elide_resource_barrier(publishes_a_write))
            << "oldLayout == newLayout alone must never permit elision";
        expect(barrier_elision_blocker(publishes_a_write) ==
               BarrierElisionBlocker::SOURCE_WRITE);

        constexpr auto reads_after_a_write = BarrierElisionInput{
            .src_access = VK_ACCESS_2_SHADER_READ_BIT,
            .dst_access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .layout_unchanged = true,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = false};
        expect(barrier_elision_blocker(reads_after_a_write) ==
               BarrierElisionBlocker::DESTINATION_WRITE)
            << "WAR/RAW still needs the dependency";

        constexpr auto write_write = BarrierElisionInput{
            .src_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dst_access = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .layout_unchanged = true,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = false};
        expect(barrier_elision_blocker(write_write) ==
               BarrierElisionBlocker::SOURCE_WRITE);

        constexpr auto transitions = BarrierElisionInput{
            .src_access = VK_ACCESS_2_SHADER_READ_BIT,
            .dst_access = VK_ACCESS_2_SHADER_READ_BIT,
            .layout_unchanged = false,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = false};
        expect(barrier_elision_blocker(transitions) ==
               BarrierElisionBlocker::LAYOUT_TRANSITION)
            << "a layout transition is never redundant, even read to read";
        expect(!can_elide_resource_barrier(transitions));

        constexpr auto external_before = BarrierElisionInput{
            .src_access = VK_ACCESS_2_TRANSFER_READ_BIT,
            .dst_access = VK_ACCESS_2_SHADER_READ_BIT,
            .layout_unchanged = true,
            .before_state_supplied = true,
            .vendor_transfer_read_widened = false};
        expect(barrier_elision_blocker(external_before) ==
               BarrierElisionBlocker::EXTERNAL_BEFORE_STATE)
            << "a caller-supplied before-state describes work the tracker did "
               "not observe, so its dependency is kept";

        constexpr auto nvidia_copy_source = BarrierElisionInput{
            .src_access = VK_ACCESS_2_SHADER_READ_BIT,
            .dst_access = VK_ACCESS_2_TRANSFER_READ_BIT,
            .layout_unchanged = true,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = true};
        expect(barrier_elision_blocker(nvidia_copy_source) ==
               BarrierElisionBlocker::VENDOR_TRANSFER_READ)
            << "the NVIDIA copy-source scope is never elided";

        // An empty dependency orders nothing at all, so it is elided too.
        constexpr auto empty = BarrierElisionInput{
            .src_access = VK_ACCESS_2_NONE,
            .dst_access = VK_ACCESS_2_NONE,
            .layout_unchanged = true,
            .before_state_supplied = false,
            .vendor_transfer_read_widened = false};
        expect(can_elide_resource_barrier(empty));
    };

    "vk_nvidia_transfer_read_workaround_is_vendor_scoped"_test = [] {
        using lc::vk::detail::needs_nvidia_transfer_read_widening;

        expect(needs_nvidia_transfer_read_widening(
            0x10deu, VK_ACCESS_2_TRANSFER_READ_BIT))
            << "an NVIDIA copy source takes the widening workaround";
        expect(needs_nvidia_transfer_read_widening(
            0x10deu,
            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT));
        expect(!needs_nvidia_transfer_read_widening(
            0x10deu, VK_ACCESS_2_SHADER_READ_BIT))
            << "the workaround only covers copy sources";
        expect(!needs_nvidia_transfer_read_widening(
            0x1002u, VK_ACCESS_2_TRANSFER_READ_BIT))
            << "other vendors keep the narrow copy-source scope";
        expect(!needs_nvidia_transfer_read_widening(
            0x8086u, VK_ACCESS_2_TRANSFER_READ_BIT));
        expect(!needs_nvidia_transfer_read_widening(
            0x10deu, VK_ACCESS_2_NONE));
    };

    "vk_barrier_writeback_keeps_the_scope_a_later_hazard_needs"_test = [] {
        using lc::vk::detail::tracked_after_barrier_scope;

        // Two consecutive read-only layers: the destination scope must not
        // replace the source scope, otherwise a later write (WAR) would only
        // order against the last reader.
        constexpr auto carried = tracked_after_barrier_scope(
            true,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
        expect(carried.stages ==
               (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_2_COPY_BIT));
        expect(carried.access ==
               (VK_ACCESS_2_SHADER_READ_BIT |
                VK_ACCESS_2_TRANSFER_READ_BIT))
            << "both read scopes survive an elided barrier";
        constexpr auto read_only_result = !(
            (carried.access & VK_ACCESS_2_TRANSFER_WRITE_BIT) != 0u);
        expect(read_only_result)
            << "carrying read scopes cannot introduce a write";

        // The emitted branch keeps exactly the submitted destination scope.
        constexpr auto emitted = tracked_after_barrier_scope(
            false,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
        expect(emitted.stages == VK_PIPELINE_STAGE_2_COPY_BIT);
        expect(emitted.access == VK_ACCESS_2_TRANSFER_READ_BIT)
            << "a kept barrier makes only its destination scope visible, so the "
               "publishing source scope is not carried forward";

        constexpr auto empty = tracked_after_barrier_scope(
            true,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
        expect(empty.stages == VK_PIPELINE_STAGE_2_NONE &&
               empty.access == VK_ACCESS_2_NONE);
    };

    // -----------------------------------------------------------------------
    // VkImageMemoryBarrier2 coalescing
    // -----------------------------------------------------------------------
    "vk_image_barriers_merge_only_identical_contiguous_mips"_test = [] {
        using lc::vk::detail::absorb_image_barrier_range;
        using lc::vk::detail::can_merge_image_barriers;

        uint64_t storage = 0u;
        auto image = fake_native_handle<VkImage>(&storage);

        auto mip0 = make_image_barrier(image, 0u, 1u);
        auto mip1 = make_image_barrier(image, 1u, 1u);
        auto mip2 = make_image_barrier(image, 2u, 1u);
        expect(can_merge_image_barriers(mip0, mip1))
            << "identical masks/layouts on adjacent mips are one subresource "
               "range";
        expect(!can_merge_image_barriers(mip0, mip0))
            << "a struct never merges with itself (the mip range would overlap)";
        expect(!can_merge_image_barriers(mip0, mip2))
            << "a gap between two mips is a subresource the caller did not "
               "synchronize and must stay unmerged";

        // Masks and layouts must be identical: a merge must not weaken either
        // side.
        auto different_dst_access = mip1;
        different_dst_access.dstAccessMask |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        expect(!can_merge_image_barriers(mip0, different_dst_access));
        auto different_src_stage = mip1;
        different_src_stage.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        expect(!can_merge_image_barriers(mip0, different_src_stage));
        auto different_src_access = mip1;
        different_src_access.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        expect(!can_merge_image_barriers(mip0, different_src_access));
        auto different_old_layout = mip1;
        different_old_layout.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        expect(!can_merge_image_barriers(mip0, different_old_layout));
        auto different_new_layout = mip1;
        different_new_layout.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        expect(!can_merge_image_barriers(mip0, different_new_layout));

        // A different image, aspect, array-layer range or queue ownership is a
        // different barrier completely.
        uint64_t other_storage = 0u;
        auto other_image = make_image_barrier(
            fake_native_handle<VkImage>(&other_storage), 1u, 1u);
        expect(!can_merge_image_barriers(mip0, other_image));
        auto different_aspect = mip1;
        different_aspect.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        expect(!can_merge_image_barriers(mip0, different_aspect));
        auto different_base_layer = mip1;
        different_base_layer.subresourceRange.baseArrayLayer = 1u;
        expect(!can_merge_image_barriers(mip0, different_base_layer));
        auto different_layer_count = mip1;
        different_layer_count.subresourceRange.layerCount = 2u;
        expect(!can_merge_image_barriers(mip0, different_layer_count));
        auto different_src_queue = mip1;
        different_src_queue.srcQueueFamilyIndex = 1u;
        expect(!can_merge_image_barriers(mip0, different_src_queue));
        auto different_dst_queue = mip1;
        different_dst_queue.dstQueueFamilyIndex = 1u;
        expect(!can_merge_image_barriers(mip0, different_dst_queue));

        // An extended pNext chain cannot be compared, so it is never merged.
        auto chained = mip1;
        chained.pNext = &mip2;
        expect(!can_merge_image_barriers(mip0, chained));

        // The fold only widens the mip range.
        auto merged = mip0;
        absorb_image_barrier_range(merged, mip1);
        expect(merged.subresourceRange.baseMipLevel == 0u);
        expect(merged.subresourceRange.levelCount == 2u)
            << "a merged barrier covers exactly the union of its mips";
        absorb_image_barrier_range(merged, mip2);
        expect(merged.subresourceRange.levelCount == 3u);
        expect(merged.srcStageMask == mip0.srcStageMask &&
               merged.dstStageMask == mip0.dstStageMask &&
               merged.srcAccessMask == mip0.srcAccessMask &&
               merged.dstAccessMask == mip0.dstAccessMask &&
               merged.oldLayout == mip0.oldLayout &&
               merged.newLayout == mip0.newLayout &&
               merged.image == mip0.image)
            << "merging never rewrites the scope, the layouts or the image";
    };

    "vk_image_barrier_merge_accepts_a_full_mip_chain"_test = [] {
        using lc::vk::detail::absorb_image_barrier_range;
        using lc::vk::detail::can_merge_image_barriers;

        uint64_t storage = 0u;
        auto image = fake_native_handle<VkImage>(&storage);
        // A sampled descriptor view over 5 mips, recorded mip by mip: the
        // in-place compaction in `ResourceBarrier::update_states` must be able to
        // fold the whole run into one struct.
        constexpr auto mip_count = 5u;
        auto merged = make_image_barrier(image, 0u, 1u);
        auto merged_count = size_t{1u};
        for (auto level = 1u; level < mip_count; ++level) {
            auto next = make_image_barrier(image, level, 1u);
            if (can_merge_image_barriers(merged, next)) {
                absorb_image_barrier_range(merged, next);
            } else {
                ++merged_count;
            }
        }
        expect(merged_count == 1u)
            << "5 contiguous read-only mips collapse into one barrier struct";
        expect(merged.subresourceRange.levelCount == mip_count);

        // ... and a hole in the chain keeps the run apart.
        auto first = make_image_barrier(image, 0u, 1u);
        auto third = make_image_barrier(image, 3u, 1u);
        expect(!can_merge_image_barriers(first, third));
    };
}
