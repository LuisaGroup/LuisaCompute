// Test for Vulkan image layout and barrier maintenance extension planning.
//
// Covers the host-visible planning helpers of this extension group:
//   * VK_KHR_unified_image_layouts: `plan_unified_image_layout_policy` and the
//     layout selection it feeds, on both the enabled and the
//     disabled/unsupported path (the pre-existing per-usage layout choice);
//   * the invariant that the layout policy may only change
//     VkImageMemoryBarrier2::oldLayout/newLayout and never a stage or access
//     mask, and never removes a barrier;
//   * `limit_texture_layout_to_access`, the pre-existing demotion rule that
//     stays in force on both paths.
//
// VK_KHR_maintenance8/9/10 add no fast path to the barrier tracker (see the
// comment block at the end of resource_barrier_contract.h): every relaxation
// they add removes friction that lives in resource creation, queue selection,
// the raster path or shader codegen. They are therefore covered by the
// "disabled path is unchanged" cases below, not by an extra plan.

#include "ut/ut.hpp"

#include "resource_barrier_contract.h"

using namespace boost::ut;
using namespace boost::ut::literals;

namespace vk_ext_layouts_test {

using lc::vk::detail::ImageBarrierPlan;
using lc::vk::detail::ImageLayoutPolicy;
using lc::vk::detail::TextureLayoutContract;
using lc::vk::detail::limit_texture_layout_to_access;
using lc::vk::detail::plan_tracked_image_barrier;
using lc::vk::detail::plan_unified_image_layout_policy;
using lc::vk::detail::resolve_texture_barrier_layout;

// The layouts `kBarrierLayoutMap` requests for backend-owned accesses, each
// paired with the access the corresponding `kBarrierAccessMap` entry records.
// A non-attachment-write access is required, because attachment writes always
// keep the attachment layout on both paths.
struct GenericUsage {
    VkImageLayout requested_layout;
    VkAccessFlags2 access;
};

inline constexpr GenericUsage kGenericUsages[] = {
    {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_2_SHADER_READ_BIT},
    {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_2_TRANSFER_READ_BIT},
    {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_2_TRANSFER_WRITE_BIT},
    {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT},
    {VK_IMAGE_LAYOUT_GENERAL,
     VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
};

}// namespace vk_ext_layouts_test

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    using namespace vk_ext_layouts_test;

    "vk_unified_image_layout_policy_is_derived_from_enable_and_sharing"_test = [] {
        constexpr auto unsupported =
            plan_unified_image_layout_policy(false, false);
        expect(!unsupported.general_for_generic_usage);
        expect(!unsupported.uses_unified_image_layouts)
            << "an unsupported or switched-off extension must not be reported "
               "through mark_optional_extension_active";

        constexpr auto enabled = plan_unified_image_layout_policy(true, false);
        expect(enabled.general_for_generic_usage);
        expect(enabled.uses_unified_image_layouts)
            << "the enabled extension is the reason GENERAL is selected";

        constexpr ImageLayoutPolicy simultaneous_only =
            plan_unified_image_layout_policy(false, true);
        expect(simultaneous_only.general_for_generic_usage)
            << "the pre-existing simultaneous-access fast path must survive an "
               "unsupported extension";
        expect(!simultaneous_only.uses_unified_image_layouts)
            << "simultaneous access is an image property, not this extension";

        constexpr auto both = plan_unified_image_layout_policy(true, true);
        expect(both.general_for_generic_usage);
        expect(both.uses_unified_image_layouts);
    };

    "vk_unified_image_layouts_keeps_backend_owned_images_in_general"_test = [] {
        constexpr auto enabled = plan_unified_image_layout_policy(true, false);
        for (auto usage : kGenericUsages) {
            auto layout = resolve_texture_barrier_layout(
                enabled, usage.access, usage.requested_layout,
                TextureLayoutContract::GENERIC_USAGE);
            expect(layout == VK_IMAGE_LAYOUT_GENERAL)
                << "unified image layouts make GENERAL free, so every "
                   "backend-owned usage may stay there (was "
                << static_cast<int32_t>(usage.requested_layout) << ")";
        }
        // The extension also covers images that are not created for
        // simultaneous access: that is the whole point of the fast path.
        expect(resolve_texture_barrier_layout(
                   enabled, VK_ACCESS_2_SHADER_READ_BIT,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   TextureLayoutContract::GENERIC_USAGE) ==
               VK_IMAGE_LAYOUT_GENERAL)
            << "a non-simultaneous image is the case this extension optimizes";
    };

    "vk_unified_image_layouts_disabled_path_is_unchanged"_test = [] {
        constexpr auto disabled =
            plan_unified_image_layout_policy(false, false);
        for (auto usage : kGenericUsages) {
            auto layout = resolve_texture_barrier_layout(
                disabled, usage.access, usage.requested_layout,
                TextureLayoutContract::GENERIC_USAGE);
            expect(layout == usage.requested_layout)
                << "without the extension the tracker keeps selecting the "
                   "usage-specific layout";
        }

        // The pre-existing simultaneous-access behaviour is untouched, both
        // through the explicit policy and through the extension-free overload.
        for (auto usage : kGenericUsages) {
            constexpr auto simultaneous =
                plan_unified_image_layout_policy(false, true);
            expect(resolve_texture_barrier_layout(
                       simultaneous, usage.access, usage.requested_layout,
                       TextureLayoutContract::GENERIC_USAGE) ==
                   VK_IMAGE_LAYOUT_GENERAL);
            expect(resolve_texture_barrier_layout(
                       true, usage.access, usage.requested_layout,
                       TextureLayoutContract::GENERIC_USAGE) ==
                   VK_IMAGE_LAYOUT_GENERAL)
                << "the 4-argument overload must delegate to the policy form";
        }
    };

    "vk_unified_image_layouts_preserves_native_contracts"_test = [] {
        constexpr auto enabled = plan_unified_image_layout_policy(true, true);
        for (auto usage : kGenericUsages) {
            expect(resolve_texture_barrier_layout(
                       enabled, usage.access, usage.requested_layout,
                       TextureLayoutContract::EXPLICIT_NATIVE) ==
                   usage.requested_layout)
                << "a native command's declared VkImageLayout must remain "
                   "exact, otherwise the barrier disagrees with the command";
        }
    };

    "vk_unified_image_layouts_preserves_attachment_layouts"_test = [] {
        constexpr auto enabled = plan_unified_image_layout_policy(true, true);
        constexpr auto attachment_writes =
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        expect(resolve_texture_barrier_layout(
                   enabled, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
                   TextureLayoutContract::GENERIC_USAGE) ==
               VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL)
            << "the render passes built by the raster path keep their exact "
               "attachment layout";
        expect(resolve_texture_barrier_layout(
                   enabled, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                   TextureLayoutContract::GENERIC_USAGE) ==
               VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        expect((attachment_writes &
                (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) ==
               attachment_writes);
    };

    "vk_unified_image_layouts_never_changes_the_hazard"_test = [] {
        constexpr auto before_stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        constexpr auto after_stage = VK_PIPELINE_STAGE_2_COPY_BIT;
        constexpr auto before_access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        constexpr auto after_access = VK_ACCESS_2_TRANSFER_READ_BIT;

        constexpr auto disabled = plan_unified_image_layout_policy(false, false);
        constexpr auto enabled = plan_unified_image_layout_policy(true, false);

        constexpr auto disabled_layout = resolve_texture_barrier_layout(
            disabled, after_access, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            TextureLayoutContract::GENERIC_USAGE);
        constexpr auto enabled_layout = resolve_texture_barrier_layout(
            enabled, after_access, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            TextureLayoutContract::GENERIC_USAGE);

        constexpr ImageBarrierPlan without_extension = plan_tracked_image_barrier(
            before_stage, before_access, VK_IMAGE_LAYOUT_GENERAL,
            after_stage, after_access, disabled_layout);
        constexpr ImageBarrierPlan with_extension = plan_tracked_image_barrier(
            before_stage, before_access, VK_IMAGE_LAYOUT_GENERAL,
            after_stage, after_access, enabled_layout);

        expect(without_extension.src_stage_mask == with_extension.src_stage_mask);
        expect(without_extension.dst_stage_mask == with_extension.dst_stage_mask);
        expect(without_extension.src_access_mask == with_extension.src_access_mask);
        expect(without_extension.dst_access_mask == with_extension.dst_access_mask)
            << "the layout policy may only touch oldLayout/newLayout";
        expect(without_extension.old_layout == with_extension.old_layout);
        expect(without_extension.new_layout != with_extension.new_layout);
        expect(without_extension.new_layout ==
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        expect(with_extension.new_layout == VK_IMAGE_LAYOUT_GENERAL);
        expect(with_extension.has_hazard())
            << "a barrier is still required: the access dependency is the same";
        expect(without_extension.transitions_layout())
            << "the disabled path still reorders the image data";
        expect(!with_extension.transitions_layout())
            << "the enabled path turns the same dependency into an "
               "availability/visibility barrier with oldLayout == newLayout";
    };

    "vk_texture_layout_is_limited_by_the_recorded_access"_test = [] {
        // Pre-existing rule, unchanged by the layout policy: a usage-specific
        // layout survives only while the recorded access still fits it.
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_SHADER_READ_BIT) ==
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_SHADER_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT) ==
               VK_IMAGE_LAYOUT_GENERAL)
            << "a widened access no longer fits a single-purpose layout";
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_ACCESS_2_TRANSFER_READ_BIT) ==
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT) ==
               VK_IMAGE_LAYOUT_GENERAL);
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT) ==
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_ACCESS_2_TRANSFER_READ_BIT) ==
               VK_IMAGE_LAYOUT_GENERAL);
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT) ==
               VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
            << "layouts without a single-access rule are left to the caller";
        expect(limit_texture_layout_to_access(
                   VK_IMAGE_LAYOUT_GENERAL,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) ==
               VK_IMAGE_LAYOUT_GENERAL);
    };

    "vk_tracked_image_barrier_always_carries_the_layout_decision"_test = [] {
        // The enabled policy resolves the same generic usage to GENERAL, so the
        // plan reports no transition while the hazard survives.
        constexpr auto enabled = plan_unified_image_layout_policy(true, false);
        constexpr auto generic_layout = resolve_texture_barrier_layout(
            enabled, VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            TextureLayoutContract::GENERIC_USAGE);
        constexpr ImageBarrierPlan steady_state = plan_tracked_image_barrier(
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT,
            generic_layout);
        expect(steady_state.old_layout == VK_IMAGE_LAYOUT_GENERAL);
        expect(steady_state.new_layout == VK_IMAGE_LAYOUT_GENERAL);
        expect(!steady_state.transitions_layout());
        expect(steady_state.has_hazard())
            << "a matching layout never removes the write-to-read dependency";

        // A barrier whose recorded access does not fit the selected layout is
        // still demoted to GENERAL, on the disabled path as well.
        constexpr auto disabled = plan_unified_image_layout_policy(false, false);
        constexpr auto mixed_read =
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        constexpr ImageBarrierPlan demoted = plan_tracked_image_barrier(
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            mixed_read,
            resolve_texture_barrier_layout(
                disabled, mixed_read,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                TextureLayoutContract::GENERIC_USAGE));
        expect(demoted.new_layout == VK_IMAGE_LAYOUT_GENERAL)
            << "a usage-specific layout that no longer fits collapses to "
               "GENERAL on the disabled path";
        expect(demoted.dst_access_mask == mixed_read);
    };
}
