// Test for the Vulkan backend's presentation / low-latency / frame-pacing
// optional extensions:
//   VK_KHR_present_mode_fifo_latest_ready, VK_KHR_surface_maintenance1,
//   VK_KHR_swapchain_maintenance1, VK_KHR_present_id2, VK_KHR_present_wait2,
//   VK_EXT_present_timing.
//
// This test covers the host-visible planning helpers of that extension group:
// - instance-extension attestation (surface maintenance1 needs the capability
//   query extensions, which the backend does not enable by itself)
// - present-mode selection, including "the driver reports the mode but the
//   extension is off"
// - swapchain present-mode / present-scaling create-info construction, which is
//   only legal once the compatibility / scaling capabilities were queried
// - present-id assignment and rollover
// - present-id2 / present-wait2 gating for every enabled/disabled/unsupported
//   combination
// - present-timing gating, including the "enabled but no public API" outcome
//
// No Vulkan object is created: the helpers are pure and the swapchain only ever
// walks the branch they select, so these cases are the primary evidence on
// headless machines that have no surface.
#include "ut/ut.hpp"
#include "present_plan.h"
#include <array>
#include <cstdint>
#include <limits>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace test_present_detail {

// Owning list so a returned span can never dangle.
struct ModeList {
    std::array<VkPresentModeKHR, 8> storage{};
    uint32_t count{0u};
    ModeList(std::initializer_list<VkPresentModeKHR> values) noexcept {
        for (auto value : values) {
            storage[count++] = value;
        }
    }
    [[nodiscard]] luisa::span<const VkPresentModeKHR> span() const noexcept {
        return luisa::span<const VkPresentModeKHR>{storage.data(),
                                                   static_cast<size_t>(count)};
    }
};

// Owning list of extension names for the attestation helper.
struct NameList {
    luisa::vector<luisa::string> storage;
    NameList(std::initializer_list<const char *> values) {
        for (auto value : values) {
            storage.emplace_back(luisa::string{value});
        }
    }
    [[nodiscard]] luisa::span<const luisa::string> span() const noexcept {
        return luisa::span<const luisa::string>{storage.data(), storage.size()};
    }
};

[[nodiscard]] bool same_name(luisa::string_view actual,
                             luisa::string_view expected) noexcept {
    return actual == expected;
}

}// namespace test_present_detail

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "vk_present_instance_extension_attestation"_test = [] {
        auto none = attest_present_instance_extensions(
            test_present_detail::NameList{}.span());
        expect(!none.surface_maintenance1);
        expect(!none.get_surface_capabilities2);
        expect(!none.full_screen_exclusive);
        expect(!none.can_query_surface_capabilities2());
        expect(!none.can_query_present_modes2());

        auto maintenance_only = attest_present_instance_extensions(
            test_present_detail::NameList{"VK_KHR_surface_maintenance1"}.span());
        expect(maintenance_only.surface_maintenance1);
        expect(!maintenance_only.get_surface_capabilities2)
            << "surface maintenance1 alone cannot reach the capability queries";
        expect(!maintenance_only.can_query_surface_capabilities2());
        expect(!maintenance_only.can_query_present_modes2());

        auto ext_spelling = attest_present_instance_extensions(
            test_present_detail::NameList{"VK_EXT_surface_maintenance1"}.span());
        expect(ext_spelling.surface_maintenance1)
            << "the EXT twin is the same extension";

        auto full = attest_present_instance_extensions(
            test_present_detail::NameList{
                "VK_KHR_get_surface_capabilities2",
                "VK_EXT_full_screen_exclusive",
                "VK_KHR_surface",
                "VK_KHR_win32_surface"}
                .span());
        expect(full.get_surface_capabilities2);
        expect(full.full_screen_exclusive);
        expect(!full.surface_maintenance1);
        expect(full.can_query_surface_capabilities2());
        expect(full.can_query_present_modes2())
            << "vkGetPhysicalDeviceSurfacePresentModes2EXT belongs to "
               "VK_EXT_full_screen_exclusive";

        auto empty_name = attest_present_instance_extensions(
            test_present_detail::NameList{""}.span());
        expect(!empty_name.surface_maintenance1);
    };

    "vk_present_mode_selection_vsync_off"_test = [] {
        auto all_modes = test_present_detail::ModeList{
            VK_PRESENT_MODE_FIFO_KHR,
            VK_PRESENT_MODE_MAILBOX_KHR,
            VK_PRESENT_MODE_IMMEDIATE_KHR};
        auto selection = plan_present_mode(false, all_modes.span(), false);
        expect(selection.status == PresentModeSelectionStatus::SUCCESS);
        expect(selection.mode == VK_PRESENT_MODE_MAILBOX_KHR);
        expect(!selection.used_fifo_latest_ready);
        expect(!selection.fifo_latest_ready_considered);

        // Without mailbox the historical code falls back to immediate.
        auto no_mailbox = test_present_detail::ModeList{
            VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR};
        selection = plan_present_mode(false, no_mailbox.span(), false);
        expect(selection.mode == VK_PRESENT_MODE_IMMEDIATE_KHR);

        // Only FIFO: vsync-off still has to pick a supported mode.
        auto fifo_only = test_present_detail::ModeList{VK_PRESENT_MODE_FIFO_KHR};
        selection = plan_present_mode(false, fifo_only.span(), true);
        expect(selection.status == PresentModeSelectionStatus::SUCCESS);
        expect(selection.mode == VK_PRESENT_MODE_FIFO_KHR)
            << "the low-latency FIFO variant is a vsync mode";
        expect(!selection.used_fifo_latest_ready);
    };

    "vk_present_mode_selection_fifo_latest_ready"_test = [] {
        auto with_latest = test_present_detail::ModeList{
            VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_FIFO_LATEST_READY_KHR};
        auto fifo_only = test_present_detail::ModeList{VK_PRESENT_MODE_FIFO_KHR};
        auto mailbox_only = test_present_detail::ModeList{VK_PRESENT_MODE_MAILBOX_KHR};

        // Enabled + reported: the fast path wins for vsync.
        auto selection = plan_present_mode(true, with_latest.span(), true);
        expect(selection.status == PresentModeSelectionStatus::SUCCESS);
        expect(selection.mode == VK_PRESENT_MODE_FIFO_LATEST_READY_KHR);
        expect(selection.used_fifo_latest_ready);
        expect(selection.fifo_latest_ready_considered);

        // The same surface with the extension switched off (or unsupported, or
        // with the feature bit missing) keeps the pre-existing mode.
        selection = plan_present_mode(true, with_latest.span(), false);
        expect(selection.status == PresentModeSelectionStatus::SUCCESS);
        expect(selection.mode == VK_PRESENT_MODE_FIFO_KHR)
            << "the driver reporting the mode is not enough";
        expect(!selection.used_fifo_latest_ready);
        expect(!selection.fifo_latest_ready_considered);

        // Enabled but the surface does not report the mode.
        selection = plan_present_mode(true, fifo_only.span(), true);
        expect(selection.mode == VK_PRESENT_MODE_FIFO_KHR);
        expect(!selection.used_fifo_latest_ready);
        expect(selection.fifo_latest_ready_considered)
            << "the one-shot log still knows the extension was enabled";

        // A conformant surface always reports FIFO.
        selection = plan_present_mode(true, mailbox_only.span(), false);
        expect(selection.status == PresentModeSelectionStatus::FIFO_UNAVAILABLE);
        expect(selection.mode == VK_PRESENT_MODE_FIFO_KHR)
            << "the caller keeps its fatal check and its fallback mode";
        expect(test_present_detail::same_name(
            present_mode_selection_status_name(selection.status),
            "surface does not report FIFO"));
    };

    "vk_swapchain_present_mode_create_info_gating"_test = [] {
        // Everything switched off: no structure is chained.
        auto plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_KHR,
            false, false, false, false, false, 0u, 0u, 0u);
        expect(!plan.chain_present_modes);
        expect(!plan.chain_present_scaling);
        expect(plan.mode == VK_PRESENT_MODE_FIFO_KHR);

        // Extensions enabled but the capability queries never ran (no
        // VK_KHR_get_surface_capabilities2 on the instance).
        plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_KHR,
            true, true, false, false, false, 0u, 0u, 0u);
        expect(!plan.chain_present_modes);
        expect(!plan.chain_present_scaling)
            << "no capability query means no supported values";

        // Only the device extension: the instance one is a hard prerequisite.
        plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_KHR,
            true, false, true, true, true,
            VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR);
        expect(!plan.chain_present_modes);
        expect(!plan.chain_present_scaling);

        // Queried, but the mode is not in the compatible list.
        plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_KHR,
            true, true, true, false, false, 0u, 0u, 0u);
        expect(!plan.chain_present_modes)
            << "VUID-VkSwapchainPresentModesCreateInfoKHR-pPresentModes-07763";

        // Fully supported: both structures, one-to-one scaling with min gravity.
        plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_LATEST_READY_KHR,
            true, true, true, true, true,
            VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR |
                VK_PRESENT_SCALING_STRETCH_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR);
        expect(plan.chain_present_modes);
        expect(plan.chain_present_scaling);
        expect(plan.mode == VK_PRESENT_MODE_FIFO_LATEST_READY_KHR);
        expect(plan.scaling_behavior == VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR)
            << "exactly one scaling bit (VUID-...-scalingBehavior-07767)";
        expect(plan.gravity_x == VK_PRESENT_GRAVITY_MIN_BIT_KHR);
        expect(plan.gravity_y == VK_PRESENT_GRAVITY_MIN_BIT_KHR);

        // Scaling supported, one-to-one not: no scaling structure at all.
        plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_KHR,
            true, true, true, true, true,
            VK_PRESENT_SCALING_STRETCH_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR);
        expect(plan.chain_present_scaling == false)
            << "VUID-...-scalingBehavior-07770";
        expect(plan.scaling_behavior == 0u);
        expect(plan.gravity_x == 0u);
        expect(plan.gravity_y == 0u);

        // Gravity reported on one axis only: both axes must stay zero
        // (VUID-VkSwapchainPresentScalingCreateInfoKHR-presentGravityX-07765).
        plan = plan_swapchain_present_mode_create_info(
            VK_PRESENT_MODE_FIFO_KHR,
            true, true, true, true, true,
            VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR,
            VK_PRESENT_GRAVITY_MIN_BIT_KHR,
            0u);
        expect(plan.chain_present_scaling);
        expect(plan.gravity_x == 0u);
        expect(plan.gravity_y == 0u);
    };

    "vk_present_id_assignment"_test = [] {
        // Zero is reserved for "no id", the namespace starts at one.
        auto plan = plan_present_id(0u);
        expect(plan.status == PresentIdStatus::ASSIGNED);
        expect(plan.present_id == 1u);
        expect(plan.next_present_id == 2u);

        plan = plan_present_id(7u);
        expect(plan.present_id == 7u);
        expect(plan.next_present_id == 8u);

        // Rollover: the last usable id is UINT64_MAX - 1, then the namespace is
        // exhausted instead of wrapping to zero.
        auto const max = std::numeric_limits<uint64_t>::max();
        plan = plan_present_id(max - 1u);
        expect(plan.status == PresentIdStatus::ASSIGNED);
        expect(plan.present_id == max - 1u);
        expect(plan.next_present_id == max);

        plan = plan_present_id(max);
        expect(plan.status == PresentIdStatus::EXHAUSTED);
        expect(plan.present_id == 0u)
            << "an exhausted namespace must not emit an id";
        expect(plan.next_present_id == max);
        expect(test_present_detail::same_name(
            present_id_status_name(plan.status), "exhausted"));

        // Monotonic across a long run without ever skipping or repeating.
        auto next = uint64_t{1u};
        auto previous = uint64_t{0u};
        for (auto i = 0u; i < 1024u; i++) {
            auto step = plan_present_id(next);
            expect(step.status == PresentIdStatus::ASSIGNED);
            expect(step.present_id == previous + 1u);
            previous = step.present_id;
            next = step.next_present_id;
        }
    };

    "vk_present_id2_gating"_test = [] {
        auto const disabled = plan_present_id2(
            false, true, true, true, true, true);
        expect(disabled.status == PresentId2Status::INACTIVE_EXTENSION_DISABLED);
        expect(!disabled.create_with_present_id2_flag);
        expect(!disabled.create_with_present_wait2_flag);
        expect(!disabled.chain_present_ids);
        expect(!disabled.wait_for_present_ids);
        expect(!disabled);

        // The surface capability query is mandatory: presentId2Supported must be
        // proven before an id may be attached.
        auto const no_query = plan_present_id2(
            true, true, false, true, true, true);
        expect(no_query.status == PresentId2Status::INACTIVE_MISSING_CAPABILITY_QUERY);
        expect(!no_query.create_with_present_id2_flag);
        expect(!no_query.chain_present_ids);

        auto const unsupported = plan_present_id2(
            true, true, true, false, true, true);
        expect(unsupported.status == PresentId2Status::INACTIVE_SURFACE_UNSUPPORTED);
        expect(!unsupported.chain_present_ids);

        // The present path cannot forward the id: enabled but inactive.
        auto const no_present_path = plan_present_id2(
            true, true, true, true, true, false);
        expect(no_present_path.status == PresentId2Status::INACTIVE_PRESENT_PATH);
        expect(!no_present_path.create_with_present_id2_flag);
        expect(!no_present_path.chain_present_ids);

        // Ids only: the swapchain is created with the present-id2 bit, but the
        // surface does not support waiting.
        auto const ids_only = plan_present_id2(
            true, true, true, true, false, true);
        expect(ids_only.status == PresentId2Status::IDS_ONLY);
        expect(ids_only.create_with_present_id2_flag);
        expect(ids_only.chain_present_ids);
        expect(!ids_only.create_with_present_wait2_flag);
        expect(!ids_only.wait_for_present_ids);

        // Present-wait2 switched off by config keeps the same fallback.
        auto const wait_off = plan_present_id2(
            true, false, true, true, true, true);
        expect(wait_off.status == PresentId2Status::IDS_ONLY);
        expect(!wait_off.wait_for_present_ids);

        // Everything on.
        auto const active = plan_present_id2(
            true, true, true, true, true, true);
        expect(active.status == PresentId2Status::ACTIVE);
        expect(active.create_with_present_id2_flag);
        expect(active.create_with_present_wait2_flag);
        expect(active.chain_present_ids);
        expect(active.wait_for_present_ids);
        expect(static_cast<bool>(active));
    };

    "vk_present_timing_gating"_test = [] {
        auto const disabled = plan_present_timing(
            false, true, true, true, true);
        expect(disabled.status == PresentTimingStatus::INACTIVE_EXTENSION_DISABLED);
        expect(!disabled.create_with_present_timing_flag);
        expect(!disabled.collect_past_timing);

        // VK_EXT_present_timing depends on VK_KHR_present_id2.
        auto const no_ids = plan_present_timing(
            true, false, true, true, true);
        expect(no_ids.status == PresentTimingStatus::INACTIVE_MISSING_PRESENT_ID2);
        expect(!no_ids.create_with_present_timing_flag);

        auto const no_query = plan_present_timing(
            true, true, false, true, true);
        expect(no_query.status == PresentTimingStatus::INACTIVE_MISSING_CAPABILITY_QUERY);
        expect(!no_query.collect_past_timing);

        auto const unsupported = plan_present_timing(
            true, true, true, false, true);
        expect(unsupported.status == PresentTimingStatus::INACTIVE_SURFACE_UNSUPPORTED);
        expect(!unsupported.create_with_present_timing_flag);

        // The backend's current state: everything the extension needs is
        // available except a public API that hands per-frame timing back.
        auto const no_public_api = plan_present_timing(
            true, true, true, true, false);
        expect(no_public_api.status == PresentTimingStatus::INACTIVE_NO_PUBLIC_API);
        expect(!no_public_api.create_with_present_timing_flag)
            << "an inactive extension must not change swapchain creation";
        expect(!no_public_api.collect_past_timing);
        expect(test_present_detail::same_name(
            present_timing_status_name(no_public_api.status),
            "no public API delivers present timing"));

        // The hypothetical fully wired case, so the gate cannot silently rot.
        auto const active = plan_present_timing(
            true, true, true, true, true);
        expect(active.status == PresentTimingStatus::ACTIVE);
        expect(active.create_with_present_timing_flag);
        expect(active.collect_past_timing);
        expect(static_cast<bool>(active));
    };

    "vk_present_status_names"_test = [] {
        using test_present_detail::same_name;
        expect(same_name(present_id2_status_name(PresentId2Status::ACTIVE),
                         "active"));
        expect(same_name(present_id2_status_name(PresentId2Status::IDS_ONLY),
                         "present-id only"));
        expect(same_name(
            present_id2_status_name(PresentId2Status::INACTIVE_PRESENT_PATH),
            "present path cannot chain present ids"));
        expect(same_name(present_id_status_name(PresentIdStatus::ASSIGNED),
                         "assigned"));
        expect(same_name(
            present_mode_selection_status_name(
                PresentModeSelectionStatus::SUCCESS),
            "success"));
        expect(same_name(
            present_timing_status_name(
                PresentTimingStatus::INACTIVE_EXTENSION_DISABLED),
            "extension disabled"));
    };
}
