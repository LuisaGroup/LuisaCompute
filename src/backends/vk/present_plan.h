#pragma once
#include <cstdint>
#include <limits>
#include <vulkan/vulkan_core.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace lc::vk::detail {

// Host-side planning for the presentation / low-latency / frame-pacing optional
// extensions of the Vulkan backend:
//
//   * VK_KHR_present_mode_fifo_latest_ready
//   * VK_KHR_surface_maintenance1 + VK_KHR_swapchain_maintenance1
//   * VK_KHR_present_id2 + VK_KHR_present_wait2
//   * VK_EXT_present_timing
//
// Nothing here touches a Vulkan entry point: every helper answers "is this fast
// path legal *and* actually usable for this swapchain, or must the fallback run
// instead". The swapchain then only ever walks the branch the plan selected, so
// a disabled/unsupported extension can never leak a struct or an entry point
// into the recorded command stream.
//
// Everything is header-only (`inline`) because the unit test that proves these
// decisions compiles the header directly and must not need the backend module.

// ---------------------------------------------------------------------------
// Instance-level extension attestation
// ---------------------------------------------------------------------------
// VK_KHR_surface_maintenance1 is an *instance* extension, and the surface
// capability queries it unlocks (VkSurfaceCapabilitiesPresentId2KHR,
// VkPresentTimingSurfaceCapabilitiesEXT,
// VkSurfacePresentScalingCapabilitiesKHR, VkSurfacePresentModeCompatibilityKHR)
// only exist through vkGetPhysicalDeviceSurfaceCapabilities2KHR
// (VK_KHR_get_surface_capabilities2) /
// vkGetPhysicalDeviceSurfacePresentModes2EXT (VK_EXT_full_screen_exclusive).
// Vulkan cannot report which extensions a live instance enabled, so the backend
// attests them from VulkanDeviceConfigExt::extra_instance_exts(), which
// device.cpp passes verbatim to vkCreateInstance. An extension that is not in
// that list must be treated as unavailable: the backend never calls an entry
// point of a disabled extension.
struct PresentInstanceExtensions {
    bool surface_maintenance1{false};
    bool get_surface_capabilities2{false};
    bool full_screen_exclusive{false};

    // vkGetPhysicalDeviceSurfaceCapabilities2KHR: the only way to reach the
    // present-id2 / present-wait2 / present-timing surface capabilities.
    [[nodiscard]] constexpr bool can_query_surface_capabilities2() const noexcept {
        return get_surface_capabilities2;
    }
    // vkGetPhysicalDeviceSurfacePresentModes2EXT. The command belongs to
    // VK_EXT_full_screen_exclusive; VK_KHR_surface_maintenance1 only adds the
    // VkSurfacePresentModeKHR input structure for it.
    [[nodiscard]] constexpr bool can_query_present_modes2() const noexcept {
        return full_screen_exclusive;
    }
};

[[nodiscard]] inline PresentInstanceExtensions
attest_present_instance_extensions(
    luisa::span<const luisa::string> instance_extension_names) noexcept {
    auto result = PresentInstanceExtensions{};
    for (auto const &name : instance_extension_names) {
        if (name.empty()) { continue; }
        if (name == "VK_KHR_surface_maintenance1" ||
            name == "VK_EXT_surface_maintenance1") {
            result.surface_maintenance1 = true;
        } else if (name == "VK_KHR_get_surface_capabilities2") {
            result.get_surface_capabilities2 = true;
        } else if (name == "VK_EXT_full_screen_exclusive") {
            result.full_screen_exclusive = true;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Present mode selection
// ---------------------------------------------------------------------------
enum class PresentModeSelectionStatus : uint8_t {
    // A mode from the surface's list was selected.
    SUCCESS,
    // The surface did not report VK_PRESENT_MODE_FIFO_KHR, which every
    // conformant surface must. The caller keeps its existing fatal check and
    // still receives FIFO as the returned mode.
    FIFO_UNAVAILABLE
};

[[nodiscard]] constexpr const char *
present_mode_selection_status_name(
    PresentModeSelectionStatus status) noexcept {
    switch (status) {
        case PresentModeSelectionStatus::SUCCESS: return "success";
        case PresentModeSelectionStatus::FIFO_UNAVAILABLE: return "surface does not report FIFO";
    }
    return "unknown";
}

struct PresentModeSelection {
    PresentModeSelectionStatus status{PresentModeSelectionStatus::SUCCESS};
    VkPresentModeKHR mode{VK_PRESENT_MODE_FIFO_KHR};
    // The low-latency tear-free mode was really selected, so the caller may
    // report VK_KHR_present_mode_fifo_latest_ready as *active*.
    bool used_fifo_latest_ready{false};
    // The extension was switched on and its feature was enabled, whether or
    // not the surface happened to report the mode. Used for one-shot logging.
    bool fifo_latest_ready_considered{false};
};

// Mirrors the pre-existing policy and adds the FIFO_LATEST_READY upgrade:
//   * vsync off: MAILBOX, else IMMEDIATE, else FIFO;
//   * vsync on and the extension is enabled and the surface reports
//     VK_PRESENT_MODE_FIFO_LATEST_READY_KHR: that mode;
//   * otherwise FIFO.
// The extension is never selected when it is disabled, even if the driver
// reports the mode ("driver reports it but the extension is off").
[[nodiscard]] inline PresentModeSelection
plan_present_mode(
    bool wants_vsync,
    luisa::span<const VkPresentModeKHR> surface_present_modes,
    bool fifo_latest_ready_extension_enabled) noexcept {
    PresentModeSelection selection;
    selection.fifo_latest_ready_considered = fifo_latest_ready_extension_enabled;
    auto reported = [&surface_present_modes](VkPresentModeKHR mode) noexcept {
        for (auto candidate : surface_present_modes) {
            if (candidate == mode) { return true; }
        }
        return false;
    };
    if (!wants_vsync) {// keep the historical mailbox/immediate preference
        for (auto mode : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
            if (reported(mode)) {
                selection.mode = mode;
                return selection;
            }
        }
    } else if (fifo_latest_ready_extension_enabled &&
               reported(VK_PRESENT_MODE_FIFO_LATEST_READY_KHR)) {
        // Tear-free with the latency of the newest ready frame instead of a
        // strictly drained FIFO queue.
        selection.mode = VK_PRESENT_MODE_FIFO_LATEST_READY_KHR;
        selection.used_fifo_latest_ready = true;
        return selection;
    }
    selection.mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!reported(VK_PRESENT_MODE_FIFO_KHR)) {
        selection.status = PresentModeSelectionStatus::FIFO_UNAVAILABLE;
    }
    return selection;
}

// ---------------------------------------------------------------------------
// Swapchain present-mode creation (surface/swapchain maintenance1)
// ---------------------------------------------------------------------------
struct SwapchainPresentModeCreateInfoPlan {
    // Chain VkSwapchainPresentModesCreateInfoKHR. Only legal when the
    // compatible-mode list was queried through
    // vkGetPhysicalDeviceSurfaceCapabilities2KHR, because every listed mode
    // must be a subset of VkSurfacePresentModeCompatibilityKHR::pPresentModes
    // (VUID-VkSwapchainPresentModesCreateInfoKHR-pPresentModes-07763).
    bool chain_present_modes{false};
    // Chain VkSwapchainPresentScalingCreateInfoKHR with a one-to-one pixel
    // mapping. Only legal when the surface reported that scaling method for
    // this present mode (VUID-...-scalingBehavior-07770) and the
    // swapchainMaintenance1 feature is enabled (VUID-...-10154).
    bool chain_present_scaling{false};
    VkPresentModeKHR mode{VK_PRESENT_MODE_FIFO_KHR};
    VkPresentScalingFlagsKHR scaling_behavior{0u};
    VkPresentGravityFlagsKHR gravity_x{0u};
    VkPresentGravityFlagsKHR gravity_y{0u};
};

// `mode_compatible`: the selected mode appears in the
// VkSurfacePresentModeCompatibilityKHR list returned for it.
// `supported_scaling` / `supported_gravity_x` / `supported_gravity_y`: the
// VkSurfacePresentScalingCapabilitiesKHR values returned for the selected mode.
// Gravity is always reported either fully or not at all, because
// VUID-VkSwapchainPresentScalingCreateInfoKHR-presentGravityX-07765/07766
// forbid setting only one axis.
[[nodiscard]] inline SwapchainPresentModeCreateInfoPlan
plan_swapchain_present_mode_create_info(
    VkPresentModeKHR selected_mode,
    bool swapchain_maintenance1_enabled,
    bool surface_maintenance1_enabled,
    bool compatible_mode_query_ran,
    bool mode_compatible,
    bool scaling_capabilities_query_ran,
    VkPresentScalingFlagsKHR supported_scaling,
    VkPresentGravityFlagsKHR supported_gravity_x,
    VkPresentGravityFlagsKHR supported_gravity_y) noexcept {
    SwapchainPresentModeCreateInfoPlan plan;
    plan.mode = selected_mode;
    // Both structures come from VK_KHR_swapchain_maintenance1, which depends on
    // the VK_KHR_surface_maintenance1 instance extension; without the attested
    // instance extension the structs must not be chained at all.
    if (!swapchain_maintenance1_enabled || !surface_maintenance1_enabled) {
        return plan;
    }
    plan.chain_present_modes =
        compatible_mode_query_ran && mode_compatible;
    auto one_to_one_supported =
        (supported_scaling & VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR) != 0u;
    // Gravity only matters when scaling is actually possible; forward it only
    // when both axes report the same supported direction.
    auto gravity = static_cast<VkPresentGravityFlagsKHR>(0u);
    if ((supported_gravity_x & VK_PRESENT_GRAVITY_MIN_BIT_KHR) != 0u &&
        (supported_gravity_y & VK_PRESENT_GRAVITY_MIN_BIT_KHR) != 0u) {
        gravity = VK_PRESENT_GRAVITY_MIN_BIT_KHR;
    } else if ((supported_gravity_x & VK_PRESENT_GRAVITY_CENTERED_BIT_KHR) != 0u &&
               (supported_gravity_y & VK_PRESENT_GRAVITY_CENTERED_BIT_KHR) != 0u) {
        gravity = VK_PRESENT_GRAVITY_CENTERED_BIT_KHR;
    }
    if (scaling_capabilities_query_ran && one_to_one_supported) {
        plan.chain_present_scaling = true;
        plan.scaling_behavior = VK_PRESENT_SCALING_ONE_TO_ONE_BIT_KHR;
        plan.gravity_x = gravity;
        plan.gravity_y = gravity;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Present-id assignment
// ---------------------------------------------------------------------------
enum class PresentIdStatus : uint8_t {
    ASSIGNED,
    // The 64-bit namespace is exhausted. Vulkan requires every non-zero
    // present id to be greater than all previously submitted ids for the same
    // swapchain, and zero means "no id", so the backend must stop attaching
    // ids instead of wrapping.
    EXHAUSTED
};

[[nodiscard]] constexpr const char *
present_id_status_name(PresentIdStatus status) noexcept {
    switch (status) {
        case PresentIdStatus::ASSIGNED: return "assigned";
        case PresentIdStatus::EXHAUSTED: return "exhausted";
    }
    return "unknown";
}

struct PresentIdPlan {
    PresentIdStatus status{PresentIdStatus::EXHAUSTED};
    uint64_t present_id{0u};
    uint64_t next_present_id{0u};
};

// Present ids are monotonic per swapchain and must stay strictly increasing
// across recreation: a recreated swapchain is a fresh VkSwapchainKHR, but
// keeping the single counter monotonic is legal for both and avoids reusing an
// id the presentation engine may still track.
[[nodiscard]] constexpr PresentIdPlan
plan_present_id(uint64_t next_present_id) noexcept {
    // Zero is reserved for "this present carries no id".
    if (next_present_id == 0u) { next_present_id = 1u; }
    if (next_present_id == std::numeric_limits<uint64_t>::max()) {
        return {.status = PresentIdStatus::EXHAUSTED,
                .present_id = 0u,
                .next_present_id = next_present_id};
    }
    return {.status = PresentIdStatus::ASSIGNED,
            .present_id = next_present_id,
            .next_present_id = next_present_id + 1u};
}

// ---------------------------------------------------------------------------
// Present id2 / present wait2 gating
// ---------------------------------------------------------------------------
enum class PresentId2Status : uint8_t {
    // Ids are attached and vkWaitForPresent2KHR may be used.
    ACTIVE,
    // Ids are attached, but present-wait2 is unavailable (extension off, or
    // the surface does not report the capability).
    IDS_ONLY,
    INACTIVE_EXTENSION_DISABLED,
    // The surface capability had to be queried through
    // vkGetPhysicalDeviceSurfaceCapabilities2KHR, which requires an instance
    // extension this instance did not enable.
    INACTIVE_MISSING_CAPABILITY_QUERY,
    INACTIVE_SURFACE_UNSUPPORTED,
    // The present path cannot chain VkPresentId2KHR into VkPresentInfoKHR, so
    // no id would ever reach the presentation engine.
    INACTIVE_PRESENT_PATH
};

[[nodiscard]] constexpr const char *
present_id2_status_name(PresentId2Status status) noexcept {
    switch (status) {
        case PresentId2Status::ACTIVE: return "active";
        case PresentId2Status::IDS_ONLY: return "present-id only";
        case PresentId2Status::INACTIVE_EXTENSION_DISABLED: return "extension disabled";
        case PresentId2Status::INACTIVE_MISSING_CAPABILITY_QUERY: return "no surface capability query";
        case PresentId2Status::INACTIVE_SURFACE_UNSUPPORTED: return "surface does not support present-id2";
        case PresentId2Status::INACTIVE_PRESENT_PATH: return "present path cannot chain present ids";
    }
    return "unknown";
}

struct PresentId2Plan {
    PresentId2Status status{PresentId2Status::INACTIVE_EXTENSION_DISABLED};
    // VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR / ..._WAIT_2_BIT_KHR
    // (VUID-VkPresentId2KHR-None-10820, VUID-vkWaitForPresent2KHR-None-10816).
    bool create_with_present_id2_flag{false};
    bool create_with_present_wait2_flag{false};
    bool chain_present_ids{false};
    bool wait_for_present_ids{false};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return chain_present_ids;
    }
};

[[nodiscard]] inline PresentId2Plan
plan_present_id2(
    bool present_id2_enabled,
    bool present_wait2_enabled,
    bool surface_capability_query_available,
    bool surface_present_id2_supported,
    bool surface_present_wait2_supported,
    bool present_path_can_chain_ids) noexcept {
    PresentId2Plan plan;
    if (!present_id2_enabled) {
        plan.status = PresentId2Status::INACTIVE_EXTENSION_DISABLED;
        return plan;
    }
    if (!surface_capability_query_available) {
        // VUID-VkPresentInfoKHR-presentId2Supported-10822 requires
        // presentId2Supported == VK_TRUE, which is only readable through the
        // chained VkSurfaceCapabilitiesPresentId2KHR. Without the query the
        // backend cannot prove support and must not attach ids.
        plan.status = PresentId2Status::INACTIVE_MISSING_CAPABILITY_QUERY;
        return plan;
    }
    if (!surface_present_id2_supported) {
        plan.status = PresentId2Status::INACTIVE_SURFACE_UNSUPPORTED;
        return plan;
    }
    if (!present_path_can_chain_ids) {
        plan.status = PresentId2Status::INACTIVE_PRESENT_PATH;
        return plan;
    }
    plan.create_with_present_id2_flag = true;
    plan.chain_present_ids = true;
    if (present_wait2_enabled && surface_present_wait2_supported) {
        plan.create_with_present_wait2_flag = true;
        plan.wait_for_present_ids = true;
        plan.status = PresentId2Status::ACTIVE;
    } else {
        plan.status = PresentId2Status::IDS_ONLY;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Present timing gating (VK_EXT_present_timing)
// ---------------------------------------------------------------------------
enum class PresentTimingStatus : uint8_t {
    ACTIVE,
    INACTIVE_EXTENSION_DISABLED,
    // VK_EXT_present_timing depends on VK_KHR_present_id2.
    INACTIVE_MISSING_PRESENT_ID2,
    INACTIVE_MISSING_CAPABILITY_QUERY,
    INACTIVE_SURFACE_UNSUPPORTED,
    // The swapchain and the present path could carry the extension, but no
    // public LuisaCompute API hands per-frame present timing back to the
    // application, so enabling it would only add queue bookkeeping.
    INACTIVE_NO_PUBLIC_API
};

[[nodiscard]] constexpr const char *
present_timing_status_name(PresentTimingStatus status) noexcept {
    switch (status) {
        case PresentTimingStatus::ACTIVE: return "active";
        case PresentTimingStatus::INACTIVE_EXTENSION_DISABLED: return "extension disabled";
        case PresentTimingStatus::INACTIVE_MISSING_PRESENT_ID2: return "VK_KHR_present_id2 unavailable";
        case PresentTimingStatus::INACTIVE_MISSING_CAPABILITY_QUERY: return "no surface capability query";
        case PresentTimingStatus::INACTIVE_SURFACE_UNSUPPORTED: return "surface does not support present timing";
        case PresentTimingStatus::INACTIVE_NO_PUBLIC_API: return "no public API delivers present timing";
    }
    return "unknown";
}

struct PresentTimingPlan {
    PresentTimingStatus status{PresentTimingStatus::INACTIVE_EXTENSION_DISABLED};
    // VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT
    bool create_with_present_timing_flag{false};
    // VkPresentTimingsInfoEXT in the VkPresentInfoKHR chain plus
    // vkGetPastPresentationTimingEXT collection.
    bool collect_past_timing{false};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return collect_past_timing;
    }
};

[[nodiscard]] inline PresentTimingPlan
plan_present_timing(
    bool extension_enabled,
    bool present_ids_available,
    bool surface_capability_query_available,
    bool surface_present_timing_supported,
    bool public_api_delivers_timing) noexcept {
    PresentTimingPlan plan;
    if (!extension_enabled) {
        plan.status = PresentTimingStatus::INACTIVE_EXTENSION_DISABLED;
        return plan;
    }
    if (!present_ids_available) {
        plan.status = PresentTimingStatus::INACTIVE_MISSING_PRESENT_ID2;
        return plan;
    }
    if (!surface_capability_query_available) {
        plan.status = PresentTimingStatus::INACTIVE_MISSING_CAPABILITY_QUERY;
        return plan;
    }
    if (!surface_present_timing_supported) {
        plan.status = PresentTimingStatus::INACTIVE_SURFACE_UNSUPPORTED;
        return plan;
    }
    if (!public_api_delivers_timing) {
        plan.status = PresentTimingStatus::INACTIVE_NO_PUBLIC_API;
        return plan;
    }
    plan.status = PresentTimingStatus::ACTIVE;
    plan.create_with_present_timing_flag = true;
    plan.collect_past_timing = true;
    return plan;
}

}// namespace lc::vk::detail
