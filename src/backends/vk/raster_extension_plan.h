#pragma once

#include <cstdint>

namespace lc::vk::detail {

// Host-side planning for the rasterisation optional extensions of the Vulkan
// backend:
//
//   * VK_EXT_primitive_restart_index          (vkCmdSetPrimitiveRestartIndexEXT)
//   * VK_KHR_depth_clamp_zero_one             ([0, 1] depth clamp range)
//   * VK_EXT_custom_resolve                   (shader-provided resolve)
//   * VK_EXT_fragment_density_map_offset      (density-map offsets / end info)
//   * VK_KHR_pipeline_library_group_handles   (RT pipeline-library groups)
//   * VK_EXT_multisampled_render_to_swapchain (MSAA-to-single-sampled swapchain)
//
// Nothing here touches a Vulkan entry point or a `Device`: every helper answers
// "is this fast path legal *and* actually usable for this pipeline / draw, or
// must the existing code run instead" from booleans the caller already
// negotiated. That keeps the decision testable on a machine without the
// extension - which is the only evidence available for the four extensions this
// backend cannot reach through its public API (see the per-helper comments) -
// and it guarantees that a disabled or unsupported extension can never leak a
// struct or an entry point into the recorded command stream.
//
// Everything is header-only (`inline`/`constexpr`) and deliberately free of
// Vulkan and `luisa` includes: the unit test compiles this header directly and
// the backend converts the PODs below into `VkOffset2D`/`VkDynamicState` at the
// one place that records them.

// ---------------------------------------------------------------------------
// VK_EXT_primitive_restart_index
// ---------------------------------------------------------------------------
// The extension adds exactly one command, vkCmdSetPrimitiveRestartIndexEXT, and
// one feature bit. It does *not* add a VkDynamicState value (unlike
// VK_EXT_extended_dynamic_state2's VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
// which only switches restart on/off): without the command the restart index is
// always the one Vulkan derives from the index buffer's type, so a custom index
// is only expressible through the command, and the pipeline keeps its static
// `primitiveRestartEnable`.
//
// The public raster API (`RasterState`, `RasterMesh`) has no restart index at
// all and only ever uses list topologies with `primitiveRestartEnable` left at
// VK_FALSE, so the backend currently requests the type-derived default. Two
// facts still have to be planned correctly for the day that changes:
//
//   * the index buffer is bound by `stream.cpp`, *after* the point where the
//     backend binds the graphics pipeline, and binding an index buffer
//     invalidates a previously recorded custom index ("Default values are
//     applied when binding an index buffer"). The command is therefore only
//     useful when the draw recording can re-record it after that bind, which is
//     why `command_recording_available` is part of the plan instead of being
//     assumed,
//   * when the command is unavailable the driver uses the type-derived index,
//     so a request for any *other* value silently degrades. The plan reports
//     that degradation instead of pretending the request was honoured.

// Vulkan derives the restart index from the index type; `RasterMesh` index
// buffers are always `BufferView<uint>` (VK_INDEX_TYPE_UINT32).
inline constexpr uint32_t primitive_restart_index_uint16 = 0xFFFFu;
inline constexpr uint32_t primitive_restart_index_uint32 = 0xFFFFFFFFu;

enum class PrimitiveRestartIndexSource : uint8_t {
    // primitiveRestartEnable == VK_FALSE: no index value is compared at all.
    NONE,
    // The type-derived index is used and nothing is recorded.
    STATIC_PIPELINE_STATE,
    // vkCmdSetPrimitiveRestartIndexEXT carries the value (fast path).
    DYNAMIC_COMMAND_STATE
};

enum class PrimitiveRestartIndexStatus : uint8_t {
    // Restart is disabled, which is the only state the public raster API can
    // currently produce.
    NO_RESTART,
    // Fast path: the draw records the requested index with
    // vkCmdSetPrimitiveRestartIndexEXT after binding its index buffer.
    DYNAMIC_EXTENSION,
    // Restart is enabled but the requested index equals the type-derived one,
    // so the static (extension-free) path is exact.
    STATIC_EQUIVALENT,
    // Restart is enabled and a custom index was requested, but neither the
    // extension nor a recording point is available: the driver falls back to
    // the type-derived index. `restart_index` is that fallback.
    STATIC_TYPE_DEFAULT_UNHONOURED
};

[[nodiscard]] constexpr const char *
primitive_restart_index_status_name(PrimitiveRestartIndexStatus status) noexcept {
    switch (status) {
        case PrimitiveRestartIndexStatus::NO_RESTART: return "restart disabled";
        case PrimitiveRestartIndexStatus::DYNAMIC_EXTENSION: return "dynamic restart index";
        case PrimitiveRestartIndexStatus::STATIC_EQUIVALENT: return "static restart index (type default)";
        case PrimitiveRestartIndexStatus::STATIC_TYPE_DEFAULT_UNHONOURED: return "custom restart index unavailable";
    }
    return "unknown";
}

struct PrimitiveRestartIndexPlan {
    PrimitiveRestartIndexStatus status{PrimitiveRestartIndexStatus::NO_RESTART};
    PrimitiveRestartIndexSource source{PrimitiveRestartIndexSource::NONE};
    // VkPipelineInputAssemblyStateCreateInfo::primitiveRestartEnable (always a
    // static pipeline field; the extension only makes the index dynamic).
    bool restart_enabled{false};
    // The draw must record vkCmdSetPrimitiveRestartIndexEXT before every draw
    // that can hit a restart index.
    bool record_command{false};
    // ... and specifically after vkCmdBindIndexBuffer, because that bind
    // restores the type-derived default.
    bool record_after_index_buffer_bind{false};
    // False when the requested index could not be expressed (see
    // STATIC_TYPE_DEFAULT_UNHONOURED); `restart_index` is then the value the
    // driver will really use.
    bool honours_requested_index{true};
    // Value the driver compares against: the requested index on the fast path,
    // the type-derived one otherwise.
    uint32_t restart_index{primitive_restart_index_uint32};
};

// `command_recording_available` is true only when the raster draw recording can
// call vkCmdSetPrimitiveRestartIndexEXT *after* vkCmdBindIndexBuffer; a false
// value forces the static path even when the extension is enabled, because a
// value recorded before the bind would be discarded again.
[[nodiscard]] constexpr PrimitiveRestartIndexPlan
plan_primitive_restart_index(
    bool extension_enabled,
    bool command_recording_available,
    bool restart_enabled,
    uint32_t requested_index,
    uint32_t index_type_default) noexcept {
    PrimitiveRestartIndexPlan plan;
    plan.restart_enabled = restart_enabled;
    plan.restart_index = index_type_default;
    if (!restart_enabled) {
        // The only state the public raster state can express: no index value is
        // treated as a restart, so neither the static nor the dynamic index
        // matters and the pipeline keeps primitiveRestartEnable = VK_FALSE.
        plan.status = PrimitiveRestartIndexStatus::NO_RESTART;
        plan.source = PrimitiveRestartIndexSource::NONE;
        return plan;
    }
    if (extension_enabled && command_recording_available) {
        plan.status = PrimitiveRestartIndexStatus::DYNAMIC_EXTENSION;
        plan.source = PrimitiveRestartIndexSource::DYNAMIC_COMMAND_STATE;
        plan.record_command = true;
        plan.record_after_index_buffer_bind = true;
        plan.restart_index = requested_index;
        return plan;
    }
    plan.source = PrimitiveRestartIndexSource::STATIC_PIPELINE_STATE;
    if (requested_index == index_type_default) {
        plan.status = PrimitiveRestartIndexStatus::STATIC_EQUIVALENT;
        return plan;
    }
    plan.status = PrimitiveRestartIndexStatus::STATIC_TYPE_DEFAULT_UNHONOURED;
    plan.honours_requested_index = false;
    return plan;
}

// ---------------------------------------------------------------------------
// VK_KHR_depth_clamp_zero_one
// ---------------------------------------------------------------------------
// The extension contributes a single device feature bit
// (`depthClampZeroOne`) and no structure and no command: enabling it changes
// the fragment depth z_f of *every* rasterisation into a clamp to
// [0, 1], independently of `depthClampEnable`, which is exactly D3D12's
// rasteriser behaviour. That is the whole fast path, so this helper models the
// resulting clamp range rather than a pNext chain.
//
// The pNext form the backend would otherwise use,
// VkPipelineViewportDepthClampControlCreateInfoEXT with a VkDepthClampRangeEXT
// of [0, 1] (VUID-VkDepthClampRangeEXT-pDepthClampRange-09648/09649), belongs
// to VK_EXT_depth_clamp_control - a *different* extension that the shared
// optional-extension registry does not negotiate, so `depth_clamp_control_available`
// is always false in this backend today. The plan already models it so the
// wiring is a one-argument change once that entry exists.
enum class DepthClampRangeSource : uint8_t {
    // Core Vulkan: z is clamped to the viewport's [minDepth, maxDepth] only
    // while depthClampEnable is set.
    VIEWPORT_RANGE,
    // depthClampZeroOne: z is additionally clamped to [0, 1] on every draw.
    ZERO_ONE
};

enum class DepthClampStatus : uint8_t {
    // depthClampEnable = VK_FALSE and no zero-one clamp: primitives outside
    // [0, 1] are clipped (RasterState::depth_clip).
    DISABLED,
    // Clamping was requested but the `depthClamp` feature is not enabled on the
    // logical device, so depthClampEnable has to stay VK_FALSE.
    MISSING_FEATURE,
    // depthClampEnable = VK_TRUE, z clamped to the viewport range only.
    CLAMP_VIEWPORT_RANGE,
    // depthClampEnable = VK_TRUE and depthClampZeroOne clamps z_f to [0, 1].
    CLAMP_ZERO_ONE,
    // Clipping stays on, but depthClampZeroOne still clamps z_f to [0, 1]
    // (the feature is not conditioned on depthClampEnable).
    ZERO_ONE_NO_CLAMP
};

[[nodiscard]] constexpr const char *
depth_clamp_status_name(DepthClampStatus status) noexcept {
    switch (status) {
        case DepthClampStatus::DISABLED: return "clip";
        case DepthClampStatus::MISSING_FEATURE: return "clamp unavailable (feature off)";
        case DepthClampStatus::CLAMP_VIEWPORT_RANGE: return "clamp (viewport range)";
        case DepthClampStatus::CLAMP_ZERO_ONE: return "clamp [0,1]";
        case DepthClampStatus::ZERO_ONE_NO_CLAMP: return "[0,1] clamp without depth clamping";
    }
    return "unknown";
}

struct DepthClampPlan {
    DepthClampStatus status{DepthClampStatus::DISABLED};
    // VkPipelineRasterizationStateCreateInfo::depthClampEnable. `RasterState::
    // depth_clip` follows the D3D12 DepthClipEnable semantics, so a false value
    // (clamping requested) is what sets this.
    bool depth_clamp_enable{false};
    bool clamp_requested{false};
    DepthClampRangeSource range_source{DepthClampRangeSource::VIEWPORT_RANGE};
    // True when the device feature really clamps the fragment depth to [0, 1]
    // for this pipeline, i.e. when VK_KHR_depth_clamp_zero_one is in use.
    bool zero_one_clamp{false};
    // D3D12 clamps to [0, 1]; these are the values the whole backend uses.
    float min_depth_clamp{0.0f};
    float max_depth_clamp{1.0f};
    // Chain VkPipelineViewportDepthClampControlCreateInfoEXT (user-defined
    // range) into VkPipelineViewportStateCreateInfo::pNext.
    bool chain_depth_clamp_control{false};
    bool user_defined_clamp_range{false};
};

// `depth_clip_enabled` is `RasterState::depth_clip` verbatim.
[[nodiscard]] constexpr DepthClampPlan
plan_depth_clamp(
    bool depth_clip_enabled,
    bool depth_clamp_feature_enabled,
    bool depth_clamp_zero_one_enabled,
    bool depth_clamp_control_available) noexcept {
    DepthClampPlan plan;
    plan.clamp_requested = !depth_clip_enabled;
    plan.zero_one_clamp = depth_clamp_zero_one_enabled;
    plan.range_source = depth_clamp_zero_one_enabled ?
                            DepthClampRangeSource::ZERO_ONE :
                            DepthClampRangeSource::VIEWPORT_RANGE;
    plan.depth_clamp_enable =
        plan.clamp_requested && depth_clamp_feature_enabled;
    // The user-defined [0, 1] range is only meaningful while the rasteriser
    // actually clamps, and only legal when the extension providing it is
    // negotiated.
    if (plan.depth_clamp_enable && depth_clamp_control_available) {
        plan.chain_depth_clamp_control = true;
        plan.user_defined_clamp_range = true;
    }
    if (plan.clamp_requested && !depth_clamp_feature_enabled) {
        plan.status = DepthClampStatus::MISSING_FEATURE;
    } else if (plan.depth_clamp_enable) {
        plan.status = depth_clamp_zero_one_enabled ?
                          DepthClampStatus::CLAMP_ZERO_ONE :
                          DepthClampStatus::CLAMP_VIEWPORT_RANGE;
    } else if (depth_clamp_zero_one_enabled) {
        plan.status = DepthClampStatus::ZERO_ONE_NO_CLAMP;
    } else {
        plan.status = DepthClampStatus::DISABLED;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// VK_EXT_custom_resolve
// ---------------------------------------------------------------------------
// A shader-provided resolve needs three things this backend does not have:
//
//   * a resolve attachment on the draw. `DrawRasterSceneCommand` carries up to
//     eight render-target views and one depth/stencil view and nothing else -
//     `create_raster_render_pass` therefore always leaves
//     `pResolveAttachments` null and every attachment is created with
//     VK_SAMPLE_COUNT_1_BIT, so there is no multisample attachment to resolve,
//   * `vkCmdBeginCustomResolveEXT` may only be recorded inside a render pass
//     instance started by vkCmdBeginRendering
//     (VUID-vkCmdBeginCustomResolveEXT-commandBuffer-11517), while the raster
//     path records vkCmdBeginRenderPass against a VkRenderPass,
//   * for the classic render-pass form the subpass flag
//     VK_SUBPASS_DESCRIPTION_CUSTOM_RESOLVE_BIT_EXT additionally requires every
//     entry of `pResolveAttachments` to be VK_ATTACHMENT_UNUSED
//     (VUID-VkSubpassDescription-flags-03341).
//
// So the fast path is unreachable through the public API and the plan always
// selects the existing fixed-function `pResolveAttachments` behaviour.
enum class CustomResolveStatus : uint8_t {
    DISABLED_EXTENSION,
    // The draw has no resolve attachment (the public API cannot express one).
    NO_RESOLVE_ATTACHMENT,
    // A resolve attachment exists but the render pass instance is a classic
    // VkRenderPass, so vkCmdBeginCustomResolveEXT must not be recorded.
    NO_DYNAMIC_RENDERING,
    // Resolving a single-sampled attachment is not a resolve.
    SINGLE_SAMPLED,
    ACTIVE
};

[[nodiscard]] constexpr const char *
custom_resolve_status_name(CustomResolveStatus status) noexcept {
    switch (status) {
        case CustomResolveStatus::DISABLED_EXTENSION: return "extension disabled";
        case CustomResolveStatus::NO_RESOLVE_ATTACHMENT: return "draw has no resolve attachment";
        case CustomResolveStatus::NO_DYNAMIC_RENDERING: return "classic render pass cannot begin a custom resolve";
        case CustomResolveStatus::SINGLE_SAMPLED: return "single-sampled attachment";
        case CustomResolveStatus::ACTIVE: return "shader resolve";
    }
    return "unknown";
}

struct CustomResolvePlan {
    CustomResolveStatus status{CustomResolveStatus::DISABLED_EXTENSION};
    // Fallback: VkSubpassDescription::pResolveAttachments keeps describing a
    // fixed-function multisample resolve, exactly as before.
    bool use_resolve_attachments{false};
    // VK_SUBPASS_DESCRIPTION_CUSTOM_RESOLVE_BIT_EXT on the subpass description.
    bool set_subpass_flag{false};
    // Record vkCmdBeginCustomResolveEXT in the current render pass instance.
    bool record_begin_custom_resolve{false};
    // Chain VkCustomResolveCreateInfoEXT with customResolve = VK_TRUE into
    // VkGraphicsPipelineCreateInfo::pNext.
    bool chain_pipeline_custom_resolve_info{false};
};

[[nodiscard]] constexpr CustomResolvePlan
plan_custom_resolve(
    bool extension_enabled,
    bool command_has_resolve_attachment,
    bool record_path_is_dynamic_rendering,
    uint32_t rasterization_samples) noexcept {
    CustomResolvePlan plan;
    if (!command_has_resolve_attachment) {
        plan.status = CustomResolveStatus::NO_RESOLVE_ATTACHMENT;
        return plan;
    }
    if (!extension_enabled) {
        plan.status = CustomResolveStatus::DISABLED_EXTENSION;
        // The pre-extension behaviour: a fixed-function multisample resolve.
        plan.use_resolve_attachments = true;
        return plan;
    }
    if (rasterization_samples <= 1u) {
        plan.status = CustomResolveStatus::SINGLE_SAMPLED;
        return plan;
    }
    if (!record_path_is_dynamic_rendering) {
        plan.status = CustomResolveStatus::NO_DYNAMIC_RENDERING;
        plan.use_resolve_attachments = true;
        return plan;
    }
    plan.status = CustomResolveStatus::ACTIVE;
    plan.chain_pipeline_custom_resolve_info = true;
    plan.record_begin_custom_resolve = true;
    return plan;
}

// ---------------------------------------------------------------------------
// VK_EXT_fragment_density_map_offset
// ---------------------------------------------------------------------------
// The offsets travel in a VkRenderPassFragmentDensityMapOffsetEndInfoEXT that is
// chained into either `VkSubpassEndInfo::pNext` (classic render pass, read by
// vkCmdEndRenderPass2) or `VkRenderingEndInfoEXT::pNext` (dynamic rendering,
// read by vkCmdEndRendering2EXT). Both are end-of-pass structures recorded by
// the command stream, and both require a *fragment density map attachment*
// (VK_EXT_fragment_density_map, which this backend does not negotiate) plus
// attachments created with VK_IMAGE_CREATE_FRAGMENT_DENSITY_MAP_OFFSET_BIT_EXT:
// VUID-VkRenderPassFragmentDensityMapOffsetEndInfoEXT-fragmentDensityMapOffsets-06503
// forces `fragmentDensityOffsetCount` to 0 without one. The plan enforces that,
// enforces the per-layer count rule and snaps the offsets to the device's
// `fragmentDensityOffsetGranularity` (VUID-...-x-06512 / VUID-...-y-06513).
struct DensityMapOffset {
    int32_t x{0};
    int32_t y{0};
};

enum class FragmentDensityMapOffsetStatus : uint8_t {
    DISABLED_EXTENSION,
    // No fragment density map is attached to the render pass, so
    // fragmentDensityOffsetCount must be 0 (VUID-...-06503) and the implicit
    // (0, 0) offset is used.
    NO_DENSITY_MAP_ATTACHMENT,
    // Nothing to chain: the caller asked for no offset, or the requested offset
    // snapped to the implicit (0, 0).
    NO_OFFSET_REQUESTED,
    ACTIVE
};

[[nodiscard]] constexpr const char *
fragment_density_map_offset_status_name(
    FragmentDensityMapOffsetStatus status) noexcept {
    switch (status) {
        case FragmentDensityMapOffsetStatus::DISABLED_EXTENSION: return "extension disabled";
        case FragmentDensityMapOffsetStatus::NO_DENSITY_MAP_ATTACHMENT: return "no fragment density map attachment";
        case FragmentDensityMapOffsetStatus::NO_OFFSET_REQUESTED: return "no offset";
        case FragmentDensityMapOffsetStatus::ACTIVE: return "density map offset";
    }
    return "unknown";
}

struct FragmentDensityMapOffsetPlan {
    FragmentDensityMapOffsetStatus status{FragmentDensityMapOffsetStatus::DISABLED_EXTENSION};
    // Chain VkRenderPassFragmentDensityMapOffsetEndInfoEXT into the end info of
    // the last subpass / dynamic render pass instance.
    bool chain_end_info{false};
    uint32_t offset_count{0u};
    DensityMapOffset offset{};
    // The requested offset was not a multiple of the granularity and had to be
    // snapped.
    bool snapped{false};
    // The requested count exceeded what this render pass allows and was
    // clamped (1 without multiview, the fragment density map attachment's
    // layerCount with it - VUID-...-fragmentDensityOffsetCount-06511).
    bool count_clamped{false};
};

// A granularity of 0 would divide by zero; Vulkan never reports it, and reading
// it as "no alignment requirement" keeps the helper total.
[[nodiscard]] constexpr uint32_t
fragment_density_map_offset_granularity_or_one(uint32_t granularity) noexcept {
    return granularity == 0u ? 1u : granularity;
}

// Snaps one offset component to the nearest multiple of `granularity`, with
// ties away from zero. Vulkan only accepts exact multiples
// (VUID-VkRenderPassFragmentDensityMapOffsetEndInfoEXT-x-06512/06513), and the
// public API hands us framebuffer-pixel offsets that need not be aligned.
[[nodiscard]] constexpr int32_t
snap_fragment_density_map_offset(int32_t value, uint32_t granularity) noexcept {
    auto step = static_cast<int64_t>(
        fragment_density_map_offset_granularity_or_one(granularity));
    auto signed_value = static_cast<int64_t>(value);
    // Round half away from zero without overflowing on INT32_MIN/INT64_MIN.
    auto magnitude = signed_value < 0 ? -signed_value : signed_value;
    auto rounded = ((magnitude + step / 2) / step) * step;
    if (signed_value < 0) { rounded = -rounded; }
    if (rounded > static_cast<int64_t>(INT32_MAX)) { return INT32_MAX; }
    if (rounded < static_cast<int64_t>(INT32_MIN)) { return INT32_MIN; }
    return static_cast<int32_t>(rounded);
}

[[nodiscard]] constexpr FragmentDensityMapOffsetPlan
plan_fragment_density_map_offset(
    bool extension_enabled,
    bool density_map_attachment_present,
    uint32_t declared_offset_count,
    DensityMapOffset requested_offset,
    uint32_t granularity_width,
    uint32_t granularity_height,
    bool multiview,
    uint32_t fragment_density_map_layer_count) noexcept {
    FragmentDensityMapOffsetPlan plan;
    if (!extension_enabled) {
        plan.status = FragmentDensityMapOffsetStatus::DISABLED_EXTENSION;
        return plan;
    }
    if (!density_map_attachment_present) {
        plan.status = FragmentDensityMapOffsetStatus::NO_DENSITY_MAP_ATTACHMENT;
        return plan;
    }
    if (declared_offset_count == 0u) {
        plan.status = FragmentDensityMapOffsetStatus::NO_OFFSET_REQUESTED;
        return plan;
    }
    auto snapped_x = snap_fragment_density_map_offset(
        requested_offset.x, granularity_width);
    auto snapped_y = snap_fragment_density_map_offset(
        requested_offset.y, granularity_height);
    plan.offset = DensityMapOffset{snapped_x, snapped_y};
    plan.snapped = snapped_x != requested_offset.x ||
                   snapped_y != requested_offset.y;
    if (snapped_x == 0 && snapped_y == 0) {
        // (0, 0) is what the driver uses when no end info is chained at all, so
        // the implicit path is exact and no structure is needed.
        plan.status = FragmentDensityMapOffsetStatus::NO_OFFSET_REQUESTED;
        return plan;
    }
    auto allowed_count = multiview ? fragment_density_map_layer_count : 1u;
    if (allowed_count == 0u) { allowed_count = 1u; }
    plan.offset_count = declared_offset_count < allowed_count ?
                            declared_offset_count :
                            allowed_count;
    plan.count_clamped = plan.offset_count != declared_offset_count;
    plan.chain_end_info = true;
    plan.status = FragmentDensityMapOffsetStatus::ACTIVE;
    return plan;
}

// ---------------------------------------------------------------------------
// VK_KHR_pipeline_library_group_handles
// ---------------------------------------------------------------------------
// Adds only the `pipelineLibraryGroupHandles` feature bit. It lets a ray
// tracing pipeline that links VK_KHR_pipeline_library libraries report shader
// group handles that stay valid for the *library group*, so an application can
// build a pipeline whose group handles were queried from its libraries. The
// backend never creates a library (no VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
// anywhere in `rt_shader.cpp`): every VkRayTracingPipelineCreateInfoKHR carries
// its stages and groups inline, so there is no group to query a handle from and
// the extension can only be enabled, never used.
enum class PipelineLibraryGroupHandleStatus : uint8_t {
    DISABLED_EXTENSION,
    // VK_KHR_ray_tracing_pipeline is not enabled, so there is no RT pipeline.
    NO_RAY_TRACING_PIPELINE,
    // The backend builds monolithic ray tracing pipelines.
    NO_PIPELINE_LIBRARY,
    ACTIVE
};

[[nodiscard]] constexpr const char *
pipeline_library_group_handle_status_name(
    PipelineLibraryGroupHandleStatus status) noexcept {
    switch (status) {
        case PipelineLibraryGroupHandleStatus::DISABLED_EXTENSION: return "extension disabled";
        case PipelineLibraryGroupHandleStatus::NO_RAY_TRACING_PIPELINE: return "no ray tracing pipeline";
        case PipelineLibraryGroupHandleStatus::NO_PIPELINE_LIBRARY: return "no pipeline library";
        case PipelineLibraryGroupHandleStatus::ACTIVE: return "pipeline library group handles";
    }
    return "unknown";
}

struct PipelineLibraryGroupHandlePlan {
    PipelineLibraryGroupHandleStatus status{
        PipelineLibraryGroupHandleStatus::DISABLED_EXTENSION};
    // The negotiated gate asks for `pipelineLibraryGroupHandles` (the shared
    // registry chains the feature structure), so enabling it needs no further
    // work; what is missing is a library to query group handles from.
    bool request_feature{false};
    // Link VK_KHR_pipeline_library libraries with
    // VK_PIPELINE_CREATE_LIBRARY_BIT_KHR and query group handles from them.
    bool link_pipeline_libraries{false};
};

[[nodiscard]] constexpr PipelineLibraryGroupHandlePlan
plan_pipeline_library_group_handles(
    bool extension_enabled,
    bool ray_tracing_pipeline_enabled,
    bool builds_pipeline_libraries) noexcept {
    PipelineLibraryGroupHandlePlan plan;
    plan.request_feature = extension_enabled;
    if (!extension_enabled) {
        plan.status = PipelineLibraryGroupHandleStatus::DISABLED_EXTENSION;
        return plan;
    }
    if (!ray_tracing_pipeline_enabled) {
        plan.status = PipelineLibraryGroupHandleStatus::NO_RAY_TRACING_PIPELINE;
        return plan;
    }
    if (!builds_pipeline_libraries) {
        plan.status = PipelineLibraryGroupHandleStatus::NO_PIPELINE_LIBRARY;
        return plan;
    }
    plan.status = PipelineLibraryGroupHandleStatus::ACTIVE;
    plan.link_pipeline_libraries = true;
    return plan;
}

// ---------------------------------------------------------------------------
// VK_EXT_multisampled_render_to_swapchain
// ---------------------------------------------------------------------------
// Adds VK_SWAPCHAIN_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT (plus
// a VkSwapchainFlagsSurfaceCapabilitiesEXT query) so that the render pass
// mechanism of VK_EXT_multisampled_render_to_single_sampled - a subpass whose
// VkRenderPassMultisampledRenderToSingleSampledInfoEXT asks the implementation
// to run `rasterizationSamples` samples against a single-sampled attachment -
// can also target swapchain images. Two prerequisites are outside this backend:
// the swapchain creation flags belong to `swapchain.cpp`, and the MSRTSS
// extension itself is not part of the optional-extension registry. The raster
// path additionally creates every attachment with VK_SAMPLE_COUNT_1_BIT, so
// there is no multisampled subpass to downgrade.
enum class MultisampledRenderToSwapchainStatus : uint8_t {
    DISABLED_EXTENSION,
    // VK_EXT_multisampled_render_to_single_sampled is not enabled, so the
    // subpass structure it contributes cannot be used.
    MISSING_SINGLE_SAMPLED_EXTENSION,
    // The draw does not render into a swapchain image.
    NO_SWAPCHAIN_ATTACHMENT,
    SINGLE_SAMPLED,
    ACTIVE
};

[[nodiscard]] constexpr const char *
multisampled_render_to_swapchain_status_name(
    MultisampledRenderToSwapchainStatus status) noexcept {
    switch (status) {
        case MultisampledRenderToSwapchainStatus::DISABLED_EXTENSION: return "extension disabled";
        case MultisampledRenderToSwapchainStatus::MISSING_SINGLE_SAMPLED_EXTENSION: return "multisampled-render-to-single-sampled not enabled";
        case MultisampledRenderToSwapchainStatus::NO_SWAPCHAIN_ATTACHMENT: return "not a swapchain attachment";
        case MultisampledRenderToSwapchainStatus::SINGLE_SAMPLED: return "single-sampled attachment";
        case MultisampledRenderToSwapchainStatus::ACTIVE: return "multisampled render to swapchain";
    }
    return "unknown";
}

struct MultisampledRenderToSwapchainPlan {
    MultisampledRenderToSwapchainStatus status{
        MultisampledRenderToSwapchainStatus::DISABLED_EXTENSION};
    // Create the swapchain with
    // VK_SWAPCHAIN_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT.
    bool create_swapchain_with_flag{false};
    // Set VkRenderPassMultisampledRenderToSingleSampledInfoEXT::
    // multisampledRenderToSingleSampledEnable on the subpass.
    bool set_subpass_msrtss_enable{false};
    // Samples the subpass keeps rendering with (the attachment stays
    // single-sampled).
    uint32_t rasterization_samples{1u};
};

[[nodiscard]] constexpr MultisampledRenderToSwapchainPlan
plan_multisampled_render_to_swapchain(
    bool extension_enabled,
    bool single_sampled_extension_enabled,
    bool renders_to_swapchain_image,
    uint32_t rasterization_samples) noexcept {
    MultisampledRenderToSwapchainPlan plan;
    plan.rasterization_samples = rasterization_samples;
    if (!extension_enabled) {
        plan.status = MultisampledRenderToSwapchainStatus::DISABLED_EXTENSION;
        return plan;
    }
    if (!single_sampled_extension_enabled) {
        plan.status =
            MultisampledRenderToSwapchainStatus::MISSING_SINGLE_SAMPLED_EXTENSION;
        return plan;
    }
    if (!renders_to_swapchain_image) {
        plan.status = MultisampledRenderToSwapchainStatus::NO_SWAPCHAIN_ATTACHMENT;
        return plan;
    }
    if (rasterization_samples <= 1u) {
        plan.status = MultisampledRenderToSwapchainStatus::SINGLE_SAMPLED;
        return plan;
    }
    plan.status = MultisampledRenderToSwapchainStatus::ACTIVE;
    plan.create_swapchain_with_flag = true;
    plan.set_subpass_msrtss_enable = true;
    return plan;
}

// ---------------------------------------------------------------------------
// Negotiated gates
// ---------------------------------------------------------------------------
// Snapshot of `Device::optional_extension_enabled()` for the six extensions this
// translation unit group plans around. Passing it explicitly (instead of letting
// the helpers read a Device) keeps every decision above a pure function; the
// backend fills it from the registry, and the unit test fills it directly, which
// is how the "unsupported / switched off" column of the matrix is covered.
struct RasterExtensionGates {
    bool custom_resolve{false};
    bool fragment_density_map_offset{false};
    bool primitive_restart_index{false};
    bool pipeline_library_group_handles{false};
    bool depth_clamp_zero_one{false};
    bool multisampled_render_to_swapchain{false};

    [[nodiscard]] constexpr bool any() const noexcept {
        return custom_resolve || fragment_density_map_offset ||
               primitive_restart_index || pipeline_library_group_handles ||
               depth_clamp_zero_one || multisampled_render_to_swapchain;
    }
};

}// namespace lc::vk::detail
