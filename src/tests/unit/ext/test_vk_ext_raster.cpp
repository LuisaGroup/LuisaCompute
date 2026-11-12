// Test for Vulkan raster extension planning (custom resolve, fragment density
// map offset, primitive restart index, depth clamp zero-one, pipeline library
// group handles, MSAA-to-swapchain).
//
// This test covers the host-visible planning helpers of the extension group
// above (`src/backends/vk/raster_extension_plan.h`):
//
// - the dynamic-versus-static primitive restart index decision, including the
//   "the extension is enabled but nothing can re-record the index after
//   vkCmdBindIndexBuffer" degradation and the "the request equals the
//   type-derived index, so the static path is exact" equivalence
// - the depth clamp range mapping: clip, viewport-range clamp, the [0, 1] clamp
//   VK_KHR_depth_clamp_zero_one adds, the missing-feature case, and the
//   user-defined range pNext chain that VK_EXT_depth_clamp_control would allow
// - the custom resolve preconditions (resolve attachment, multisampling, and the
//   dynamic-rendering requirement of vkCmdBeginCustomResolveEXT)
// - the fragment density map offset end-info construction: the "no density map
//   attachment therefore no offsets" rule, granularity snapping, negative and
//   saturating offsets, and the per-layer offset count limit
// - the two "enabled but inactive" extensions (pipeline library group handles,
//   multisampled render to swapchain), whose fast path the backend cannot reach
// - every status-name mapping
//
// No Vulkan object is created and no device is required: the helpers are pure,
// the backend only ever walks the branch they select, and these cases are the
// only evidence available for the four extensions the test GPU does not expose
// (VK_EXT_fragment_density_map_offset, VK_EXT_primitive_restart_index,
// VK_EXT_multisampled_render_to_swapchain and, as far as a draw is concerned,
// VK_EXT_custom_resolve).
#include "ut/ut.hpp"

#include "raster_extension_plan.h"

#include <cstdint>
#include <cstring>

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace lc::vk::detail;

namespace test_vk_ext_raster_detail {

// boost::ut prints both operands of a failed `expect(lhs == rhs)`, which needs an
// `operator<<` for the enum. Comparing the underlying value keeps the enum classes
// strongly typed while giving readable failures.
template<typename E>
[[nodiscard]] bool same(E lhs, E rhs) noexcept {
    return static_cast<int>(lhs) == static_cast<int>(rhs);
}

[[nodiscard]] bool same_text(const char *lhs, const char *rhs) noexcept {
    return lhs != nullptr && rhs != nullptr && std::strcmp(lhs, rhs) == 0;
}

[[nodiscard]] bool non_empty_text(const char *text) noexcept {
    return text != nullptr && text[0] != '\0' &&
           !same_text(text, "unknown");
}

}// namespace test_vk_ext_raster_detail

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    using namespace test_vk_ext_raster_detail;

    // -----------------------------------------------------------------------
    // VK_EXT_primitive_restart_index
    // -----------------------------------------------------------------------
    "primitive_restart_index_no_restart"_test = [] {
        // The only state the public raster API can produce: list topologies, no
        // restart index. Neither the extension nor a recording point changes it.
        for (auto extension_enabled : {false, true}) {
            for (auto recordable : {false, true}) {
                auto plan = plan_primitive_restart_index(
                    extension_enabled, recordable,
                    /*restart_enabled=*/false,
                    /*requested_index=*/0x1234u,
                    primitive_restart_index_uint32);
                expect(same(plan.status, PrimitiveRestartIndexStatus::NO_RESTART));
                expect(same(plan.source, PrimitiveRestartIndexSource::NONE));
                expect(!plan.restart_enabled);
                expect(!plan.record_command);
                expect(!plan.record_after_index_buffer_bind);
                expect(plan.honours_requested_index);
                // Unused, so the type-derived default is reported.
                expect(plan.restart_index == primitive_restart_index_uint32);
            }
        }
    };

    "primitive_restart_index_dynamic"_test = [] {
        auto plan = plan_primitive_restart_index(
            /*extension_enabled=*/true,
            /*command_recording_available=*/true,
            /*restart_enabled=*/true,
            /*requested_index=*/0xCAFEu,
            primitive_restart_index_uint32);
        expect(same(plan.status, PrimitiveRestartIndexStatus::DYNAMIC_EXTENSION));
        expect(same(plan.source, PrimitiveRestartIndexSource::DYNAMIC_COMMAND_STATE));
        expect(plan.restart_enabled);
        expect(plan.record_command);
        // Binding an index buffer restores the default, so the value has to be
        // recorded after the bind.
        expect(plan.record_after_index_buffer_bind);
        expect(plan.honours_requested_index);
        expect(plan.restart_index == 0xCAFEu);
    };

    "primitive_restart_index_enabled_but_unrecordable"_test = [] {
        // VK_EXT_primitive_restart_index is enabled, but no draw recording point
        // can re-record the value after vkCmdBindIndexBuffer: the driver keeps
        // the type-derived index and the plan must say so instead of pretending.
        auto plan = plan_primitive_restart_index(
            /*extension_enabled=*/true,
            /*command_recording_available=*/false,
            /*restart_enabled=*/true,
            /*requested_index=*/0xCAFEu,
            primitive_restart_index_uint32);
        expect(same(plan.status,
                    PrimitiveRestartIndexStatus::STATIC_TYPE_DEFAULT_UNHONOURED));
        expect(same(plan.source, PrimitiveRestartIndexSource::STATIC_PIPELINE_STATE));
        expect(plan.restart_enabled);
        expect(!plan.record_command);
        expect(!plan.record_after_index_buffer_bind);
        expect(!plan.honours_requested_index);
        expect(plan.restart_index == primitive_restart_index_uint32);
    };

    "primitive_restart_index_static_equivalence"_test = [] {
        // No extension, but the request is exactly what the driver derives from
        // the index type, so the static path is exact and nothing degrades.
        auto plan = plan_primitive_restart_index(
            /*extension_enabled=*/false,
            /*command_recording_available=*/true,
            /*restart_enabled=*/true,
            primitive_restart_index_uint32,
            primitive_restart_index_uint32);
        expect(same(plan.status, PrimitiveRestartIndexStatus::STATIC_EQUIVALENT));
        expect(same(plan.source, PrimitiveRestartIndexSource::STATIC_PIPELINE_STATE));
        expect(plan.honours_requested_index);
        expect(!plan.record_command);
        expect(plan.restart_index == primitive_restart_index_uint32);
        // The 16-bit default only differs from the 32-bit one, and every
        // RasterMesh index buffer is a BufferView<uint>.
        expect(primitive_restart_index_uint16 == 0xFFFFu);
        expect(primitive_restart_index_uint32 == 0xFFFFFFFFu);
        auto u16_plan = plan_primitive_restart_index(
            /*extension_enabled=*/false,
            /*command_recording_available=*/false,
            /*restart_enabled=*/true,
            primitive_restart_index_uint16,
            primitive_restart_index_uint16);
        expect(same(u16_plan.status,
                    PrimitiveRestartIndexStatus::STATIC_EQUIVALENT));
        expect(u16_plan.restart_index == 0xFFFFu);
    };

    "primitive_restart_index_disabled_extension_custom_index"_test = [] {
        auto plan = plan_primitive_restart_index(
            /*extension_enabled=*/false,
            /*command_recording_available=*/true,
            /*restart_enabled=*/true,
            /*requested_index=*/0u,
            primitive_restart_index_uint32);
        expect(same(plan.status,
                    PrimitiveRestartIndexStatus::STATIC_TYPE_DEFAULT_UNHONOURED));
        expect(!plan.honours_requested_index);
        expect(plan.restart_index == primitive_restart_index_uint32);
    };

    // -----------------------------------------------------------------------
    // VK_KHR_depth_clamp_zero_one
    // -----------------------------------------------------------------------
    "depth_clamp_disabled"_test = [] {
        // RasterState::depth_clip is the default: primitives outside [0, 1] are
        // clipped, no clamp range is involved.
        auto plan = plan_depth_clamp(
            /*depth_clip_enabled=*/true,
            /*depth_clamp_feature_enabled=*/true,
            /*depth_clamp_zero_one_enabled=*/false,
            /*depth_clamp_control_available=*/false);
        expect(same(plan.status, DepthClampStatus::DISABLED));
        expect(!plan.depth_clamp_enable);
        expect(!plan.clamp_requested);
        expect(!plan.zero_one_clamp);
        expect(same(plan.range_source, DepthClampRangeSource::VIEWPORT_RANGE));
        expect(!plan.chain_depth_clamp_control);
        expect(plan.min_depth_clamp == 0.0f);
        expect(plan.max_depth_clamp == 1.0f);
    };

    "depth_clamp_viewport_range"_test = [] {
        // Clamping requested (depth_clip == false) without the extension: core
        // Vulkan clamps z to the viewport's [minDepth, maxDepth].
        auto plan = plan_depth_clamp(
            /*depth_clip_enabled=*/false,
            /*depth_clamp_feature_enabled=*/true,
            /*depth_clamp_zero_one_enabled=*/false,
            /*depth_clamp_control_available=*/false);
        expect(same(plan.status, DepthClampStatus::CLAMP_VIEWPORT_RANGE));
        expect(plan.depth_clamp_enable);
        expect(plan.clamp_requested);
        expect(!plan.zero_one_clamp);
        expect(same(plan.range_source, DepthClampRangeSource::VIEWPORT_RANGE));
        // The user-defined [0, 1] range needs VK_EXT_depth_clamp_control.
        expect(!plan.chain_depth_clamp_control);
        expect(!plan.user_defined_clamp_range);
    };

    "depth_clamp_zero_one_enabled"_test = [] {
        // The extension's whole effect: the fragment depth is clamped to [0, 1],
        // which is D3D12's range.
        auto plan = plan_depth_clamp(
            /*depth_clip_enabled=*/false,
            /*depth_clamp_feature_enabled=*/true,
            /*depth_clamp_zero_one_enabled=*/true,
            /*depth_clamp_control_available=*/false);
        expect(same(plan.status, DepthClampStatus::CLAMP_ZERO_ONE));
        expect(plan.depth_clamp_enable);
        expect(plan.zero_one_clamp);
        expect(same(plan.range_source, DepthClampRangeSource::ZERO_ONE));
        expect(plan.min_depth_clamp == 0.0f);
        expect(plan.max_depth_clamp == 1.0f);
        // Even with the [0, 1] clamp available, the pNext form needs
        // VK_EXT_depth_clamp_control, which this backend never negotiates.
        expect(!plan.chain_depth_clamp_control);
    };

    "depth_clamp_zero_one_without_depth_clamping"_test = [] {
        // The feature is not conditioned on depthClampEnable, so a clipping
        // pipeline still clamps the fragment depth to [0, 1].
        auto plan = plan_depth_clamp(
            /*depth_clip_enabled=*/true,
            /*depth_clamp_feature_enabled=*/true,
            /*depth_clamp_zero_one_enabled=*/true,
            /*depth_clamp_control_available=*/false);
        expect(same(plan.status, DepthClampStatus::ZERO_ONE_NO_CLAMP));
        expect(!plan.depth_clamp_enable);
        expect(plan.zero_one_clamp);
        expect(same(plan.range_source, DepthClampRangeSource::ZERO_ONE));
    };

    "depth_clamp_missing_feature"_test = [] {
        // Clamping requested on a device without the depthClamp feature: the
        // pipeline cannot enable it, and the plan reports that instead of
        // building an invalid pipeline.
        auto plan = plan_depth_clamp(
            /*depth_clip_enabled=*/false,
            /*depth_clamp_feature_enabled=*/false,
            /*depth_clamp_zero_one_enabled=*/false,
            /*depth_clamp_control_available=*/false);
        expect(same(plan.status, DepthClampStatus::MISSING_FEATURE));
        expect(!plan.depth_clamp_enable);
        expect(plan.clamp_requested);
        auto with_zero_one = plan_depth_clamp(
            /*depth_clip_enabled=*/false,
            /*depth_clamp_feature_enabled=*/false,
            /*depth_clamp_zero_one_enabled=*/true,
            /*depth_clamp_control_available=*/true);
        expect(same(with_zero_one.status, DepthClampStatus::MISSING_FEATURE));
        expect(!with_zero_one.depth_clamp_enable);
        // No clamping means no user-defined clamp range either.
        expect(!with_zero_one.chain_depth_clamp_control);
    };

    "depth_clamp_user_defined_range"_test = [] {
        // What the backend would chain if the shared registry negotiated
        // VK_EXT_depth_clamp_control: the same [0, 1] range, this time as
        // VkPipelineViewportDepthClampControlCreateInfoEXT +
        // VkDepthClampRangeEXT{pDepthClampRange} in
        // VkPipelineViewportStateCreateInfo::pNext.
        auto plan = plan_depth_clamp(
            /*depth_clip_enabled=*/false,
            /*depth_clamp_feature_enabled=*/true,
            /*depth_clamp_zero_one_enabled=*/true,
            /*depth_clamp_control_available=*/true);
        expect(plan.chain_depth_clamp_control);
        expect(plan.user_defined_clamp_range);
        expect(plan.min_depth_clamp >= 0.0f);
        expect(plan.max_depth_clamp <= 1.0f);
        expect(plan.min_depth_clamp <= plan.max_depth_clamp);
        // The chain is only for a pipeline that really clamps.
        auto clipping = plan_depth_clamp(
            /*depth_clip_enabled=*/true,
            /*depth_clamp_feature_enabled=*/true,
            /*depth_clamp_zero_one_enabled=*/true,
            /*depth_clamp_control_available=*/true);
        expect(!clipping.chain_depth_clamp_control);
        expect(!clipping.user_defined_clamp_range);
    };

    // -----------------------------------------------------------------------
    // VK_EXT_custom_resolve
    // -----------------------------------------------------------------------
    "custom_resolve_no_resolve_attachment"_test = [] {
        // The public raster command carries render targets and a depth buffer
        // only, so there is never a resolve attachment to write.
        for (auto extension_enabled : {false, true}) {
            for (auto dynamic_rendering : {false, true}) {
                auto plan = plan_custom_resolve(
                    extension_enabled,
                    /*command_has_resolve_attachment=*/false,
                    dynamic_rendering,
                    /*rasterization_samples=*/4u);
                expect(same(plan.status, CustomResolveStatus::NO_RESOLVE_ATTACHMENT));
                expect(!plan.use_resolve_attachments);
                expect(!plan.set_subpass_flag);
                expect(!plan.record_begin_custom_resolve);
                expect(!plan.chain_pipeline_custom_resolve_info);
            }
        }
    };

    "custom_resolve_disabled_extension"_test = [] {
        // Pre-extension behaviour: a fixed-function multisample resolve through
        // VkSubpassDescription::pResolveAttachments.
        auto plan = plan_custom_resolve(
            /*extension_enabled=*/false,
            /*command_has_resolve_attachment=*/true,
            /*record_path_is_dynamic_rendering=*/true,
            /*rasterization_samples=*/4u);
        expect(same(plan.status, CustomResolveStatus::DISABLED_EXTENSION));
        expect(plan.use_resolve_attachments);
        expect(!plan.record_begin_custom_resolve);
        expect(!plan.set_subpass_flag);
    };

    "custom_resolve_requires_multisampling"_test = [] {
        auto plan = plan_custom_resolve(
            /*extension_enabled=*/true,
            /*command_has_resolve_attachment=*/true,
            /*record_path_is_dynamic_rendering=*/true,
            /*rasterization_samples=*/1u);
        expect(same(plan.status, CustomResolveStatus::SINGLE_SAMPLED));
        expect(!plan.use_resolve_attachments);
        expect(!plan.record_begin_custom_resolve);
    };

    "custom_resolve_requires_dynamic_rendering"_test = [] {
        // vkCmdBeginCustomResolveEXT is only legal inside a render pass instance
        // opened by vkCmdBeginRendering
        // (VUID-vkCmdBeginCustomResolveEXT-commandBuffer-11517), which is what
        // the classic VkRenderPass path of this backend is not.
        auto plan = plan_custom_resolve(
            /*extension_enabled=*/true,
            /*command_has_resolve_attachment=*/true,
            /*record_path_is_dynamic_rendering=*/false,
            /*rasterization_samples=*/4u);
        expect(same(plan.status, CustomResolveStatus::NO_DYNAMIC_RENDERING));
        expect(plan.use_resolve_attachments);
        expect(!plan.record_begin_custom_resolve);
        expect(!plan.chain_pipeline_custom_resolve_info);
    };

    "custom_resolve_active"_test = [] {
        auto plan = plan_custom_resolve(
            /*extension_enabled=*/true,
            /*command_has_resolve_attachment=*/true,
            /*record_path_is_dynamic_rendering=*/true,
            /*rasterization_samples=*/4u);
        expect(same(plan.status, CustomResolveStatus::ACTIVE));
        expect(!plan.use_resolve_attachments);
        expect(plan.record_begin_custom_resolve);
        expect(plan.chain_pipeline_custom_resolve_info);
    };

    // -----------------------------------------------------------------------
    // VK_EXT_fragment_density_map_offset
    // -----------------------------------------------------------------------
    "density_map_offset_disabled_extension"_test = [] {
        auto plan = plan_fragment_density_map_offset(
            /*extension_enabled=*/false,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/1u,
            DensityMapOffset{4, 4},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/1u);
        expect(same(plan.status,
                    FragmentDensityMapOffsetStatus::DISABLED_EXTENSION));
        expect(!plan.chain_end_info);
        expect(plan.offset_count == 0u);
        expect(plan.offset.x == 0 && plan.offset.y == 0);
    };

    "density_map_offset_requires_density_map_attachment"_test = [] {
        // VUID-VkRenderPassFragmentDensityMapOffsetEndInfoEXT-
        // fragmentDensityMapOffsets-06503 forces the count to 0 when the render
        // pass has no fragment density map attachment, which is every render
        // pass this backend creates (VK_EXT_fragment_density_map is not
        // negotiated).
        auto plan = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/false,
            /*declared_offset_count=*/1u,
            DensityMapOffset{8, -8},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/1u);
        expect(same(plan.status,
                    FragmentDensityMapOffsetStatus::NO_DENSITY_MAP_ATTACHMENT));
        expect(!plan.chain_end_info);
        expect(plan.offset_count == 0u);
    };

    "density_map_offset_zero_count"_test = [] {
        auto plan = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/0u,
            DensityMapOffset{4, 4},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/1u);
        expect(same(plan.status, FragmentDensityMapOffsetStatus::NO_OFFSET_REQUESTED));
        expect(!plan.chain_end_info);
    };

    "density_map_offset_active"_test = [] {
        auto plan = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/1u,
            DensityMapOffset{8, -12},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/1u);
        expect(same(plan.status, FragmentDensityMapOffsetStatus::ACTIVE));
        expect(plan.chain_end_info);
        expect(plan.offset_count == 1u);
        expect(!plan.snapped);
        expect(!plan.count_clamped);
        expect(plan.offset.x == 8);
        expect(plan.offset.y == -12);
    };

    "density_map_offset_granularity_snapping"_test = [] {
        // Offsets must be integer multiples of fragmentDensityOffsetGranularity
        // (VUID-VkRenderPassFragmentDensityMapOffsetEndInfoEXT-x-06512 and
        // -y-06513), so an unaligned request is snapped instead of rejected.
        expect(snap_fragment_density_map_offset(0, 4u) == 0);
        expect(snap_fragment_density_map_offset(4, 4u) == 4);
        expect(snap_fragment_density_map_offset(5, 4u) == 4);
        expect(snap_fragment_density_map_offset(6, 4u) == 8);
        expect(snap_fragment_density_map_offset(-5, 4u) == -4);
        expect(snap_fragment_density_map_offset(-6, 4u) == -8);
        // A reported granularity of 0 means "no alignment requirement" and must
        // not divide by zero.
        expect(snap_fragment_density_map_offset(7, 0u) == 7);
        expect(fragment_density_map_offset_granularity_or_one(0u) == 1u);
        expect(fragment_density_map_offset_granularity_or_one(16u) == 16u);
        // Saturating, never overflowing.
        expect(snap_fragment_density_map_offset(INT32_MAX, 4u) == INT32_MAX);
        expect(snap_fragment_density_map_offset(INT32_MIN, 4u) == INT32_MIN);
        // A tie rounds away from zero, which lands exactly on INT32_MIN here and
        // must not wrap.
        expect(snap_fragment_density_map_offset(INT32_MIN + 1, 2u) == INT32_MIN);
        expect(snap_fragment_density_map_offset(INT32_MIN + 2, 2u) == INT32_MIN + 2);

        auto plan = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/1u,
            DensityMapOffset{3, -3},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/1u);
        expect(same(plan.status, FragmentDensityMapOffsetStatus::ACTIVE));
        expect(plan.snapped);
        expect(plan.offset.x == 4);
        expect(plan.offset.y == -4);
    };

    "density_map_offset_snaps_to_implicit_zero"_test = [] {
        // (0, 0) is what the driver uses when no end info is chained at all, so
        // an offset that snaps to it needs no structure.
        auto plan = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/1u,
            DensityMapOffset{1, -1},
            /*granularity_width=*/8u,
            /*granularity_height=*/8u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/1u);
        expect(same(plan.status, FragmentDensityMapOffsetStatus::NO_OFFSET_REQUESTED));
        expect(!plan.chain_end_info);
        expect(plan.snapped);
        expect(plan.offset.x == 0);
        expect(plan.offset.y == 0);
    };

    "density_map_offset_count_limits"_test = [] {
        // VUID-VkRenderPassFragmentDensityMapOffsetEndInfoEXT-
        // fragmentDensityOffsetCount-06511: the count is 1 without multiview and
        // the fragment density map attachment's layerCount with it.
        auto without_multiview = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/4u,
            DensityMapOffset{4, 0},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/false,
            /*fragment_density_map_layer_count=*/4u);
        expect(same(without_multiview.status, FragmentDensityMapOffsetStatus::ACTIVE));
        expect(without_multiview.offset_count == 1u);
        expect(without_multiview.count_clamped);
        auto with_multiview = plan_fragment_density_map_offset(
            /*extension_enabled=*/true,
            /*density_map_attachment_present=*/true,
            /*declared_offset_count=*/8u,
            DensityMapOffset{4, 0},
            /*granularity_width=*/4u,
            /*granularity_height=*/4u,
            /*multiview=*/true,
            /*fragment_density_map_layer_count=*/4u);
        expect(with_multiview.offset_count == 4u);
        expect(with_multiview.count_clamped);
        expect(with_multiview.chain_end_info);
    };

    // -----------------------------------------------------------------------
    // VK_KHR_pipeline_library_group_handles
    // -----------------------------------------------------------------------
    "pipeline_library_group_handles_disabled"_test = [] {
        auto plan = plan_pipeline_library_group_handles(
            /*extension_enabled=*/false,
            /*ray_tracing_pipeline_enabled=*/true,
            /*builds_pipeline_libraries=*/true);
        expect(same(plan.status,
                    PipelineLibraryGroupHandleStatus::DISABLED_EXTENSION));
        expect(!plan.request_feature);
        expect(!plan.link_pipeline_libraries);
    };

    "pipeline_library_group_handles_without_ray_tracing"_test = [] {
        auto plan = plan_pipeline_library_group_handles(
            /*extension_enabled=*/true,
            /*ray_tracing_pipeline_enabled=*/false,
            /*builds_pipeline_libraries=*/true);
        expect(same(plan.status,
                    PipelineLibraryGroupHandleStatus::NO_RAY_TRACING_PIPELINE));
        // The feature bit is still requested so the logical device matches the
        // negotiated status, but nothing uses it.
        expect(plan.request_feature);
        expect(!plan.link_pipeline_libraries);
    };

    "pipeline_library_group_handles_without_library"_test = [] {
        // The backend creates monolithic ray tracing pipelines (no
        // VK_PIPELINE_CREATE_LIBRARY_BIT_KHR), so there is no library group whose
        // handles could be queried.
        auto plan = plan_pipeline_library_group_handles(
            /*extension_enabled=*/true,
            /*ray_tracing_pipeline_enabled=*/true,
            /*builds_pipeline_libraries=*/false);
        expect(same(plan.status,
                    PipelineLibraryGroupHandleStatus::NO_PIPELINE_LIBRARY));
        expect(plan.request_feature);
        expect(!plan.link_pipeline_libraries);
    };

    "pipeline_library_group_handles_active"_test = [] {
        auto plan = plan_pipeline_library_group_handles(
            /*extension_enabled=*/true,
            /*ray_tracing_pipeline_enabled=*/true,
            /*builds_pipeline_libraries=*/true);
        expect(same(plan.status, PipelineLibraryGroupHandleStatus::ACTIVE));
        expect(plan.request_feature);
        expect(plan.link_pipeline_libraries);
    };

    // -----------------------------------------------------------------------
    // VK_EXT_multisampled_render_to_swapchain
    // -----------------------------------------------------------------------
    "multisampled_render_to_swapchain_disabled"_test = [] {
        auto plan = plan_multisampled_render_to_swapchain(
            /*extension_enabled=*/false,
            /*single_sampled_extension_enabled=*/true,
            /*renders_to_swapchain_image=*/true,
            /*rasterization_samples=*/4u);
        expect(same(plan.status,
                    MultisampledRenderToSwapchainStatus::DISABLED_EXTENSION));
        expect(!plan.create_swapchain_with_flag);
        expect(!plan.set_subpass_msrtss_enable);
    };

    "multisampled_render_to_swapchain_missing_single_sampled"_test = [] {
        // The swapchain flag is only usable together with
        // VK_EXT_multisampled_render_to_single_sampled, which contributes the
        // subpass structure.
        auto plan = plan_multisampled_render_to_swapchain(
            /*extension_enabled=*/true,
            /*single_sampled_extension_enabled=*/false,
            /*renders_to_swapchain_image=*/true,
            /*rasterization_samples=*/4u);
        expect(same(plan.status,
                    MultisampledRenderToSwapchainStatus::MISSING_SINGLE_SAMPLED_EXTENSION));
        expect(!plan.create_swapchain_with_flag);
        expect(!plan.set_subpass_msrtss_enable);
    };

    "multisampled_render_to_swapchain_needs_swapchain_attachment"_test = [] {
        auto plan = plan_multisampled_render_to_swapchain(
            /*extension_enabled=*/true,
            /*single_sampled_extension_enabled=*/true,
            /*renders_to_swapchain_image=*/false,
            /*rasterization_samples=*/4u);
        expect(same(plan.status,
                    MultisampledRenderToSwapchainStatus::NO_SWAPCHAIN_ATTACHMENT));
        expect(!plan.create_swapchain_with_flag);
    };

    "multisampled_render_to_swapchain_single_sampled"_test = [] {
        // The raster path creates every attachment with
        // VK_SAMPLE_COUNT_1_BIT, i.e. there is no multisampled subpass to
        // downgrade.
        auto plan = plan_multisampled_render_to_swapchain(
            /*extension_enabled=*/true,
            /*single_sampled_extension_enabled=*/true,
            /*renders_to_swapchain_image=*/true,
            /*rasterization_samples=*/1u);
        expect(same(plan.status, MultisampledRenderToSwapchainStatus::SINGLE_SAMPLED));
        expect(!plan.create_swapchain_with_flag);
        expect(!plan.set_subpass_msrtss_enable);
        expect(plan.rasterization_samples == 1u);
    };

    "multisampled_render_to_swapchain_active"_test = [] {
        auto plan = plan_multisampled_render_to_swapchain(
            /*extension_enabled=*/true,
            /*single_sampled_extension_enabled=*/true,
            /*renders_to_swapchain_image=*/true,
            /*rasterization_samples=*/4u);
        expect(same(plan.status, MultisampledRenderToSwapchainStatus::ACTIVE));
        expect(plan.create_swapchain_with_flag);
        expect(plan.set_subpass_msrtss_enable);
        // The attachment stays single-sampled; only the rasterisation uses four
        // samples.
        expect(plan.rasterization_samples == 4u);
    };

    // -----------------------------------------------------------------------
    // Negotiated gates and status names
    // -----------------------------------------------------------------------
    "raster_extension_gates"_test = [] {
        auto gates = RasterExtensionGates{};
        expect(!gates.any());
        gates.depth_clamp_zero_one = true;
        expect(gates.any());
        auto all = RasterExtensionGates{
            .custom_resolve = true,
            .fragment_density_map_offset = true,
            .primitive_restart_index = true,
            .pipeline_library_group_handles = true,
            .depth_clamp_zero_one = true,
            .multisampled_render_to_swapchain = true};
        expect(all.any());
    };

    "status_names"_test = [] {
        expect(non_empty_text(primitive_restart_index_status_name(
            PrimitiveRestartIndexStatus::NO_RESTART)));
        expect(non_empty_text(primitive_restart_index_status_name(
            PrimitiveRestartIndexStatus::DYNAMIC_EXTENSION)));
        expect(non_empty_text(primitive_restart_index_status_name(
            PrimitiveRestartIndexStatus::STATIC_EQUIVALENT)));
        expect(non_empty_text(primitive_restart_index_status_name(
            PrimitiveRestartIndexStatus::STATIC_TYPE_DEFAULT_UNHONOURED)));
        expect(non_empty_text(depth_clamp_status_name(DepthClampStatus::DISABLED)));
        expect(non_empty_text(depth_clamp_status_name(
            DepthClampStatus::MISSING_FEATURE)));
        expect(non_empty_text(depth_clamp_status_name(
            DepthClampStatus::CLAMP_VIEWPORT_RANGE)));
        expect(non_empty_text(depth_clamp_status_name(
            DepthClampStatus::CLAMP_ZERO_ONE)));
        expect(non_empty_text(depth_clamp_status_name(
            DepthClampStatus::ZERO_ONE_NO_CLAMP)));
        expect(non_empty_text(custom_resolve_status_name(
            CustomResolveStatus::NO_RESOLVE_ATTACHMENT)));
        expect(non_empty_text(custom_resolve_status_name(
            CustomResolveStatus::NO_DYNAMIC_RENDERING)));
        expect(non_empty_text(custom_resolve_status_name(
            CustomResolveStatus::SINGLE_SAMPLED)));
        expect(non_empty_text(custom_resolve_status_name(
            CustomResolveStatus::ACTIVE)));
        expect(non_empty_text(fragment_density_map_offset_status_name(
            FragmentDensityMapOffsetStatus::NO_DENSITY_MAP_ATTACHMENT)));
        expect(non_empty_text(fragment_density_map_offset_status_name(
            FragmentDensityMapOffsetStatus::NO_OFFSET_REQUESTED)));
        expect(non_empty_text(fragment_density_map_offset_status_name(
            FragmentDensityMapOffsetStatus::ACTIVE)));
        expect(non_empty_text(pipeline_library_group_handle_status_name(
            PipelineLibraryGroupHandleStatus::NO_PIPELINE_LIBRARY)));
        expect(non_empty_text(multisampled_render_to_swapchain_status_name(
            MultisampledRenderToSwapchainStatus::MISSING_SINGLE_SAMPLED_EXTENSION)));
        expect(non_empty_text(multisampled_render_to_swapchain_status_name(
            MultisampledRenderToSwapchainStatus::ACTIVE)));
        // A spot check that the strings are the ones the backend logs.
        expect(same_text(depth_clamp_status_name(DepthClampStatus::CLAMP_ZERO_ONE),
                         "clamp [0,1]"));
        expect(same_text(
            custom_resolve_status_name(CustomResolveStatus::NO_RESOLVE_ATTACHMENT),
            "draw has no resolve attachment"));
    };
}
