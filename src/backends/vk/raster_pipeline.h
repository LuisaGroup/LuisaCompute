#pragma once

#include <cstddef>
#include <cstdint>

#include "raster_extension_plan.h"

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/rhi/argument.h>
#include <volk.h>

namespace lc::vk {

// Every other header of this backend opens its `lc::vk` scope with the same
// using-directives; the declarations below are spelled with `luisa::uint` etc.
// and this header does not include a sibling that would provide them.
using namespace luisa;
using namespace luisa::compute;

class Device;

// A graphics pipeline and the render pass it was built for. Both are owned by
// the raster shader that created them (they are destroyed with it), so a cached
// hit simply hands the same pair out again.
struct RasterPipelineHandle {
    VkPipeline pipeline{};
    VkRenderPass render_pass{};
};

// Creates the render pass of one raster draw: one attachment per render-target
// view, plus the depth/stencil attachment when `dsv_texture` is valid. The
// attachment load/store/layout choices come from `state`'s blend and depth
// configuration, so a draw that blends, or that reads its depth buffer, needs
// its own render pass - it is part of the pipeline cache key for that reason.
//
// The caller owns the returned render pass (`vkDestroyRenderPass`).
[[nodiscard]] VkRenderPass create_raster_render_pass(
    Device *device,
    luisa::compute::RasterState const &state,
    luisa::span<luisa::compute::Argument::Texture const> rtv_textures,
    luisa::compute::Argument::Texture dsv_texture);

// Builds the byte blob that identifies everything about a raster draw that is
// baked into the graphics pipeline (Vulkan has no equivalent of D3D12's dynamic
// state for these):
//
//  * the vertex input layout derived from `mesh_format`, i.e. one attribute
//    description per declared attribute - at `location = attribute type`, with
//    the offsets implied by the attribute order - and one binding description
//    per vertex stream, carrying that stream's `strides` entry,
//  * the whole `RasterState` (fill mode, cull mode, topology, front face, depth
//    clip, blend, depth and stencil state),
//  * the render-target formats and the depth format, because the render pass -
//    and therefore the pipeline - is format specific.
//
// `vertex_input_create_info` is filled in and its pointers alias the returned
// blob, so the blob must outlive any use of the create-info. Two draws with
// equal blobs are interchangeable for the same shader.
[[nodiscard]] luisa::vector<std::byte> build_raster_pipeline_key(
    luisa::compute::MeshFormat const &mesh_format,
    luisa::compute::RasterState const &state,
    luisa::span<luisa::compute::Argument::Texture const> rtv_textures,
    luisa::compute::Argument::Texture dsv_texture,
    luisa::span<uint const> strides,
    VkPipelineVertexInputStateCreateInfo &vertex_input_create_info);

// The graphics-pipeline state that the optional raster extensions change,
// resolved once per draw so the DSL raster path (`RasterShader`) and the native
// raster path (`NativeRasterShader`) cannot disagree. Every field is a value the
// caller copies into the `VkGraphicsPipelineCreateInfo` it is about to build;
// nothing here records a command.
//
// The plans are the pure decisions of `raster_extension_plan.h`; this struct
// only adds what is Vulkan-typed (`VkDynamicState`, `VkBool32`) and reads the
// negotiated gates off the device. With every optional extension switched off -
// the default on an imported device or with the switches turned off - the result
// is exactly the pre-extension state: `depthClampEnable` unchanged and the
// dynamic-state list still limited to viewport and scissor.
struct RasterPipelineExtensionPlan {
    // VkPipelineRasterizationStateCreateInfo::depthClampEnable.
    VkBool32 depth_clamp_enable{VK_FALSE};
    // VkPipelineDynamicStateCreateInfo list. Viewport and scissor are always
    // dynamic; none of the six extensions adds a further entry today, so the
    // list stays exactly what it was before they existed.
    luisa::fixed_vector<VkDynamicState, 4> dynamic_states;
    detail::DepthClampPlan depth_clamp;
    detail::PrimitiveRestartIndexPlan primitive_restart;
    // The four extensions of this group whose fast path cannot be entered at
    // all, because the public raster API (`DrawRasterSceneCommand`, `RasterState`,
    // `RasterMesh`, the swapchain) cannot express what they need - see the
    // per-extension note in `vk_raster_ext.cpp`. They are still *planned* here,
    // from the negotiated gate and the draw's real attachment/sample state, so
    // the "enabled but inactive" decision is made at the one place that decides
    // raster pipeline state instead of being an implicit absence of code. Their
    // status is never ACTIVE with the current public API, and none of them
    // contributes a field to the pipeline; `plan_*` in raster_extension_plan.h
    // documents the exact argument that would have to change.
    detail::CustomResolvePlan custom_resolve;
    detail::FragmentDensityMapOffsetPlan density_map_offset;
    detail::PipelineLibraryGroupHandlePlan pipeline_library_group_handles;
    detail::MultisampledRenderToSwapchainPlan multisampled_render_to_swapchain;
    // The pipeline really renders through VK_KHR_depth_clamp_zero_one, so the
    // caller may report that extension as active.
    bool uses_depth_clamp_zero_one{false};
    // The draw recording must record vkCmdSetPrimitiveRestartIndexEXT after
    // binding its index buffer.
    bool uses_primitive_restart_index{false};
};

// `index_buffer_present` / `requested_restart_index` describe the draw the
// pipeline is created for. The public raster API has no restart index, so the
// backend always passes the type-derived default and "restart disabled", which
// is what keeps the extension inactive; the parameters exist so the plan stays
// correct when that changes.
[[nodiscard]] RasterPipelineExtensionPlan plan_raster_pipeline_extensions(
    Device *device,
    luisa::compute::RasterState const &state,
    bool index_buffer_present = true,
    uint32_t requested_restart_index = detail::primitive_restart_index_uint32);

// Records the custom primitive restart index of one indexed draw.
//
// Only legal when VK_EXT_primitive_restart_index's `primitiveRestartIndex`
// feature is enabled on the logical device, and it must be recorded *after*
// vkCmdBindIndexBuffer: "Binding an index buffer invalidates the custom index
// value" (Vulkan 1.4, vkCmdSetPrimitiveRestartIndexEXT), so a value recorded
// before the bind is silently replaced by the type-derived default. The raster
// draw recording lives in `stream.cpp`, which is why no current draw takes this
// path - see `plan_primitive_restart_index` for the gate that decides it.
void cmd_set_primitive_restart_index(VkCommandBuffer cmdbuffer,
                                     uint32_t restart_index) noexcept;

}// namespace lc::vk
