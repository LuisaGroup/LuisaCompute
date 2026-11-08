#pragma once

#include <cstddef>
#include <cstdint>

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

}// namespace lc::vk
