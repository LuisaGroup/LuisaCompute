#include "raster_pipeline.h"

#include <array>
#include <cstring>

#include "device.h"
#include "log.h"
#include "texture.h"

namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;

// The two helpers below were moved verbatim out of `RasterShader`
// (`RasterShader::create_render_pass` / `RasterShader::_make_pipeline_key`) so
// that the native raster shader (`NativeRasterShader`) can build its graphics
// pipelines with exactly the same render-pass and pipeline-key choices as the
// DSL raster path. `RasterShader` keeps its static member as a thin forwarder.

VkRenderPass create_raster_render_pass(
    Device *device,
    RasterState const &state,
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_textures) {
    VkAttachmentLoadOp color_attachment_load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp color_attachment_store_op = VK_ATTACHMENT_STORE_OP_STORE;
    VkImageLayout color_attachment_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool blend = false;
    if (state.blend_state.enable_blend) {
        switch (state.blend_state.img_op) {
            case BlendWeight::One:
            case BlendWeight::ImgColor:
            case BlendWeight::ImgAlpha:
            case BlendWeight::OneMinusImgColor:
            case BlendWeight::OneMinusImgAlpha:
                blend = true;
                break;
        }
        switch (state.blend_state.prim_op) {
            case BlendWeight::ImgColor:
            case BlendWeight::ImgAlpha:
            case BlendWeight::OneMinusImgColor:
            case BlendWeight::OneMinusImgAlpha:
                blend = true;
                break;
        }
    }
    // Samples can keep the color attachment contents, e.g. if they have previously written to the swap chain images
    if (blend) {
        color_attachment_load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
        color_attachment_image_layout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
    }
    luisa::fixed_vector<VkAttachmentDescription, 9> attachments;
    luisa::fixed_vector<VkAttachmentReference, 8> color_references;
    VkAttachmentReference depth_reference{};
    VkAttachmentReference *depth_reference_ptr{};
    for (auto &i : rtv_textures) {
        auto attach_idx = attachments.size();
        auto &a = attachments.emplace_back();
        auto tex = reinterpret_cast<Texture *>(i.handle);
        a.format = Texture::to_vk_format(tex->format());
        a.samples = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp = color_attachment_load_op;
        a.storeOp = color_attachment_store_op;
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = color_attachment_image_layout;
        a.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        auto &color_reference = color_references.emplace_back();
        color_reference.attachment = attach_idx;
        color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    // DSV
    if (dsv_textures.handle != invalid_resource_handle) {
        auto attach_idx = attachments.size();
        auto &a = attachments.emplace_back();
        auto tex = reinterpret_cast<Texture *>(dsv_textures.handle);
        a.format = Texture::to_vk_format(tex->format());
        a.samples = VK_SAMPLE_COUNT_1_BIT;
        if (state.depth_state.enable_depth) {
            a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            a.storeOp = state.depth_state.write ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            a.stencilStoreOp = state.depth_state.write ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        } else {
            a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        a.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        depth_reference.attachment = attach_idx;
        depth_reference.layout = a.finalLayout;
        depth_reference_ptr = &depth_reference;
    }
    VkSubpassDescription subpass_description = {};
    subpass_description.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_description.colorAttachmentCount = color_references.size();
    subpass_description.pColorAttachments = color_references.data();
    subpass_description.pDepthStencilAttachment = depth_reference_ptr;
    subpass_description.inputAttachmentCount = 0;
    subpass_description.pInputAttachments = nullptr;
    subpass_description.preserveAttachmentCount = 0;
    subpass_description.pPreserveAttachments = nullptr;
    subpass_description.pResolveAttachments = nullptr;
    std::array<VkSubpassDependency, 2> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_NONE_KHR;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo render_pass_create_info = {};
    render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    render_pass_create_info.pAttachments = attachments.data();
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &subpass_description;
    render_pass_create_info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    render_pass_create_info.pDependencies = dependencies.data();
    VkRenderPass render_pass;
    VK_CHECK_RESULT(vkCreateRenderPass(device->logic_device(), &render_pass_create_info, Device::alloc_callbacks(), &render_pass));
    return render_pass;
}

luisa::vector<std::byte> build_raster_pipeline_key(
    MeshFormat const &mesh_format,
    RasterState const &state,
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_texture,
    luisa::span<uint const> strides,
    VkPipelineVertexInputStateCreateInfo &vertex_input_create_info) {
    size_t key_size = 0;
    vertex_input_create_info = VkPipelineVertexInputStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    // compute size
    {
        // vertices
        vertex_input_create_info.vertexAttributeDescriptionCount = mesh_format.vertex_attribute_count();
        vertex_input_create_info.vertexBindingDescriptionCount = mesh_format.vertex_stream_count();
        key_size += vertex_input_create_info.vertexAttributeDescriptionCount * sizeof(VkVertexInputAttributeDescription) + vertex_input_create_info.vertexBindingDescriptionCount * sizeof(VkVertexInputBindingDescription);
        key_size += sizeof(RasterState);
        // The render pass (and therefore the pipeline) depends on the
        // attachment formats; a draw with a different color/depth format
        // must not reuse a pipeline cached for another one.
        key_size += rtv_textures.size() * sizeof(VkFormat);
        key_size += sizeof(VkFormat);
    }
    if (key_size == 0) [[unlikely]]
        return {};
    // Value-initialised so that the blob carries the same bytes as the
    // zero-filled buffer this key was built in before the refactor: hashing and
    // comparison run over every byte, including any the writer does not touch.
    luisa::vector<std::byte> result(key_size, std::byte{0u});

    // push data
    {
        std::byte *ptr = result.data();
        vertex_input_create_info.pVertexAttributeDescriptions = reinterpret_cast<const VkVertexInputAttributeDescription *>(ptr);
        for (auto stream_idx : vstd::range((int64_t)mesh_format.vertex_stream_count())) {
            uint32_t offset = 0;
            for (auto &&attrs : mesh_format.attributes(stream_idx)) {
                // Vertex input locations follow the canonical semantic order
                // used by raster codegen (position, normal, tangent, color,
                // uv0..uv3), not the order the attributes happen to appear
                // in the stream, so a mesh format may list its attributes
                // in any order like on the DX/Metal backends.
                VkVertexInputAttributeDescription desc{
                    .location = static_cast<uint32_t>(attrs.type),
                    .binding = (uint32_t)stream_idx,
                    .format = Texture::to_vk_format(attrs.format),
                    .offset = offset};
                offset += pixel_format_size(attrs.format, uint3(1));
                std::memcpy(ptr, &desc, sizeof(desc));
                ptr += sizeof(desc);
            }
        }
        vertex_input_create_info.pVertexBindingDescriptions = reinterpret_cast<const VkVertexInputBindingDescription *>(ptr);
        LUISA_ASSERT(
            strides.size() == mesh_format.vertex_stream_count(),
            "Raster pipeline requires one vertex stride per mesh stream (got {}, expected {}).",
            strides.size(), mesh_format.vertex_stream_count());
        for (auto stream_idx : vstd::range((int64_t)mesh_format.vertex_stream_count())) {
            VkVertexInputBindingDescription desc{
                .binding = (uint32_t)stream_idx,
                .stride = strides[stream_idx],
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
            std::memcpy(ptr, &desc, sizeof(desc));
            ptr += sizeof(desc);
        }
        std::memcpy(ptr, &state, sizeof(state));
        ptr += sizeof(RasterState);
        for (auto &i : rtv_textures) {
            auto tex = reinterpret_cast<Texture const *>(i.handle);
            auto format = Texture::to_vk_format(tex->format());
            std::memcpy(ptr, &format, sizeof(format));
            ptr += sizeof(format);
        }
        auto dsv_format = dsv_texture.handle != invalid_resource_handle ?
                              Texture::to_vk_format(reinterpret_cast<Texture const *>(dsv_texture.handle)->format()) :
                              VK_FORMAT_UNDEFINED;
        std::memcpy(ptr, &dsv_format, sizeof(dsv_format));
        ptr += sizeof(dsv_format);
        LUISA_ASSERT(ptr == result.data() + result.size());
    }
    return result;
}

RasterPipelineExtensionPlan plan_raster_pipeline_extensions(
    Device *device,
    RasterState const &state,
    bool index_buffer_present,
    uint32_t requested_restart_index) {
    RasterPipelineExtensionPlan plan;
    // Only `optional_extension_enabled()` may open a fast path: `supported`
    // alone means the driver knows the extension, and an imported logical device
    // reports every entry as disabled because Vulkan cannot attest what a
    // borrowed VkDevice was created with. Every decision below reads this one
    // snapshot, so an extension can never be half-gated.
    auto gates = detail::RasterExtensionGates{
        .custom_resolve = device->optional_extension_enabled(
            detail::OptionalExtension::CUSTOM_RESOLVE),
        .fragment_density_map_offset = device->optional_extension_enabled(
            detail::OptionalExtension::FRAGMENT_DENSITY_MAP_OFFSET),
        .primitive_restart_index = device->optional_extension_enabled(
            detail::OptionalExtension::PRIMITIVE_RESTART_INDEX),
        .pipeline_library_group_handles = device->optional_extension_enabled(
            detail::OptionalExtension::PIPELINE_LIBRARY_GROUP_HANDLES),
        .depth_clamp_zero_one = device->optional_extension_enabled(
            detail::OptionalExtension::DEPTH_CLAMP_ZERO_ONE),
        .multisampled_render_to_swapchain = device->optional_extension_enabled(
            detail::OptionalExtension::MULTISAMPLED_RENDER_TO_SWAPCHAIN),
    };
    // ---- depth clamping -----------------------------------------------------
    // VK_EXT_depth_clamp_control, which contributes the user-defined
    // [0, 1] range (VkPipelineViewportDepthClampControlCreateInfoEXT /
    // VkDepthClampRangeEXT), is not an entry of the shared optional-extension
    // registry, so that pNext chain may never be used here; the [0, 1] range
    // this backend asks for is the one VK_KHR_depth_clamp_zero_one applies to
    // the fragment depth itself.
    constexpr auto depth_clamp_control_available = false;
    plan.depth_clamp = detail::plan_depth_clamp(
        state.depth_clip,
        device->enabled_features().depthClamp == VK_TRUE,
        gates.depth_clamp_zero_one,
        depth_clamp_control_available);
    plan.depth_clamp_enable =
        plan.depth_clamp.depth_clamp_enable ? VK_TRUE : VK_FALSE;
    // The feature is device-wide, so every raster pipeline created on such a
    // device really renders through it.
    plan.uses_depth_clamp_zero_one = plan.depth_clamp.zero_one_clamp;
    // ---- primitive restart index -------------------------------------------
    // `RasterState` has list topologies only and Vulkan forbids primitive
    // restart for those without VK_EXT_primitive_topology_list_restart
    // (VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252), and no
    // public API carries a custom index, so nothing ever asks for one today.
    // The plan still decides the mechanism, so a future restart/index field only
    // has to change these two inputs.
    constexpr auto state_requests_primitive_restart = false;
    // A draw without an index buffer compares no index value at all.
    auto primitive_restart_enabled =
        index_buffer_present && state_requests_primitive_restart;
    // A pipeline may only record the custom index when the draw recording can
    // re-record it after vkCmdBindIndexBuffer; otherwise the value is discarded
    // by that bind. The raster draw loop lives in `stream.cpp`, which does not
    // call `cmd_set_primitive_restart_index` yet, so the recording point does
    // not exist.
    constexpr auto restart_index_recordable = false;
    plan.primitive_restart = detail::plan_primitive_restart_index(
        gates.primitive_restart_index,
        restart_index_recordable,
        primitive_restart_enabled,
        requested_restart_index,
        // Every RasterMesh index buffer is a BufferView<uint>, i.e.
        // VK_INDEX_TYPE_UINT32.
        detail::primitive_restart_index_uint32);
    plan.uses_primitive_restart_index = plan.primitive_restart.record_command;
    // ---- extensions whose fast path this backend cannot enter --------------
    // Each of the four plans below is computed from the real negotiated gate so
    // that the decision lives next to the pipeline state it would have changed.
    // The arguments after the gate are the ones the public API cannot supply;
    // `vk_raster_ext.cpp` names the missing API for each of them, and
    // `plan_custom_resolve` / `plan_fragment_density_map_offset` /
    // `plan_pipeline_library_group_handles` /
    // `plan_multisampled_render_to_swapchain` receive them, so making one of
    // these extensions usable means changing exactly one argument here.
    //
    //  * VK_EXT_custom_resolve: `DrawRasterSceneCommand` has render targets and
    //    one depth buffer, no resolve attachment; the raster path records
    //    vkCmdBeginRenderPass (not vkCmdBeginRendering); and every attachment is
    //    created with VK_SAMPLE_COUNT_1_BIT. The fallback is the current
    //    `pResolveAttachments = nullptr` render pass.
    plan.custom_resolve = detail::plan_custom_resolve(
        gates.custom_resolve,
        /*command_has_resolve_attachment=*/false,
        /*record_path_is_dynamic_rendering=*/false,
        /*rasterization_samples=*/1u);
    //  * VK_EXT_fragment_density_map_offset: no fragment density map attachment
    //    (VK_EXT_fragment_density_map is not negotiated), so
    //    fragmentDensityOffsetCount must be 0 (VUID-...-fragmentDensityMapOffsets-06503).
    //    The fallback is the implicit (0, 0) offset the end info would describe.
    plan.density_map_offset = detail::plan_fragment_density_map_offset(
        gates.fragment_density_map_offset,
        /*density_map_attachment_present=*/false,
        /*declared_offset_count=*/0u,
        detail::DensityMapOffset{},
        /*granularity_width=*/1u,
        /*granularity_height=*/1u,
        /*multiview=*/false,
        /*fragment_density_map_layer_count=*/0u);
    //  * VK_KHR_pipeline_library_group_handles: only a ray tracing pipeline that
    //    links VK_KHR_pipeline_library libraries can query group handles, and
    //    `rt_shader.cpp` builds monolithic pipelines. The fallback is the
    //    existing monolithic ray tracing pipeline.
    plan.pipeline_library_group_handles =
        detail::plan_pipeline_library_group_handles(
            gates.pipeline_library_group_handles,
            device->enable_raytracing(),
            /*builds_pipeline_libraries=*/false);
    //  * VK_EXT_multisampled_render_to_swapchain: needs the swapchain create
    //    flag (`swapchain.cpp`), VK_EXT_multisampled_render_to_single_sampled,
    //    a multisampled subpass and a swapchain-image attachment. The fallback is
    //    the current single-sampled attachment.
    plan.multisampled_render_to_swapchain =
        detail::plan_multisampled_render_to_swapchain(
            gates.multisampled_render_to_swapchain,
            /*single_sampled_extension_enabled=*/false,
            /*renders_to_swapchain_image=*/false,
            /*rasterization_samples=*/1u);
    // ---- dynamic state -----------------------------------------------------
    // Neither of these two extensions contributes a VkDynamicState entry:
    // vkCmdSetPrimitiveRestartIndexEXT overrides command-buffer state without
    // being pipeline dynamic state, and VK_KHR_depth_clamp_zero_one is a device
    // feature. The list is therefore unchanged from before the extensions
    // existed, in both the enabled and the disabled case.
    plan.dynamic_states.emplace_back(VK_DYNAMIC_STATE_VIEWPORT);
    plan.dynamic_states.emplace_back(VK_DYNAMIC_STATE_SCISSOR);
    return plan;
}

void cmd_set_primitive_restart_index(VkCommandBuffer cmdbuffer,
                                     uint32_t restart_index) noexcept {
    vkCmdSetPrimitiveRestartIndexEXT(cmdbuffer, restart_index);
}

}// namespace lc::vk
