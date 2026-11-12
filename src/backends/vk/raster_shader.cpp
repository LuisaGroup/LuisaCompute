#include "raster_shader.h"
#include "device.h"
#include "log.h"
#include "raster_pipeline.h"
namespace lc::vk {
RasterShader::RasterShader(
    Device *device,
    vstd::vector<Argument> &&captured,
    vstd::vector<SavedArgument> &&saved_arguments,
    vstd::span<hlsl::Property const> binds,
    vstd::span<std::byte const> cache_code,
    vstd::vector<uint> &&vertex_spv_code,
    vstd::vector<uint> &&pixel_spv_code,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless,
    uint validation_count,
    detail::ShaderCodegenDialect codegen_dialect)
    : Shader(device, ShaderTag::kRasterShader, std::move(captured), std::move(saved_arguments), binds, use_tex2d_bindless, use_tex3d_bindless, use_buffer_bindless, {}, {}, validation_count, 32u, codegen_dialect), _vertex_spv_code(std::move(vertex_spv_code)), _pixel_spv_code(std::move(pixel_spv_code)) {
    VkPipelineCacheCreateInfo pso_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (!cache_code.empty()) {
        pso_ci.initialDataSize = cache_code.size();
        pso_ci.pInitialData = cache_code.data();
    }
    VK_CHECK_RESULT(vkCreatePipelineCache(device->logic_device(), &pso_ci, Device::alloc_callbacks(), &_pipe_cache));
}
RasterShader::~RasterShader() {
    for (auto &i : _pipelines) {
        vkDestroyPipeline(device()->logic_device(), i.second.pipeline, Device::alloc_callbacks());
        vkDestroyRenderPass(device()->logic_device(), i.second.render_pass, Device::alloc_callbacks());
    }
    vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, Device::alloc_callbacks());
}
auto RasterShader::_make_pipeline_key(
    luisa::compute::MeshFormat const &mesh_format,
    RasterState const &state,
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_texture,
    luisa::span<uint const> strides,
    VkPipelineVertexInputStateCreateInfo &vertex_input_create_info)
    -> luisa::vector<std::byte> {
    // The body lives in raster_pipeline.cpp so that the native raster shader
    // builds its pipeline cache keys exactly like the DSL raster path.
    return build_raster_pipeline_key(
        mesh_format, state, rtv_textures, dsv_texture, strides,
        vertex_input_create_info);
}
auto RasterShader::create_pipeline(
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_textures,
    luisa::compute::MeshFormat const &mesh_format,
    RasterState const &state,
    luisa::span<uint const> strides) -> Pipeline {
    VkPrimitiveTopology topo;
    VkCullModeFlags cull_mode;
    VkPipelineVertexInputStateCreateInfo vertex_input;
    auto binary_blob = _make_pipeline_key(
        mesh_format,
        state,
        rtv_textures,
        dsv_textures,
        strides,
        vertex_input);
    auto iter = _pipelines.try_emplace(std::move(binary_blob));
    auto &v = iter.first.value();
    if (!iter.second) return v;
    switch (state.cull_mode) {
        case CullMode::None:
            cull_mode = VK_CULL_MODE_NONE;
            break;
        case CullMode::Back:
            cull_mode = VK_CULL_MODE_BACK_BIT;
            break;
        case CullMode::Front:
            cull_mode = VK_CULL_MODE_FRONT_BIT;
            break;
    }
    switch (state.topology) {
        case TopologyType::Point:
            topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            break;
        case TopologyType::Line:
            topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            break;
        case TopologyType::Triangle:
            topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
    }
    VkPipelineInputAssemblyStateCreateInfo input_asm_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topo,
        .primitiveRestartEnable = VK_FALSE};

    // Depth clamping (VK_KHR_depth_clamp_zero_one) and the primitive restart
    // index (VK_EXT_primitive_restart_index) are decided by the shared planner,
    // so this path and the native raster path cannot disagree. With both
    // extensions disabled the plan reproduces the previous values exactly.
    auto extension_plan = plan_raster_pipeline_extensions(device(), state);
    VkPipelineRasterizationStateCreateInfo raster_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        // RasterState::depth_clip follows the D3D12 DepthClipEnable
        // semantics: enabled clips primitives outside [0, 1], which is the
        // Vulkan default (depth clamp off); disabled requests clamping,
        // which is only available when the depthClamp feature is enabled.
        // When VK_KHR_depth_clamp_zero_one is enabled the device additionally
        // clamps the fragment depth itself to [0, 1] on every draw, which is
        // the range D3D12 uses (`uses_depth_clamp_zero_one`).
        .depthClampEnable = extension_plan.depth_clamp_enable,
        .rasterizerDiscardEnable = false,
        .polygonMode = state.fill_mode == FillMode::Solid ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE,
        .cullMode = cull_mode,
        .frontFace = state.front_counter_clockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 1.f,
        .depthBiasSlopeFactor = 1.f,
        .lineWidth = 1.f};
    uint32_t sample_mask = -1;
    VkPipelineMultisampleStateCreateInfo ms_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0,
        .pSampleMask = &sample_mask,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE};
    auto get_blend_op = [](BlendOp op) {
        switch (op) {
            case BlendOp::Add:
                return VK_BLEND_OP_ADD;
            case BlendOp::Max:
                return VK_BLEND_OP_MAX;
            case BlendOp::Min:
                return VK_BLEND_OP_MIN;
            default:
                return VK_BLEND_OP_SUBTRACT;
        }
    };
    auto get_blend_factor = [](BlendWeight weight) {
        switch (weight) {
            case BlendWeight::Zero:
                return VK_BLEND_FACTOR_ZERO;
            case BlendWeight::One:
                return VK_BLEND_FACTOR_ONE;
            case BlendWeight::PrimColor:
                return VK_BLEND_FACTOR_SRC_COLOR;
            case BlendWeight::ImgColor:
                return VK_BLEND_FACTOR_DST_COLOR;
            case BlendWeight::PrimAlpha:
                return VK_BLEND_FACTOR_SRC_ALPHA;
            case BlendWeight::ImgAlpha:
                return VK_BLEND_FACTOR_DST_ALPHA;
            case BlendWeight::OneMinusPrimColor:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            case BlendWeight::OneMinusImgColor:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            case BlendWeight::OneMinusPrimAlpha:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            default:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        }
    };

    // Vulkan does not allow the *_COLOR blend factors on the alpha channel;
    // map them to their alpha counterparts like D3D12 does implicitly.
    auto get_alpha_blend_factor = [&get_blend_factor](BlendWeight weight) {
        switch (weight) {
            case BlendWeight::PrimColor:
                return VK_BLEND_FACTOR_SRC_ALPHA;
            case BlendWeight::ImgColor:
                return VK_BLEND_FACTOR_DST_ALPHA;
            case BlendWeight::OneMinusPrimColor:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            case BlendWeight::OneMinusImgColor:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
            default:
                return get_blend_factor(weight);
        }
    };

    VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = state.blend_state.enable_blend,
        .srcColorBlendFactor = get_blend_factor(state.blend_state.prim_op),
        .dstColorBlendFactor = get_blend_factor(state.blend_state.img_op),
        .colorBlendOp = get_blend_op(state.blend_state.op),
        .srcAlphaBlendFactor = get_alpha_blend_factor(state.blend_state.prim_op),
        .dstAlphaBlendFactor = get_alpha_blend_factor(state.blend_state.img_op),
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    luisa::vector<VkPipelineColorBlendAttachmentState> blend_attachments(rtv_textures.size(), blend_attachment);
    VkPipelineColorBlendStateCreateInfo blend_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_CLEAR,
        .attachmentCount = static_cast<uint32_t>(blend_attachments.size()),
        .pAttachments = blend_attachments.data(),
        .blendConstants = {1.f, 1.f, 1.f, 1.f}};
    VkViewport view;
    VkRect2D scissors;
    VkPipelineViewportStateCreateInfo viewport_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &view,
        .scissorCount = 1,
        .pScissors = &scissors};
    VkPipelineTessellationStateCreateInfo tess_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .patchControlPoints = 1};
    auto get_compare_op = [](Comparison const &comp) {
        switch (comp) {
            case Comparison::Never:
                return VK_COMPARE_OP_NEVER;
            case Comparison::Less:
                return VK_COMPARE_OP_LESS;
            case Comparison::Equal:
                return VK_COMPARE_OP_EQUAL;
            case Comparison::LessEqual:
                return VK_COMPARE_OP_LESS_OR_EQUAL;
            case Comparison::Greater:
                return VK_COMPARE_OP_GREATER;
            case Comparison::NotEqual:
                return VK_COMPARE_OP_NOT_EQUAL;
            case Comparison::GreaterEqual:
                return VK_COMPARE_OP_GREATER_OR_EQUAL;
            default:
                return VK_COMPARE_OP_ALWAYS;
        }
    };
    auto get_stencil_state = [](StencilOp s) {
        switch (s) {
            case StencilOp::Keep:
                return VK_STENCIL_OP_KEEP;
            case StencilOp::Replace:
                return VK_STENCIL_OP_REPLACE;
            default:
                return VK_STENCIL_OP_ZERO;
        }
    };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = state.depth_state.enable_depth,
        .depthWriteEnable = state.depth_state.write,
        .depthCompareOp = get_compare_op(state.depth_state.comparison),
        // No depth-bounds test is exposed through RasterState, and the
        // depthBounds feature is not enabled on the logical device.
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = state.stencil_state.enable_stencil,
        .front = VkStencilOpState{
            .failOp = get_stencil_state(state.stencil_state.front_face_op.stencil_fail_op),
            .passOp = get_stencil_state(state.stencil_state.front_face_op.pass_op),
            .depthFailOp = get_stencil_state(state.stencil_state.front_face_op.depth_fail_op),
            .compareOp = get_compare_op(state.stencil_state.front_face_op.comparison),
            .compareMask = state.stencil_state.read_mask,
            .writeMask = state.stencil_state.write_mask,
            .reference = state.stencil_state.reference,
        },
        .back = VkStencilOpState{
            .failOp = get_stencil_state(state.stencil_state.back_face_op.stencil_fail_op),
            .passOp = get_stencil_state(state.stencil_state.back_face_op.pass_op),
            .depthFailOp = get_stencil_state(state.stencil_state.back_face_op.depth_fail_op),
            .compareOp = get_compare_op(state.stencil_state.back_face_op.comparison),
            .compareMask = state.stencil_state.read_mask,
            .writeMask = state.stencil_state.write_mask,
            .reference = state.stencil_state.reference,
        },
        .minDepthBounds = 0.f,
        .maxDepthBounds = 1.f,
    };
    VkPipelineDynamicStateCreateInfo dynamic_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(
            extension_plan.dynamic_states.size()),
        .pDynamicStates = extension_plan.dynamic_states.data()};
    VkShaderModule vertex_shader_module;
    VkShaderModule pixel_shader_module;
    {
        VkShaderModuleCreateInfo module_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = luisa::size_bytes(_vertex_spv_code),
            .pCode = _vertex_spv_code.data()};
        VK_CHECK_RESULT(vkCreateShaderModule(device()->logic_device(), &module_create_info, Device::alloc_callbacks(), &vertex_shader_module));
    }
    {
        VkShaderModuleCreateInfo module_create_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = luisa::size_bytes(_pixel_spv_code),
            .pCode = _pixel_spv_code.data()};
        VK_CHECK_RESULT(vkCreateShaderModule(device()->logic_device(), &module_create_info, Device::alloc_callbacks(), &pixel_shader_module));
    }
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(device()->logic_device(), vertex_shader_module, Device::alloc_callbacks());
        vkDestroyShaderModule(device()->logic_device(), pixel_shader_module, Device::alloc_callbacks());
    });
    VkPipelineShaderStageCreateInfo shader_stages[] = {
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main"},
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = pixel_shader_module,
            .pName = "main"}};
    v.render_pass = create_render_pass(device(), state, rtv_textures, dsv_textures);
    VkGraphicsPipelineCreateInfo graphics_pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = vstd::array_count(shader_stages),
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_asm_info,
        .pTessellationState = &tess_info,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_info,
        .pMultisampleState = &ms_info,
        .pDepthStencilState = &depth_stencil_info,
        .pColorBlendState = &blend_info,
        .pDynamicState = &dynamic_info,
        .layout = pipeline_layout(),
        .renderPass = v.render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = ~0};

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device()->logic_device(), _pipe_cache, 1, &graphics_pipeline_create_info, Device::alloc_callbacks(), &v.pipeline));
    if (extension_plan.uses_depth_clamp_zero_one) {
        // VK_KHR_depth_clamp_zero_one is device-wide: every raster pipeline
        // created on a device that enabled its feature clamps the fragment depth
        // to [0, 1], so this draw really renders through it instead of the
        // viewport-range-only fallback.
        device()->mark_optional_extension_active(
            detail::OptionalExtension::DEPTH_CLAMP_ZERO_ONE);
    }
    return v;
}
VkRenderPass RasterShader::create_render_pass(
    Device *device,
    RasterState const &state,
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_textures) {
    // The body lives in raster_pipeline.cpp so that the native raster shader
    // gets the same render pass for the same attachment configuration. This
    // static member is kept because it is part of the class' public surface.
    return create_raster_render_pass(device, state, rtv_textures, dsv_textures);
}
}// namespace lc::vk
