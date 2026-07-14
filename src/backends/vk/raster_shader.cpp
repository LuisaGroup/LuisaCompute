#include "raster_shader.h"
#include "device.h"
#include "log.h"
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
    uint validation_count)
    : Shader(device, ShaderTag::kRasterShader, std::move(captured), std::move(saved_arguments), binds, use_tex2d_bindless, use_tex3d_bindless, use_buffer_bindless, {}, {}, validation_count), _vertex_spv_code(std::move(vertex_spv_code)), _pixel_spv_code(std::move(pixel_spv_code)) {
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
    VkPipelineVertexInputStateCreateInfo &vertex_input_create_info) -> BinaryBlob {
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
    }
    if (key_size == 0) [[unlikely]]
        return {};
    BinaryBlob result(key_size);// make memory clear

    // push data
    {
        std::byte *ptr = result.data();
        luisa::fixed_vector<uint32_t, 16> offsets;
        offsets.reserve(mesh_format.vertex_stream_count());
        vertex_input_create_info.pVertexAttributeDescriptions = reinterpret_cast<const VkVertexInputAttributeDescription *>(ptr);
        uint32_t location = 0;
        for (auto stream_idx : vstd::range((int64_t)mesh_format.vertex_stream_count())) {
            uint32_t offset = 0;
            for (auto &&attrs : mesh_format.attributes(stream_idx)) {
                VkVertexInputAttributeDescription desc{
                    .location = location,
                    .binding = (uint32_t)stream_idx,
                    .format = Texture::to_vk_format(attrs.format),
                    .offset = offset};
                ++location;
                offset += pixel_format_size(attrs.format, uint3(1));
                std::memcpy(ptr, &desc, sizeof(desc));
                ptr += sizeof(desc);
            }
            offsets.emplace_back(offset);
        }
        vertex_input_create_info.pVertexBindingDescriptions = reinterpret_cast<const VkVertexInputBindingDescription *>(ptr);
        for (auto stream_idx : vstd::range((int64_t)mesh_format.vertex_stream_count())) {
            VkVertexInputBindingDescription desc{
                .binding = (uint32_t)stream_idx,
                .stride = offsets[stream_idx],
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
            std::memcpy(ptr, &desc, sizeof(desc));
            ptr += sizeof(desc);
        }
        std::memcpy(ptr, &state, sizeof(state));
        ptr += sizeof(RasterState);
        LUISA_ASSERT(ptr == result.data() + result.size());
    }
    return result;
}
auto RasterShader::create_pipeline(
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_textures,
    luisa::compute::MeshFormat const &mesh_format,
    RasterState const &state) -> Pipeline {
    VkPrimitiveTopology topo;
    VkCullModeFlags cull_mode;
    VkPipelineVertexInputStateCreateInfo vertex_input;
    auto binary_blob = _make_pipeline_key(
        mesh_format,
        state,
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

    VkPipelineRasterizationStateCreateInfo raster_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = state.depth_clip,
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

    VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = state.blend_state.enable_blend,
        .srcColorBlendFactor = get_blend_factor(state.blend_state.prim_op),
        .dstColorBlendFactor = get_blend_factor(state.blend_state.img_op),
        .colorBlendOp = get_blend_op(state.blend_state.op),
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo blend_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_CLEAR,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
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
        .depthBoundsTestEnable = VK_TRUE,
        .stencilTestEnable = state.stencil_state.enable_stencil,
        .front = VkStencilOpState{
            .failOp = get_stencil_state(state.stencil_state.front_face_op.stencil_fail_op),
            .passOp = get_stencil_state(state.stencil_state.front_face_op.pass_op),
            .depthFailOp = get_stencil_state(state.stencil_state.front_face_op.depth_fail_op),
            .compareOp = get_compare_op(state.stencil_state.front_face_op.comparison),
            .compareMask = ~0u,
            .writeMask = ~0u,
            .reference = ~0u,
        },
        .back = VkStencilOpState{
            .failOp = get_stencil_state(state.stencil_state.back_face_op.stencil_fail_op),
            .passOp = get_stencil_state(state.stencil_state.back_face_op.pass_op),
            .depthFailOp = get_stencil_state(state.stencil_state.back_face_op.depth_fail_op),
            .compareOp = get_compare_op(state.stencil_state.back_face_op.comparison),
            .compareMask = ~0u,
            .writeMask = ~0u,
            .reference = ~0u,
        },
        .minDepthBounds = 0.f,
        .maxDepthBounds = 1.f,
    };
    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = vstd::array_count(dynamic_states),
        .pDynamicStates = dynamic_states};
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
    return v;
}
VkRenderPass RasterShader::create_render_pass(
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
}// namespace lc::vk
