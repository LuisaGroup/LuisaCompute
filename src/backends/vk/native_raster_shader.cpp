#include "native_raster_shader.h"

#include "device.h"
#include "log.h"
#include "stream.h"
#include "buffer.h"
#include "raster_pipeline.h"

#include <limits>

#include <luisa/core/logging.h>

namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;

// Every graphics-pipeline-state choice below mirrors the DSL raster path
// (`RasterShader::create_pipeline` in raster_shader.cpp); the render pass and
// the pipeline cache key come from the very same helpers
// (`raster_pipeline.h`). The only things a native raster shader brings itself
// are the two SPIR-V modules, the pipeline layout built from their merged
// reflection, and the entry-point names.
namespace native_raster_shader_detail {

// Both stages of a native raster shader share one pipeline layout, so every
// descriptor set binding and the push-constant range are visible to both.
constexpr auto both_graphics_stages =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

[[nodiscard]] VkDescriptorType descriptor_type_of(
    NativeShaderResourceKind kind) noexcept {
    switch (kind) {
        case NativeShaderResourceKind::ConstantBuffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
}

[[nodiscard]] VkCullModeFlags cull_mode_of(CullMode mode) noexcept {
    switch (mode) {
        case CullMode::None: return VK_CULL_MODE_NONE;
        case CullMode::Back: return VK_CULL_MODE_BACK_BIT;
        case CullMode::Front: return VK_CULL_MODE_FRONT_BIT;
    }
    return VK_CULL_MODE_NONE;
}

[[nodiscard]] VkPrimitiveTopology topology_of(TopologyType topology) noexcept {
    switch (topology) {
        case TopologyType::Point: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case TopologyType::Line: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case TopologyType::Triangle: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

[[nodiscard]] VkBlendOp blend_op_of(BlendOp op) noexcept {
    switch (op) {
        case BlendOp::Add: return VK_BLEND_OP_ADD;
        case BlendOp::Max: return VK_BLEND_OP_MAX;
        case BlendOp::Min: return VK_BLEND_OP_MIN;
        default: return VK_BLEND_OP_SUBTRACT;
    }
}

[[nodiscard]] VkBlendFactor blend_factor_of(BlendWeight weight) noexcept {
    switch (weight) {
        case BlendWeight::Zero: return VK_BLEND_FACTOR_ZERO;
        case BlendWeight::One: return VK_BLEND_FACTOR_ONE;
        case BlendWeight::PrimColor: return VK_BLEND_FACTOR_SRC_COLOR;
        case BlendWeight::ImgColor: return VK_BLEND_FACTOR_DST_COLOR;
        case BlendWeight::PrimAlpha: return VK_BLEND_FACTOR_SRC_ALPHA;
        case BlendWeight::ImgAlpha: return VK_BLEND_FACTOR_DST_ALPHA;
        case BlendWeight::OneMinusPrimColor: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BlendWeight::OneMinusImgColor: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BlendWeight::OneMinusPrimAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        default: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    }
}

// Vulkan does not allow the *_COLOR blend factors on the alpha channel; map
// them to their alpha counterparts like D3D12 does implicitly.
[[nodiscard]] VkBlendFactor alpha_blend_factor_of(BlendWeight weight) noexcept {
    switch (weight) {
        case BlendWeight::PrimColor: return VK_BLEND_FACTOR_SRC_ALPHA;
        case BlendWeight::ImgColor: return VK_BLEND_FACTOR_DST_ALPHA;
        case BlendWeight::OneMinusPrimColor: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BlendWeight::OneMinusImgColor: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        default: return blend_factor_of(weight);
    }
}

[[nodiscard]] VkCompareOp compare_op_of(Comparison const &comp) noexcept {
    switch (comp) {
        case Comparison::Never: return VK_COMPARE_OP_NEVER;
        case Comparison::Less: return VK_COMPARE_OP_LESS;
        case Comparison::Equal: return VK_COMPARE_OP_EQUAL;
        case Comparison::LessEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case Comparison::Greater: return VK_COMPARE_OP_GREATER;
        case Comparison::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
        case Comparison::GreaterEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        default: return VK_COMPARE_OP_ALWAYS;
    }
}

[[nodiscard]] VkStencilOp stencil_op_of(StencilOp op) noexcept {
    switch (op) {
        case StencilOp::Keep: return VK_STENCIL_OP_KEEP;
        case StencilOp::Replace: return VK_STENCIL_OP_REPLACE;
        default: return VK_STENCIL_OP_ZERO;
    }
}

[[nodiscard]] VkStencilOpState stencil_state_of(
    StencilFaceOp const &op, StencilState const &state) noexcept {
    return VkStencilOpState{
        .failOp = stencil_op_of(op.stencil_fail_op),
        .passOp = stencil_op_of(op.pass_op),
        .depthFailOp = stencil_op_of(op.depth_fail_op),
        .compareOp = compare_op_of(op.comparison),
        .compareMask = state.read_mask,
        .writeMask = state.write_mask,
        .reference = state.reference,
    };
}

}// namespace native_raster_shader_detail

NativeRasterShader::NativeRasterShader(
    Device *device,
    luisa::span<const NativeShaderResourceBinding> bindings,
    luisa::span<const std::byte> vertex_spirv,
    luisa::string_view vertex_entry_point,
    luisa::span<const std::byte> pixel_spirv,
    luisa::string_view pixel_entry_point,
    uint32_t push_constant_size,
    uint32_t object_id_bytes) noexcept
    : Resource{device},
      _push_constant_size{push_constant_size},
      _object_id_bytes{object_id_bytes},
      _vertex_entry_point{vertex_entry_point},
      _pixel_entry_point{pixel_entry_point},
      _bindings{bindings.begin(), bindings.end()} {
    // The object-id slot is a single 32-bit word (or absent); anything else
    // would silently misalign the application's uniform block.
    LUISA_ASSERT(object_id_bytes == 0u || object_id_bytes == 4u,
                 "A native Vulkan raster shader reserves 0 or 4 push-constant "
                 "bytes for the mesh object id (got {}).",
                 object_id_bytes);
    // Note: unlike the compute route there is no "must declare at least one
    // binding or a push-constant block" requirement - a constant vertex/pixel
    // pair with neither is a legal pipeline.
    auto logic_device = device->logic_device();
    auto callbacks = Device::alloc_callbacks();

    // ---- descriptor set layouts ------------------------------------------
    // One set layout per descriptor set index the merged table uses, so the
    // module's `set` numbers map onto pipeline-layout indices. Both stages see
    // every binding, because they share one pipeline layout.
    for (auto i = 0u; i < _bindings.size(); i++) {
        auto &&binding = _bindings[i];
        LUISA_ASSERT(binding.array_size == 1u,
                     "Native Vulkan raster shader binding (set {}, binding {}) "
                     "is an array of {} descriptors, which is not supported yet.",
                     binding.space_index, binding.register_index, binding.array_size);
        if (_set_layouts.size() <= binding.space_index) {
            _set_layouts.resize(binding.space_index + 1u, VK_NULL_HANDLE);
        }
        NativeShader::LayoutBinding layout_binding;
        layout_binding.binding_index = i;
        layout_binding.set = binding.space_index;
        layout_binding.binding = binding.register_index;
        layout_binding.descriptor_count = binding.array_size;
        layout_binding.descriptor_type =
            native_raster_shader_detail::descriptor_type_of(binding.kind);
        _layout_bindings.emplace_back(layout_binding);
    }
    _descriptor_set_count = static_cast<uint32_t>(_set_layouts.size());
    for (auto set = 0u; set < _set_layouts.size(); set++) {
        vstd::vector<VkDescriptorSetLayoutBinding> set_bindings;
        for (auto &&binding : _layout_bindings) {
            if (binding.set != set) { continue; }
            VkDescriptorSetLayoutBinding layout_binding{
                .binding = binding.binding,
                .descriptorType = binding.descriptor_type,
                .descriptorCount = binding.descriptor_count,
                .stageFlags = native_raster_shader_detail::both_graphics_stages,
                .pImmutableSamplers = nullptr};
            set_bindings.emplace_back(layout_binding);
        }
        VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(set_bindings.size()),
            .pBindings = set_bindings.data()};
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
            logic_device, &layout_info, callbacks, &_set_layouts[set]));
    }
    // Anything created below is released here when the constructor fails
    // half-way, so the object stays destructible (mirrors NativeShader).
    auto ok = false;
    auto dispose_partial = vstd::scope_exit([&] {
        if (ok) { return; }
        for (auto layout : _set_layouts) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(logic_device, layout, callbacks);
            }
        }
        _set_layouts.clear();
        if (_descriptor_pool != nullptr) {
            vkDestroyDescriptorPool(logic_device, _descriptor_pool->pool, callbacks);
            _descriptor_pool->pool = VK_NULL_HANDLE;
            _descriptor_pool = nullptr;
        }
        if (_pipeline_cache != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(logic_device, _pipeline_cache, callbacks);
            _pipeline_cache = VK_NULL_HANDLE;
        }
        if (_vertex_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(logic_device, _vertex_module, callbacks);
            _vertex_module = VK_NULL_HANDLE;
        }
        if (_pixel_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(logic_device, _pixel_module, callbacks);
            _pixel_module = VK_NULL_HANDLE;
        }
        if (_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(logic_device, _pipeline_layout, callbacks);
            _pipeline_layout = VK_NULL_HANDLE;
        }
    });

    // ---- pipeline layout -------------------------------------------------
    // The push-constant range covers the reserved object-id word *and* the
    // application's uniform bytes; the launcher pushes its blob at
    // `object_id_bytes`, the backend pushes the mesh object id at 0.
    auto push_constant_range_size = _object_id_bytes + _push_constant_size;
    // Report an over-large block with an actionable message instead of letting
    // vkCreatePipelineLayout fail opaquely.
    if (push_constant_range_size > device->properties().limits.maxPushConstantsSize) {
        LUISA_ERROR_WITH_LOCATION(
            "A native raster shader needs {} push-constant bytes ({} reserved for "
            "the mesh object id plus {} declared by the application), exceeding "
            "the device's maxPushConstantsSize ({}).",
            push_constant_range_size, _object_id_bytes, _push_constant_size,
            device->properties().limits.maxPushConstantsSize);
    }
    VkPushConstantRange push_constant_range{
        .stageFlags = native_raster_shader_detail::both_graphics_stages,
        .offset = 0u,
        .size = push_constant_range_size};
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(_set_layouts.size()),
        .pSetLayouts = _set_layouts.data(),
        .pushConstantRangeCount = push_constant_range_size > 0u ? 1u : 0u,
        .pPushConstantRanges = push_constant_range_size > 0u ? &push_constant_range : nullptr};
    VK_CHECK_RESULT(vkCreatePipelineLayout(
        logic_device, &pipeline_layout_info, callbacks, &_pipeline_layout));

    // ---- shader modules --------------------------------------------------
    // Created once here: the pipeline cache creates one pipeline per
    // (vertex input, raster state, attachment format) key, and all of them use
    // these two modules.
    VkShaderModuleCreateInfo module_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vertex_spirv.size_bytes(),
        .pCode = reinterpret_cast<const uint32_t *>(vertex_spirv.data())};
    VK_CHECK_RESULT(vkCreateShaderModule(
        logic_device, &module_info, callbacks, &_vertex_module));
    module_info.codeSize = pixel_spirv.size_bytes();
    module_info.pCode = reinterpret_cast<const uint32_t *>(pixel_spirv.data());
    VK_CHECK_RESULT(vkCreateShaderModule(
        logic_device, &module_info, callbacks, &_pixel_module));

    // ---- pipeline cache --------------------------------------------------
    VkPipelineCacheCreateInfo cache_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VK_CHECK_RESULT(vkCreatePipelineCache(
        logic_device, &cache_info, callbacks, &_pipeline_cache));

    // ---- descriptor pool -------------------------------------------------
    // Same shape as NativeShader's: per-set descriptor counts times the number
    // of descriptor sets a frame may hold, refilled as frames retire.
    vstd::vector<VkDescriptorPoolSize> pool_sizes;
    auto count_of = [&](VkDescriptorType type) noexcept {
        auto count = 0u;
        for (auto &&binding : _layout_bindings) {
            if (binding.descriptor_type == type) {
                count += binding.descriptor_count;
            }
        }
        return count;
    };
    for (auto type : {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}) {
        auto per_set = count_of(type);
        if (per_set > 0u) {
            pool_sizes.emplace_back(
                VkDescriptorPoolSize{type, per_set * max_in_flight_descriptor_sets});
        }
    }
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = max_in_flight_descriptor_sets,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data()};
    VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
    VK_CHECK_RESULT(vkCreateDescriptorPool(
        logic_device, &pool_info, callbacks, &descriptor_pool));
    _descriptor_pool = luisa::make_shared<NativeShaderDescriptorPool>(
        logic_device, descriptor_pool);
    ok = true;
}

NativeRasterShader::~NativeRasterShader() noexcept {
    auto logic_device = device()->logic_device();
    auto callbacks = Device::alloc_callbacks();
    // Pipelines and their render passes are owned by this shader and must go
    // first: every pipeline references the render pass it was built for and the
    // pipeline layout destroyed below.
    for (auto &&i : _pipelines) {
        if (i.second.pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(logic_device, i.second.pipeline, callbacks);
        }
        if (i.second.render_pass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(logic_device, i.second.render_pass, callbacks);
        }
    }
    // The descriptor pool is released by the shared holder, once no recorded
    // draw still references its sets (see NativeShaderDescriptorPool).
    _descriptor_pool = nullptr;
    if (_pipeline_cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(logic_device, _pipeline_cache, callbacks);
    }
    if (_pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(logic_device, _pipeline_layout, callbacks);
    }
    for (auto layout : _set_layouts) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(logic_device, layout, callbacks);
        }
    }
    if (_vertex_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(logic_device, _vertex_module, callbacks);
    }
    if (_pixel_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(logic_device, _pixel_module, callbacks);
    }
}

RasterPipelineHandle NativeRasterShader::create_pipeline(
    luisa::span<Argument::Texture const> rtv_textures,
    Argument::Texture dsv_texture,
    MeshFormat const &mesh_format,
    RasterState const &state,
    luisa::span<uint const> strides) noexcept {
    using namespace native_raster_shader_detail;
    auto logic_device = device()->logic_device();
    // The key blob owns the vertex-input descriptions the create-info points at,
    // so it must outlive pipeline creation; moving it into the cache steals its
    // buffer (luisa::vector move), which keeps those pointers valid.
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    auto key = build_raster_pipeline_key(
        mesh_format, state, rtv_textures, dsv_texture, strides, vertex_input);
    auto iter = _pipelines.try_emplace(std::move(key));
    auto &pipe = iter.first.value();
    if (!iter.second) { return pipe; }// cache hit
    VkPipelineInputAssemblyStateCreateInfo input_asm_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topology_of(state.topology),
        .primitiveRestartEnable = VK_FALSE};
    // The same planner the DSL raster path uses, so depth clamping
    // (VK_KHR_depth_clamp_zero_one) and the dynamic-state list can never drift
    // between the two pipeline construction sites.
    auto extension_plan = plan_raster_pipeline_extensions(device(), state);
    VkPipelineRasterizationStateCreateInfo raster_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        // RasterState::depth_clip follows the D3D12 DepthClipEnable semantics:
        // enabled clips primitives outside [0, 1], which is the Vulkan default
        // (depth clamp off); disabled requests clamping, which is only
        // available when the depthClamp feature is enabled. With
        // VK_KHR_depth_clamp_zero_one enabled the device clamps the fragment
        // depth itself to [0, 1] on every draw.
        .depthClampEnable = extension_plan.depth_clamp_enable,
        .rasterizerDiscardEnable = false,
        .polygonMode = state.fill_mode == FillMode::Solid ?
                           VK_POLYGON_MODE_FILL :
                           VK_POLYGON_MODE_LINE,
        .cullMode = cull_mode_of(state.cull_mode),
        .frontFace = state.front_counter_clockwise ?
                         VK_FRONT_FACE_COUNTER_CLOCKWISE :
                         VK_FRONT_FACE_CLOCKWISE,
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
    VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = state.blend_state.enable_blend,
        .srcColorBlendFactor = blend_factor_of(state.blend_state.prim_op),
        .dstColorBlendFactor = blend_factor_of(state.blend_state.img_op),
        .colorBlendOp = blend_op_of(state.blend_state.op),
        .srcAlphaBlendFactor = alpha_blend_factor_of(state.blend_state.prim_op),
        .dstAlphaBlendFactor = alpha_blend_factor_of(state.blend_state.img_op),
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    luisa::vector<VkPipelineColorBlendAttachmentState> blend_attachments(
        rtv_textures.size(), blend_attachment);
    VkPipelineColorBlendStateCreateInfo blend_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_CLEAR,
        .attachmentCount = static_cast<uint32_t>(blend_attachments.size()),
        .pAttachments = blend_attachments.data(),
        .blendConstants = {1.f, 1.f, 1.f, 1.f}};
    // Viewport and scissor are dynamic state, so the pipeline only carries the
    // counts; the values are recorded per draw.
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
    VkPipelineDepthStencilStateCreateInfo depth_stencil_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = state.depth_state.enable_depth,
        .depthWriteEnable = state.depth_state.write,
        .depthCompareOp = compare_op_of(state.depth_state.comparison),
        // No depth-bounds test is exposed through RasterState, and the
        // depthBounds feature is not enabled on the logical device.
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = state.stencil_state.enable_stencil,
        .front = stencil_state_of(state.stencil_state.front_face_op,
                                  state.stencil_state),
        .back = stencil_state_of(state.stencil_state.back_face_op,
                                 state.stencil_state),
        .minDepthBounds = 0.f,
        .maxDepthBounds = 1.f,
    };
    VkPipelineDynamicStateCreateInfo dynamic_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(
            extension_plan.dynamic_states.size()),
        .pDynamicStates = extension_plan.dynamic_states.data()};
    VkPipelineShaderStageCreateInfo shader_stages[] = {
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = _vertex_module,
            .pName = _vertex_entry_point.c_str()},
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = _pixel_module,
            .pName = _pixel_entry_point.c_str()}};
    auto render_pass = create_raster_render_pass(
        device(), state, rtv_textures, dsv_texture);
    pipe.render_pass = render_pass;
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
        .layout = _pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = ~0};
    auto create_result = vkCreateGraphicsPipelines(
        logic_device, _pipeline_cache, 1u, &graphics_pipeline_create_info,
        Device::alloc_callbacks(), &pipe.pipeline);
    if (create_result != VK_SUCCESS) {
        // Fail closed: drop the half-built render pass and the cache entry, so a
        // later draw retries instead of receiving a null pipeline from the
        // cache. Removing the entry invalidates `pipe`, hence the local
        // `render_pass`.
        _pipelines.remove(iter.first);
        vkDestroyRenderPass(logic_device, render_pass, Device::alloc_callbacks());
        LUISA_ERROR_WITH_LOCATION(
            "Native raster shader could not create the graphics pipeline "
            "(VkResult {}). The two modules must declare matching vertex "
            "outputs/pixel inputs and pipeline-compatible interfaces.",
            luisa::to_underlying(create_result));
        return {};
    }
    if (extension_plan.uses_depth_clamp_zero_one) {
        // VK_KHR_depth_clamp_zero_one is device-wide: every raster pipeline
        // created on a device that enabled its feature clamps the fragment depth
        // to [0, 1], so this pipeline really renders through the extension
        // instead of the viewport-range-only fallback.
        device()->mark_optional_extension_active(
            detail::OptionalExtension::DEPTH_CLAMP_ZERO_ONE);
    }
    return RasterPipelineHandle{.pipeline = pipe.pipeline,
                                .render_pass = render_pass};
}

void NativeRasterShader::push_object_id(
    VkCommandBuffer cmdbuffer, uint32_t object_id) noexcept {
    LUISA_ASSERT(_object_id_bytes == 4u,
                 "A native raster shader only pushes a mesh object id when it "
                 "reserved the 4-byte slot (object_id_bytes = {}).",
                 _object_id_bytes);
    vkCmdPushConstants(
        cmdbuffer, _pipeline_layout,
        native_raster_shader_detail::both_graphics_stages,
        0u, sizeof(uint32_t), &object_id);
}

VkRenderPass NativeRasterShader::encode_draw_state(
    CommandBufferState *state, VkCommandBuffer cmdbuffer,
    DrawRasterSceneCommand const *cmd,
    luisa::span<uint const> strides) noexcept {
    using namespace native_raster_shader_detail;
    // ---- pipeline --------------------------------------------------------
    auto pipe = create_pipeline(
        cmd->rtv_texs(), cmd->dsv_tex(), cmd->mesh_format(),
        cmd->raster_state(), strides);
    if (pipe.pipeline == VK_NULL_HANDLE) { return VK_NULL_HANDLE; }// already reported
    auto logic_device = device()->logic_device();
    auto arguments = cmd->arguments();
    // ---- descriptor sets -------------------------------------------------
    // One set per descriptor set index; the recording view is kept in `sets`
    // while a copy is handed to the command-buffer state, which releases them
    // once the frame has been retired.
    vstd::vector<VkDescriptorSet> sets(_descriptor_set_count, VK_NULL_HANDLE);
    if (_descriptor_set_count > 0u) {
        LUISA_ASSERT(_descriptor_pool != nullptr,
                     "Native raster shader has no descriptor pool.");
        VkDescriptorSetAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _descriptor_pool->pool,
            .descriptorSetCount = static_cast<uint32_t>(_set_layouts.size()),
            .pSetLayouts = _set_layouts.data()};
        auto allocate_result = vkAllocateDescriptorSets(
            logic_device, &allocate_info, sets.data());
        if (allocate_result != VK_SUCCESS) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader draw could not allocate {} descriptor sets "
                "from the shader's pool (VkResult {}); at most {} draws of one "
                "native raster shader may be recorded per command buffer.",
                sets.size(), luisa::to_underlying(allocate_result),
                max_in_flight_descriptor_sets);
            return VK_NULL_HANDLE;
        }
        state->dispose_after_flush(NativeShaderDescriptorDeferredRelease{
            _descriptor_pool, vstd::vector<VkDescriptorSet>{sets}});
    }
    // ---- descriptor writes -----------------------------------------------
    // `cmd->arguments()` is in canonical (set, binding) order, the same order as
    // the merged reflection table, so argument i (uniforms excluded) is binding
    // i.
    vstd::vector<VkDescriptorBufferInfo> buffer_infos;
    vstd::vector<VkWriteDescriptorSet> writes;
    buffer_infos.reserve(arguments.size());
    writes.reserve(arguments.size());
    for (auto i = 0u; i < arguments.size(); i++) {
        auto &&argument = arguments[i];
        if (argument.tag == Argument::Tag::UNIFORM) { continue; }
        if (i >= _bindings.size()) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader draw supplied {} resource arguments for {} "
                "reflected bindings.",
                arguments.size(), _bindings.size());
            return VK_NULL_HANDLE;
        }
        auto &&binding = _bindings[i];
        if (argument.tag != Argument::Tag::BUFFER) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader draw argument {} is not a buffer; textures, "
                "bindless arrays and acceleration structures are rejected at "
                "load_raster().",
                i);
            return VK_NULL_HANDLE;
        }
        auto buffer = reinterpret_cast<Buffer const *>(argument.buffer.handle);
        if (buffer == nullptr) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader draw argument {} has a null buffer.", i);
            return VK_NULL_HANDLE;
        }
        auto offset = static_cast<VkDeviceSize>(argument.buffer.offset);
        auto addressable = static_cast<VkDeviceSize>(buffer->addressable_byte_size());
        if (offset >= addressable) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader draw argument {} has offset {} beyond the "
                "buffer's addressable size {}.",
                i, offset, addressable);
            return VK_NULL_HANDLE;
        }
        auto requested = static_cast<VkDeviceSize>(argument.buffer.size);
        auto range = requested == 0u ? addressable - offset :
                                       std::min(requested, addressable - offset);
        auto descriptor_type = descriptor_type_of(binding.kind);
        if (descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
            range > device()->properties().limits.maxUniformBufferRange) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader constant buffer at set {} binding {} needs "
                "{} bytes, exceeding maxUniformBufferRange ({}).",
                binding.space_index, binding.register_index, range,
                device()->properties().limits.maxUniformBufferRange);
            return VK_NULL_HANDLE;
        }
        buffer_infos.emplace_back(
            VkDescriptorBufferInfo{buffer->vk_buffer(), offset, range});
        writes.emplace_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = sets[binding.space_index],
            .dstBinding = binding.register_index,
            .dstArrayElement = 0u,
            .descriptorCount = 1u,
            .descriptorType = descriptor_type,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_infos.back(),
            .pTexelBufferView = nullptr});
    }
    // A binding with no argument leaves its descriptor unwritten, i.e. the
    // shader would read undefined contents, so an incomplete set is an error in
    // both directions (the per-argument loop above already rejects extras).
    if (writes.size() != _bindings.size()) {
        LUISA_ERROR_WITH_LOCATION(
            "Native raster draw supplied {} resource arguments for {} "
            "reflected bindings; every binding must be bound exactly once.",
            writes.size(), _bindings.size());
        return VK_NULL_HANDLE;
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(logic_device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0u, nullptr);
    }
    vkCmdBindPipeline(cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe.pipeline);
    if (!sets.empty()) {
        vkCmdBindDescriptorSets(
            cmdbuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipeline_layout,
            0u, static_cast<uint32_t>(sets.size()), sets.data(), 0u, nullptr);
    }
    // ---- push constants (the launcher's uniform block) -------------------
    // The application's bytes live at `object_id_bytes`; the reserved slot in
    // front of them is written per mesh by `push_object_id`. The range check is
    // unconditional (unlike the compute route, which skips an empty block) so
    // that a payload that does not fit is reported instead of being dropped.
    auto begin = std::numeric_limits<uint64_t>::max();
    auto end = uint64_t{0u};
    for (auto &&argument : arguments) {
        if (argument.tag == Argument::Tag::UNIFORM) {
            begin = std::min<uint64_t>(begin, argument.uniform.offset);
            end = std::max<uint64_t>(end, argument.uniform.offset +
                                              argument.uniform.size);
        }
    }
    if (end > begin) {
        auto bytes = end - begin;
        if (bytes % sizeof(uint32_t) != 0u || bytes > _push_constant_size) {
            LUISA_ERROR_WITH_LOCATION(
                "Native raster shader uniform payload ({} bytes) does not fit the "
                "declared push-constant block ({} bytes).",
                bytes, _push_constant_size);
            return VK_NULL_HANDLE;
        }
        auto payload = cmd->uniform(Argument::Uniform{
            static_cast<size_t>(begin), static_cast<size_t>(bytes), 4u});
        vkCmdPushConstants(
            cmdbuffer, _pipeline_layout, both_graphics_stages,
            _object_id_bytes, static_cast<uint32_t>(bytes), payload.data());
    }
    return pipe.render_pass;
}

}// namespace lc::vk
