#include "shader.h"
#include "log.h"
#include "device.h"
#include "buffer.h"
#include "upload_buffer.h"
#include "shader_binary_contract.h"
#include "descriptor_interface_plan.h"
#include "shader_interface_plan.h"
#include <limits>
namespace lc::vk {

SavedArgument::SavedArgument(luisa::compute::Type const *type) {
    auto checked_u32_size = [](size_t size, luisa::string_view description) {
        LUISA_ASSERT(
            size <= std::numeric_limits<uint>::max(),
            "Vulkan saved-argument size for '{}' exceeds the persisted "
            "32-bit ABI: {} bytes.",
            description, size);
        return static_cast<uint>(size);
    };
    tag = type->tag();
    if (!type->is_resource() && !type->is_custom() &&
        !type->is_cooperative_vector_ref() &&
        !type->is_cooperative_matrix_ref()) {
        struct_size = checked_u32_size(type->size(), type->description());
    } else if (type->is_buffer() && type->element() != nullptr) {
        // Reused by the XIR/SPIR-V runtime to choose a descriptor base whose
        // relative bias remains an exact element multiple.
        struct_size = checked_u32_size(
            type->element()->size(), type->element()->description());
    }
}
Shader::Shader(
    Device *device,
    ShaderTag tag,
    vstd::vector<Argument> &&captured,
    vstd::vector<SavedArgument> &&saved_arguments,
    vstd::span<hlsl::Property const> binds,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless,
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers,
    luisa::span<const std::byte> constant_ubo_data,
    uint validation_count,
    uint32_t push_constant_size,
    detail::ShaderCodegenDialect codegen_dialect)
    : Resource{device}, _captured{std::move(captured)}, _saved_arguments(std::move(saved_arguments)),
      _shader_tag(tag), _use_tex2d_bindless(use_tex2d_bindless), _use_tex3d_bindless(use_tex3d_bindless), _use_buffer_bindless(use_buffer_bindless), _printers(std::move(printers)), _validation_count(validation_count), _push_constant_size(push_constant_size), _codegen_dialect(codegen_dialect) {
    auto argument_contract = plan_saved_argument_contract(
        _saved_arguments, validation_count);
    LUISA_ASSERT(
        argument_contract,
        "Vulkan shader argument-block ABI is invalid: {}.",
        saved_argument_contract_status_name(argument_contract.status));
    LUISA_ASSERT(
        _captured.size() <= _saved_arguments.size(),
        "Vulkan shader has {} captured arguments but only {} saved ABI entries.",
        _captured.size(), _saved_arguments.size());
    for (size_t i = 0u; i < _captured.size(); ++i) {
        auto runtime_tag = _captured[i].tag;
        auto saved_tag = _saved_arguments[i].tag;
        auto matches = [&]() noexcept {
            switch (runtime_tag) {
                case Argument::Tag::BUFFER:
                    return saved_tag == Type::Tag::BUFFER ||
                           saved_tag == Type::Tag::CUSTOM;
                case Argument::Tag::TEXTURE:
                    return saved_tag == Type::Tag::TEXTURE;
                case Argument::Tag::BINDLESS_ARRAY:
                    return saved_tag == Type::Tag::BINDLESS_ARRAY;
                case Argument::Tag::ACCEL:
                    return saved_tag == Type::Tag::ACCEL;
                case Argument::Tag::UNIFORM: return false;
            }
            return false;
        }();
        LUISA_ASSERT(
            matches,
            "Vulkan captured argument {} with runtime tag {} does not match "
            "saved ABI tag {}.",
            i, luisa::to_underlying(runtime_tag),
            luisa::to_underlying(saved_tag));
    }
    LUISA_ASSERT(
        detail::valid_shader_constant_payload_size(
            constant_ubo_data.size_bytes(),
            device->properties().limits.maxUniformBufferRange),
        "Vulkan shader constant UBO payload ({} bytes) exceeds the device "
        "maxUniformBufferRange ({} bytes) or the serialized payload cap "
        "({} bytes).",
        constant_ubo_data.size_bytes(),
        device->properties().limits.maxUniformBufferRange,
        detail::max_shader_constant_payload_size);
    if ((!device->enable_bindless()) && (use_tex2d_bindless || use_tex3d_bindless || use_buffer_bindless)) [[unlikely]] {
        LUISA_ERROR("Bindless not enabled, shader can not be load.");
    }
    auto interface_stage_mask = [&] {
        switch (tag) {
            case ShaderTag::kComputeShader:
                return detail::DescriptorInterfaceStageMask::COMPUTE;
            case ShaderTag::kRasterShader:
                return detail::DescriptorInterfaceStageMask::RASTER;
            case ShaderTag::kRayTracingShader:
                return detail::DescriptorInterfaceStageMask::RAY_TRACING;
        }
        return static_cast<detail::DescriptorInterfaceStageMask>(0u);
    }();
    auto runtime_interface_plan = detail::plan_shader_interface(
        {.properties = binds,
         .arguments = _saved_arguments,
         .stage_mask = interface_stage_mask,
         .dialect = codegen_dialect,
         .printer_count = static_cast<uint32_t>(_printers.size()),
         .validation_count = validation_count,
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .has_constant_ubo_payload = !constant_ubo_data.empty()});
    LUISA_ASSERT(
        runtime_interface_plan,
        "Vulkan shader runtime descriptor interface is invalid: {}.",
        detail::shader_interface_error_name(
            runtime_interface_plan.error));
    auto interface_plan = detail::plan_descriptor_interface(
        {.properties = binds,
         .stage_mask = interface_stage_mask,
         .bindless_heap_capacity = device->bindless_heap_capacity(),
         .use_buffer_bindless = use_buffer_bindless,
         .use_tex2d_bindless = use_tex2d_bindless,
         .use_tex3d_bindless = use_tex3d_bindless,
         .has_constant_ubo_payload = !constant_ubo_data.empty(),
         .acceleration_structure_available = device->enable_raytracing(),
         .sampled_image_update_after_bind_enabled = device->enable_bindless(),
         .storage_buffer_update_after_bind_enabled = device->enable_bindless()},
        detail::descriptor_interface_limits_from(
            device->properties().limits,
            device->descriptor_indexing_properties(),
            device->acceleration_structure_properties()));
    LUISA_ASSERT(
        interface_plan,
        "Vulkan shader descriptor interface is invalid: {}.",
        detail::descriptor_interface_error_name(interface_plan.error));
    LUISA_ASSERT(
        interface_plan.local_binding_count ==
            runtime_interface_plan.local_binding_count,
        "Vulkan runtime interface planned {} local bindings, but the device "
        "descriptor plan produced {}.",
        runtime_interface_plan.local_binding_count,
        interface_plan.local_binding_count);
    _local_descriptor_binding_count =
        runtime_interface_plan.local_binding_count;
    _resource_argument_binding_offset =
        runtime_interface_plan.argument_buffer_binding_count +
        runtime_interface_plan.constant_ubo_binding_count;
    _uses_indirect_dispatch =
        runtime_interface_plan.indirect_binding_count != 0u;
    LUISA_ASSERT(
        push_constant_size != 0u &&
            push_constant_size % sizeof(uint32_t) == 0u &&
            push_constant_size <=
                device->properties().limits.maxPushConstantsSize,
        "Vulkan shader push-constant size {} is invalid or exceeds the device limit {}.",
        push_constant_size,
        device->properties().limits.maxPushConstantsSize);
    VkShaderStageFlagBits stage_bits = [&]() -> VkShaderStageFlagBits {
        switch (tag) {
            case ShaderTag::kComputeShader:
                return VK_SHADER_STAGE_COMPUTE_BIT;
            case ShaderTag::kRasterShader:
                return static_cast<VkShaderStageFlagBits>(
                    VK_SHADER_STAGE_VERTEX_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT);
            case ShaderTag::kRayTracingShader:
                return static_cast<VkShaderStageFlagBits>(
                    VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                    VK_SHADER_STAGE_MISS_BIT_KHR);
            default:
                return VK_SHADER_STAGE_ALL;
        }
    }();
    vstd::vector<vstd::vector<VkDescriptorSetLayoutBinding>> bindings;
    vstd::vector<uint8_t> is_bindless;
    vstd::vector<vstd::vector<VkDescriptorBindingFlags>> binding_flags;
    bindings.resize(interface_plan.descriptor_set_count);
    is_bindless.resize(interface_plan.descriptor_set_count);
    binding_flags.resize(interface_plan.descriptor_set_count);
    LUISA_ASSERT(
        device->samplers().size() ==
            detail::descriptor_interface_sampler_count,
        "Vulkan device exposes {} immutable samplers but the shader ABI requires {}.",
        device->samplers().size(),
        detail::descriptor_interface_sampler_count);
    for (auto &&i : binds) {
        // ConstantValue is the legacy HLSL push-constant marker. It never
        // owns a Vulkan descriptor binding, even when set 0 is otherwise empty.
        if (i.type == hlsl::ShaderVariableType::ConstantValue) { continue; }
        auto &vec = bindings[i.space_index];
        vec.resize(std::max<size_t>(vec.size(), i.register_index + 1));
        auto &flags = binding_flags[i.space_index];
        flags.resize(vec.size());
        auto &v = vec[i.register_index];
        v.pImmutableSamplers = nullptr;
        v.binding = i.register_index;
        switch (i.type) {
            case hlsl::ShaderVariableType::ConstantBuffer:
            case hlsl::ShaderVariableType::ConstantValue:
            case hlsl::ShaderVariableType::CBVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case hlsl::ShaderVariableType::SRVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case hlsl::ShaderVariableType::UAVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case hlsl::ShaderVariableType::StructuredBuffer:
            case hlsl::ShaderVariableType::RWStructuredBuffer:
            case hlsl::ShaderVariableType::SPIRVAccelInstance:
            case hlsl::ShaderVariableType::SPIRVAccelInstanceRW:
            case hlsl::ShaderVariableType::SPIRVBindlessBufferMetadata:
            case hlsl::ShaderVariableType::SPIRVIndirectDispatch:
            case hlsl::ShaderVariableType::UAVBufferHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case hlsl::ShaderVariableType::SPIRVAccel:
                v.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                break;
            case hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                v.pImmutableSamplers = device->samplers().data();
                break;
            default:
                assert(false);
                break;
        }
        auto bindless = i.array_size == ~0u;
        is_bindless[i.space_index] |= bindless;
        if (bindless) {
            LUISA_ASSERT(device->bindless_heap_capacity() > 0u,
                         "Vulkan bindless shader layout requested without a "
                         "valid descriptor heap capacity.");
            flags[i.register_index] =
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        }
        v.descriptorCount = bindless ?
                                device->bindless_heap_capacity() :
                                i.array_size;
        v.stageFlags = (v.pImmutableSamplers != nullptr || i.array_size == ~0u) ? VK_SHADER_STAGE_ALL : stage_bits;
    }
    LUISA_ASSERT(
        bindings.size() == interface_plan.descriptor_set_count &&
            bindings[0u].size() ==
                interface_plan.local_binding_count,
        "Vulkan descriptor layout construction drifted from its validated plan.");
    vstd::push_back_all(_binds, binds);
    _desc_set_layout.reserve(bindings.size());
    auto make_layout_create_info = [&](size_t set_index,
                                       VkDescriptorSetLayoutBindingFlagsCreateInfo &flags_info) {
        auto &set_bindings = bindings[set_index];
        auto &set_flags = binding_flags[set_index];
        LUISA_ASSERT(
            set_bindings.size() <=
                    detail::descriptor_interface_max_local_bindings &&
                set_flags.size() == set_bindings.size(),
            "Vulkan descriptor layout set {} exceeded its validated binding count.",
            set_index);
        flags_info = VkDescriptorSetLayoutBindingFlagsCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = static_cast<uint32_t>(set_flags.size()),
            .pBindingFlags = set_flags.data()};
        auto bindless = is_bindless[set_index] != 0u;
        return VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = bindless ? &flags_info : nullptr,
            .flags = bindless ?
                         static_cast<VkDescriptorSetLayoutCreateFlags>(
                             VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT) :
                         VkDescriptorSetLayoutCreateFlags{0u},
            .bindingCount =
                static_cast<uint32_t>(set_bindings.size()),
            .pBindings = set_bindings.data()};
    };
    // Query every set before creating the first Vulkan object. This covers
    // implementation-specific layout support (including update-after-bind
    // and acceleration-structure descriptors) beyond the numeric plan.
    for (auto set_index = 0u;
         set_index < interface_plan.descriptor_set_count; ++set_index) {
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindless_binding_flags{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        auto descriptor_layout = make_layout_create_info(
            set_index, bindless_binding_flags);
        VkDescriptorSetLayoutSupport support{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT};
        vkGetDescriptorSetLayoutSupport(
            device->logic_device(), &descriptor_layout, &support);
        LUISA_ASSERT(
            support.supported == VK_TRUE,
            "Vulkan device rejected validated descriptor set layout {}.",
            set_index);
    }
    for (auto set_index = 0u;
         set_index < interface_plan.descriptor_set_count; ++set_index) {
        VkDescriptorSetLayoutBindingFlagsCreateInfo bindless_binding_flags{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        auto descriptor_layout = make_layout_create_info(
            set_index, bindless_binding_flags);
        auto &r = _desc_set_layout.emplace_back();
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logic_device(), &descriptor_layout, Device::alloc_callbacks(), &r));
    }
    VkPushConstantRange push_const_range{
        VkShaderStageFlags(stage_bits),
        0,
        push_constant_size};
    VkPipelineLayoutCreateInfo pipeline_layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = interface_plan.descriptor_set_count,
        .pSetLayouts = _desc_set_layout.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_const_range};
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(
            device->logic_device(),
            &pipeline_layout_create_info,
            Device::alloc_callbacks(),
            &_pipeline_layout));

    if (!constant_ubo_data.empty()) {
        _has_constant_ubo = true;
        _constant_ubo = luisa::make_unique<UploadBuffer>(device, constant_ubo_data.size_bytes());
        _constant_ubo->copy_from(constant_ubo_data.data(), 0, constant_ubo_data.size_bytes());
        _constant_ubo->flush_host();
    }
}
Shader::~Shader() {
    for (auto &&i : _desc_set_layout) {
        vkDestroyDescriptorSetLayout(device()->logic_device(), i, Device::alloc_callbacks());
    }
    vkDestroyPipelineLayout(device()->logic_device(), _pipeline_layout, Device::alloc_callbacks());
}
}// namespace lc::vk
