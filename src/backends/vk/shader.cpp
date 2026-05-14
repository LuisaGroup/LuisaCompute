#include "shader.h"
#include "log.h"
#include "device.h"
namespace lc::vk {
SavedArgument::SavedArgument(luisa::compute::Type const *type) {
    if (luisa::to_underlying(type->tag()) < luisa::to_underlying(luisa::compute::Type::Tag::BUFFER)) {
        struct_size = type->size();
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
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers)
    : Resource{device}, _captured{std::move(captured)}, _saved_arguments(std::move(saved_arguments)),
      _shader_tag(tag), _use_tex2d_bindless(use_tex2d_bindless), _use_tex3d_bindless(use_tex3d_bindless), _use_buffer_bindless(use_buffer_bindless), _printers(std::move(printers)) {
    if ((!device->enable_bindless()) && (use_tex2d_bindless || use_tex3d_bindless || use_buffer_bindless)) [[unlikely]] {
        LUISA_ERROR("Bindless not enabled, shader can not be load.");
    }
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
    vstd::fixed_vector<bool, 4> is_bindless;
    for (auto &&i : binds) {
        bindings.resize(std::max<size_t>(bindings.size(), i.space_index + 1));
        is_bindless.resize(bindings.size());
        auto &vec = bindings[i.space_index];
        vec.resize(std::max<size_t>(vec.size(), i.register_index + 1));
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
        is_bindless[i.space_index] = i.array_size == ~0u;
        v.descriptorCount = i.array_size == ~0u ? 262144u : i.array_size;
        v.stageFlags = (v.pImmutableSamplers != nullptr || i.array_size == ~0u) ? VK_SHADER_STAGE_ALL : stage_bits;
    }
    vstd::push_back_all(_binds, binds);
    _desc_set_layout.reserve(bindings.size());
    auto is_bindless_iter = is_bindless.begin();
    VkDescriptorBindingFlags desc_binding_flag =
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindless_binding_flags{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 1,
        .pBindingFlags = &desc_binding_flag};
    for (auto &&i : bindings) {
        VkDescriptorSetLayoutCreateInfo descriptor_layout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = (*is_bindless_iter) ? &bindless_binding_flags : nullptr,
            .flags = (*is_bindless_iter) ? (VkDescriptorSetLayoutCreateFlags)VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : (VkDescriptorSetLayoutCreateFlags)0,
            .bindingCount = static_cast<uint>(i.size()),
            .pBindings = i.data()};
        auto &r = _desc_set_layout.emplace_back();
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logic_device(), &descriptor_layout, Device::alloc_callbacks(), &r));
        ++is_bindless_iter;
    }
    VkPushConstantRange push_const_range{
        VkShaderStageFlags(stage_bits),
        0,
        16};
    VkPipelineLayoutCreateInfo pipeline_layout_create_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint>(_desc_set_layout.size()),
        .pSetLayouts = _desc_set_layout.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_const_range};
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(
            device->logic_device(),
            &pipeline_layout_create_info,
            Device::alloc_callbacks(),
            &_pipeline_layout));
}
Shader::~Shader() {
    for (auto &&i : _desc_set_layout) {
        vkDestroyDescriptorSetLayout(device()->logic_device(), i, Device::alloc_callbacks());
    }
    vkDestroyPipelineLayout(device()->logic_device(), _pipeline_layout, Device::alloc_callbacks());
}
}// namespace lc::vk
