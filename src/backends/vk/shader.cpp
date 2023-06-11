#include "shader.h"
#include "log.h"
#include "device.h"
namespace lc::vk {
Shader::Shader(
    Device *device,
    ShaderTag tag,
    vstd::vector<Argument> &&captured,
    vstd::span<hlsl::Property const> binds)
    : Resource{device}, _captured{std::move(captured)} {
    VkShaderStageFlagBits stage_bits = [&]() -> VkShaderStageFlagBits {
        switch (tag) {
            case ShaderTag::ComputeShader:
                return VK_SHADER_STAGE_COMPUTE_BIT;
            case ShaderTag::RasterShader:
                return static_cast<VkShaderStageFlagBits>(
                    VK_SHADER_STAGE_VERTEX_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT);
            default:
                return VK_SHADER_STAGE_ALL;
        }
    }();
    vstd::vector<VkDescriptorSetLayout> descriptorSetLayouts;
    vstd::vector<vstd::vector<VkDescriptorSetLayoutBinding>> bindings;
    for (auto &&i : binds) {
        bindings.resize(std::max<size_t>(bindings.size(), i.space_index + 1));
        auto &vec = bindings[i.space_index];
        vec.resize(std::max<size_t>(vec.size(), i.register_index + 1));
        auto &v = vec[i.register_index];
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
            case hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            default:
                assert(false);
                break;
        }
        v.descriptorCount = i.array_size;
        v.stageFlags = stage_bits;
        v.pImmutableSamplers = nullptr;
        vstd::push_back_all(_binds, binds);
    }

    descriptorSetLayouts.reserve(bindings.size());
    for (auto &&i : bindings) {
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint>(i.size()),
            .pBindings = i.data()};
        auto &r = descriptorSetLayouts.emplace_back();
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logic_device(), &descriptorLayout, nullptr, &r));
    }

    auto disposer = vstd::scope_exit([&] {
        for (auto &&i : descriptorSetLayouts) {
            vkDestroyDescriptorSetLayout(device->logic_device(), i, nullptr);
        }
    });
    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.data()};
    VK_CHECK_RESULT(vkCreatePipelineLayout(device->logic_device(), &pPipelineLayoutCreateInfo, nullptr, &_pipeline_layout));
}
Shader::~Shader() {
    vkDestroyPipelineLayout(device()->logic_device(), _pipeline_layout, nullptr);
}
}// namespace lc::vk
