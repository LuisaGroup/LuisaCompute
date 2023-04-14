#include "shader.h"
#include "../log.h"
#include "device.h"
namespace lc::vk {
Shader::Shader(
    Device *device,
    ShaderTag tag,
    vstd::span<Bind const> binds)
    : Resource{device} {
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
        bindings.resize(std::max<size_t>(bindings.size(), i.table_idx + 1));
        auto &vec = bindings[i.table_idx];
        vec.resize(std::max<size_t>(vec.size(), i.binding_idx + 1));
        auto &v = vec[i.binding_idx];
        v.binding = i.binding_idx;
        v.descriptorType = static_cast<VkDescriptorType>(i.bind_type);
        v.descriptorCount = 1;
        v.stageFlags = stage_bits;
        v.pImmutableSamplers = nullptr;
    }

    descriptorSetLayouts.reserve(bindings.size());
    for (auto &&i : bindings) {
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint>(bindings.size()),
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