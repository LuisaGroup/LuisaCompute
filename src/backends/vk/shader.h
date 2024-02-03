#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
#include "../common/hlsl/shader_property.h"
#include <luisa/runtime/rhi/argument.h>
#include "buffer.h"
#include "texture.h"
namespace lc::vk {
using namespace luisa::compute;
class Shader : public Resource {
public:
    enum class ShaderTag : uint {
        ComputeShader,
        RasterShader
    };

protected:
    vstd::vector<VkDescriptorSetLayout> _desc_set_layout;
    VkPipelineLayout _pipeline_layout;
    vstd::vector<hlsl::Property> _binds;
    vstd::vector<Argument> _captured;

public:
    auto pipeline_layout() const { return _pipeline_layout; }
    virtual bool serialize_pso(vstd::vector<std::byte> &result) const { return false; }
    auto binds() const { return vstd::span<const hlsl::Property>{_binds}; }
    auto captured() const { return vstd::span<const Argument>{_captured}; }
    auto desc_set_layout() const { return vstd::span{_desc_set_layout}; }
    Shader(
        Device *device,
        ShaderTag tag,
        vstd::vector<Argument> &&captured,
        vstd::span<hlsl::Property const> binds);
    virtual ~Shader();
    vstd::span<VkDescriptorSet> allocate_desc_set(VkDescriptorPool pool, vstd::vector<VkDescriptorSet> &descs);
    void update_desc_set(
        VkDescriptorSet set,
        vstd::vector<VkWriteDescriptorSet>& write_buffer,
        vstd::vector<VkImageView>& img_view_buffer,
        vstd::span<vstd::variant<BufferView, TexView>> texs);
};
}// namespace lc::vk
