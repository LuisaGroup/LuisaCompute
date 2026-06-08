#pragma once
#include "resource.h"
#include <volk.h>
#include "../common/hlsl/shader_property.h"
#include <luisa/ast/usage.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/argument.h>
#include "buffer.h"
#include "texture.h"
namespace lc::vk {
using namespace luisa::compute;
struct SavedArgument {
    luisa::compute::Type::Tag tag{};
    Usage var_usage{};
    uint struct_size{};
    SavedArgument() = default;
    SavedArgument(Function kernel, Variable const &var) : SavedArgument(var.type()) {
        var_usage = kernel.variable_usage(var.uid());
    }
    SavedArgument(Usage usage, Variable const &var) : SavedArgument(var.type()) {
        var_usage = usage;
    }
    explicit SavedArgument(luisa::compute::Type const *type);
};
class Shader : public Resource {
public:
    enum class ShaderTag : uint {
        kComputeShader,
        kRasterShader,
        kRayTracingShader
    };

protected:
    vstd::vector<VkDescriptorSetLayout> _desc_set_layout;
    VkPipelineLayout _pipeline_layout{};
    vstd::vector<hlsl::Property> _binds;
    vstd::vector<Argument> _captured;
    vstd::vector<SavedArgument> _saved_arguments;
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> _printers;
    luisa::unique_ptr<class UploadBuffer> _constant_ubo;
    ShaderTag _shader_tag;
    bool _has_constant_ubo{false};
    bool _use_tex2d_bindless;
    bool _use_tex3d_bindless;
    bool _use_buffer_bindless;
    uint _validation_count{0};
public:
    auto pipeline_layout() const { return _pipeline_layout; }
    auto shader_tag() const { return _shader_tag; }
    virtual bool serialize_pso(vstd::vector<std::byte> &result) const { return false; }
    auto binds() const { return vstd::span<const hlsl::Property>{_binds}; }
    auto captured() const { return vstd::span<const Argument>{_captured}; }
    auto desc_set_layout() const { return vstd::span{_desc_set_layout}; }
    auto saved_arguments() const { return vstd::span{_saved_arguments}; }
    bool use_tex2d_bindless() const { return _use_tex2d_bindless; }
    bool use_tex3d_bindless() const { return _use_tex3d_bindless; }
    bool use_buffer_bindless() const { return _use_buffer_bindless; }
    uint validation_count() const { return _validation_count; }
    bool has_constant_ubo() const { return _has_constant_ubo; }
    UploadBuffer const *constant_ubo() const { return _constant_ubo.get(); }
    auto printers() const { return luisa::span{_printers}; }
    Shader(
        Device *device,
        ShaderTag tag,
        vstd::vector<Argument> &&captured,
        vstd::vector<SavedArgument> &&saved_arguments,
        vstd::span<hlsl::Property const> binds,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers,
        luisa::span<const std::byte> constant_ubo_data = {},
        uint validation_count = 0);
    virtual ~Shader();
    vstd::span<VkDescriptorSet> allocate_desc_set(VkDescriptorPool pool, vstd::vector<VkDescriptorSet> &descs) const;
    void update_desc_set(
        VkDescriptorSet set,
        vstd::vector<VkWriteDescriptorSet> &write_buffer,
        vstd::vector<VkImageView> &img_view_buffer,
        vstd::span<vstd::variant<BufferView, TexView>> texs);
};
}// namespace lc::vk
