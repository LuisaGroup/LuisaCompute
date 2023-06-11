#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
#include "../common/hlsl/shader_property.h"
#include <luisa/runtime/rhi/argument.h>
namespace lc::vk {
using namespace luisa::compute;
class Shader : public Resource {
public:
    enum class ShaderTag : uint {
        ComputeShader,
        RasterShader
    };

protected:
    VkPipelineLayout _pipeline_layout;
    vstd::vector<hlsl::Property> _binds;
    vstd::vector<Argument> _captured;

public:
    auto pipeline_layout() const { return _pipeline_layout; }
    virtual bool serialize_pso(vstd::vector<std::byte> &result) const { return false; }
    auto binds() const { return vstd::span<const hlsl::Property>{_binds}; }
    Shader(
        Device *device,
        ShaderTag tag,
        vstd::vector<Argument> &&captured,
        vstd::span<hlsl::Property const> binds);
    virtual ~Shader();
};
}// namespace lc::vk
