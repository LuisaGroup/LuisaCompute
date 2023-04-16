#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
#include <backends/common/hlsl/shader_property.h>
namespace lc::vk {
class Shader : public Resource {
public:
    enum class ShaderTag : uint {
        ComputeShader,
        RasterShader
    };

protected:
    VkPipelineLayout _pipeline_layout;

public:
    auto pipeline_layout() const { return _pipeline_layout; }
    Shader(
        Device *device,
        ShaderTag tag,
        vstd::span<hlsl::Property const> binds);
    ~Shader();
};
}// namespace lc::vk