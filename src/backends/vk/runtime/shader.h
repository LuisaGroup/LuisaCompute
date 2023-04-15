#pragma once
#include "resource.h"
#include <vulkan/vulkan.h>
namespace lc::vk {
class Shader : public Resource {
public:
    enum class BindTag : uint {
        Sampler = VK_DESCRIPTOR_TYPE_SAMPLER,
        RWImage = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        SampleImage = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        StructuredBuffer = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        ConstantBuffer = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        Accel = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    };
    struct Bind {
        uint table_idx;
        uint binding_idx;
        BindTag bind_type;
    };
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
        vstd::span<Bind const> binds);
    ~Shader();
};
}// namespace lc::vk