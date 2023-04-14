#pragma once
#include "shader.h"
namespace lc::vk {
class ComputeShader : public Shader {
    VkPipelineCache _pipe_cache{};
    VkPipeline _pipeline;

public:
    void serialize_pso(vstd::vector<std::byte> &result);
    ComputeShader(
        Device *device,
        vstd::span<Bind const> binds,
        vstd::span<uint const> spv_code,
        vstd::span<std::byte const> cache_code);
    ~ComputeShader();
};
}// namespace lc::vk