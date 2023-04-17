#pragma once
#include "shader.h"
#include <runtime/rhi/resource.h>
#include <vstl/md5.h>
#include <vstl/functional.h>
#include <ast/function.h>
#include "serde_type.h"

namespace luisa{
    class BinaryIO;
}
namespace lc::hlsl {
struct CodegenResult;
}
namespace lc::vk {
    using namespace luisa;
using namespace luisa::compute;
class ComputeShader : public Shader {
    VkPipelineCache _pipe_cache{};
    VkPipeline _pipeline;

public:
    bool serialize_pso(vstd::vector<std::byte> &result) const override;
    ComputeShader(
        Device *device,
        vstd::span<hlsl::Property const> binds,
        vstd::span<uint const> spv_code,
        vstd::vector<Argument> &&captured,
        vstd::span<std::byte const> cache_code);
    ~ComputeShader();
    static ComputeShader *compile(
        BinaryIO const *bin_io,
        Device *device,
        Function kernel,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::optional<vstd::MD5> const &code_md5,
        vstd::vector<Argument> &&bindings,
        uint3 blockSize,
        uint shaderModel,
        vstd::string_view file_name,
        SerdeType serde_type,
        bool unsafe_mat);
};
}// namespace lc::vk