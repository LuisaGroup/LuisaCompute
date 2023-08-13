#pragma once
#include "shader.h"
#include <luisa/runtime/rhi/resource.h>
#include <luisa/vstl/md5.h>
#include <luisa/vstl/functional.h>
#include <luisa/ast/function.h>
#include "serde_type.h"

namespace luisa {
class BinaryIO;
}// namespace luisa
namespace lc::hlsl {
struct CodegenResult;
}// namespace lc::hlsl
namespace lc::vk {
using namespace luisa;
using namespace luisa::compute;
class ComputeShader : public Shader {
    VkPipelineCache _pipe_cache{};
    VkPipeline _pipeline;

public:
    auto pipeline() const { return _pipeline; }
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
        vstd::string_view file_name,
        SerdeType serde_type,
        uint shader_model,
        bool unsafe_math);
};
}// namespace lc::vk
