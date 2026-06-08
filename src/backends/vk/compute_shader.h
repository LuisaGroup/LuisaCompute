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
    VkPipeline _pipeline{};
    uint3 _block_size;
public:
    static bool verify_type_md5(luisa::span<const luisa::compute::Type *const> arg_types, vstd::MD5 md5);
    auto pipeline() const { return _pipeline; }
    bool serialize_pso(vstd::vector<std::byte> &result) const override;
    auto block_size() const { return _block_size; }
    ComputeShader(
        Device *device,
        uint3 block_size,
        vstd::span<hlsl::Property const> binds,
        vstd::vector<SavedArgument> &&saved_args,
        vstd::span<uint const> spv_code,
        vstd::vector<Argument> &&captured,
        vstd::span<std::byte const> cache_code,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers,
        luisa::span<const std::byte> constant_ubo_data = {},
        uint validation_count = 0);
    ~ComputeShader();
    static ComputeShader *compile(
        BinaryIO const *bin_io,
        Device *device,
        vstd::vector<SavedArgument> &&saved_args,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::optional<vstd::MD5> const &code_md5,
        vstd::vector<Argument> &&bindings,
        uint3 block_size,
        vstd::string_view file_name,
        SerdeType serde_type,
        uint shader_model,
        bool unsafe_math,
        uint validation_count = 0);
};
}// namespace lc::vk
