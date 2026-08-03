#pragma once
#include "shader.h"
#include "pipeline_ref.h"
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
    PipelineRef *_pipeline_ref{};
    uint3 _block_size;
    VkPipelineCreateFlags _pipeline_create_flags{};
public:
    [[nodiscard]] static constexpr VkPipelineCreateFlags
    pipeline_create_flags(bool enable_driver_optimization) noexcept {
        return enable_driver_optimization ?
                   VkPipelineCreateFlags{0u} :
                   VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    auto pipeline() const { return _pipeline_ref ? _pipeline_ref->pipeline : VK_NULL_HANDLE; }
    auto pipeline_cache() const { return _pipeline_ref ? _pipeline_ref->pipeline_cache : VK_NULL_HANDLE; }
    PipelineRef *pipeline_ref() const noexcept override { return _pipeline_ref; }
    bool serialize_pso(vstd::vector<std::byte> &result) const override;
    [[nodiscard]] uint64_t pipeline_cache_identity() const noexcept override {
        return _pipeline_create_flags;
    }
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
        uint validation_count = 0,
        luisa::optional<uint8_t> required_subgroup_size = luisa::nullopt,
        uint32_t push_constant_size = 32u,
        detail::ShaderCodegenDialect codegen_dialect =
            detail::ShaderCodegenDialect::HLSL_SPIRV,
        bool enable_driver_optimization = true);
    ~ComputeShader();
    static ComputeShader *compile(
        BinaryIO const *bin_io,
        Device *device,
        vstd::vector<SavedArgument> &&saved_args,
        vstd::function<hlsl::CodegenResult()> const &codegen,
        vstd::optional<vstd::MD5> const &code_md5,
        luisa::optional<vstd::MD5> expected_type_md5,
        vstd::vector<Argument> &&bindings,
        uint3 block_size,
        vstd::string_view file_name,
        SerdeType serde_type,
        uint shader_model,
        bool unsafe_math,
        uint validation_count = 0,
        luisa::optional<uint8_t> required_subgroup_size = luisa::nullopt,
        bool requires_sampler_anisotropy = false,
        uint32_t push_constant_size = 32u,
        detail::ShaderCodegenDialect codegen_dialect =
            detail::ShaderCodegenDialect::HLSL_SPIRV,
        bool enable_driver_optimization = true);
    static ComputeShader *compile_builtin_hlsl_to_spirv(
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
        uint validation_count = 0,
        luisa::optional<uint8_t> required_subgroup_size = luisa::nullopt,
        bool requires_sampler_anisotropy = false,
        uint32_t push_constant_size = 32u);
};
}// namespace lc::vk
