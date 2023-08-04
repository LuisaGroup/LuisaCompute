#include "compute_shader.h"
#include "device.h"
#include "log.h"
#include "../common/hlsl/hlsl_codegen.h"
#include <luisa/core/stl/filesystem.h>
#include "shader_serializer.h"
#include <luisa/core/logging.h>
#include "../common/hlsl/shader_compiler.h"

static constexpr bool PRINT_CODE = false;

namespace lc::vk {
ComputeShader::ComputeShader(
    Device *device,
    vstd::span<hlsl::Property const> binds,
    vstd::span<uint const> spv_code,
    vstd::vector<Argument> &&captured,
    vstd::span<std::byte const> cache_code)
    : Shader{device, ShaderTag::ComputeShader, std::move(captured), binds} {
    VkPipelineCacheCreateInfo pso_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (!cache_code.empty()) {
        pso_ci.initialDataSize = cache_code.size();
        pso_ci.pInitialData = cache_code.data();
    }
    VK_CHECK_RESULT(vkCreatePipelineCache(device->logic_device(), &pso_ci, nullptr, &_pipe_cache));
    VkShaderModuleCreateInfo module_create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_code.size_bytes(),
        .pCode = spv_code.data()};
    VkShaderModule shader_module;
    VK_CHECK_RESULT(vkCreateShaderModule(device->logic_device(), &module_create_info, nullptr, &shader_module));
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(device->logic_device(), shader_module, nullptr);
    });
    VkComputePipelineCreateInfo pipe_ci{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = 0,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main"},
        .layout = _pipeline_layout};

    VK_CHECK_RESULT(vkCreateComputePipelines(device->logic_device(), _pipe_cache, 1, &pipe_ci, nullptr, &_pipeline));
}
bool ComputeShader::serialize_pso(vstd::vector<std::byte> &result) const {
    auto last_size = result.size();
    size_t pso_size = 0;
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, nullptr));
    result.resize_uninitialized(last_size + pso_size);
    if (pso_size <= sizeof(VkPipelineCacheHeaderVersionOne)) return false;
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, result.data() + last_size));
    result.resize_uninitialized(last_size + pso_size);
    return true;
}
ComputeShader::~ComputeShader() {
    vkDestroyPipeline(device()->logic_device(), _pipeline, nullptr);
    vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, nullptr);
}
ComputeShader *ComputeShader::compile(
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
    bool unsafe_math) {

    auto result = ShaderSerializer::try_deser_compute(device, code_md5, std::move(bindings), file_name, serde_type, bin_io);
    // cache invalid, need compile
    bool write_cache = !file_name.empty();
    if (!result.shader) {
        auto str = codegen();
        vstd::MD5 md5;
        if (write_cache) {
            if (code_md5) {
                md5 = *code_md5;
            } else {
                md5 = vstd::MD5({reinterpret_cast<uint8_t const *>(str.result.data() + str.immutableHeaderSize), str.result.size() - str.immutableHeaderSize});
            }
        }
        if constexpr (PRINT_CODE) {
            auto f = fopen("hlsl_output.hlsl", "ab");
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
        auto comp_result = Device::Compiler()->compile_compute(
            str.result.view(),
            true,
            shader_model,
            unsafe_math,
            true);
        return comp_result.multi_visit_or(
            vstd::UndefEval<ComputeShader *>{},
            [&](vstd::unique_ptr<hlsl::DxcByteBlob> const &buffer) {
                auto shader = new ComputeShader(
                    device,
                    str.properties,
                    {reinterpret_cast<const uint *>(buffer->data()), buffer->size() / sizeof(uint)},
                    std::move(bindings),
                    {});
                if (write_cache) {
                    ShaderSerializer::serialize_bytecode(
                        shader->binds(),
                        md5,
                        vstd::MD5(vstd::MD5::MD5Data{0, 0}),
                        kernel.block_size(),
                        file_name,
                        {reinterpret_cast<const uint *>(buffer->data()), buffer->size() / sizeof(uint)},
                        serde_type,
                        bin_io);
                    ShaderSerializer::serialize_pso(
                        device,
                        shader,
                        md5,
                        bin_io);
                }
                return shader;
            },
            [](auto &&err) {
                LUISA_ERROR("Compile Error: {}", err);
                return nullptr;
            });
    }
    return static_cast<ComputeShader *>(result.shader);
}
}// namespace lc::vk
