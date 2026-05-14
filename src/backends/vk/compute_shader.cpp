#include "compute_shader.h"
#include "device.h"
#include "log.h"
#include "../common/hlsl/hlsl_codegen.h"
#include <luisa/core/stl/filesystem.h>
#include "shader_serializer.h"
#include <luisa/core/logging.h>
#include "../common/hlsl/shader_compiler.h"
#include <cstdio>
#include <chrono>
#include <atomic>

// Track whether an RT motion blur pipeline has ever been created and destroyed on this device.
// This is used to prove the NVIDIA driver bug: vkCreateComputePipelines deadlocks
// ONLY after an RT motion blur pipeline has been used and destroyed.
static std::atomic<int> g_rt_motion_pipeline_created{0};
static std::atomic<int> g_rt_motion_pipeline_destroyed{0};
static std::atomic<int> g_compute_pipeline_create_count{0};

static void cs_trace_log(const char *fmt, ...) {
    static FILE *f = nullptr;
    if (!f) f = fopen("D:\\smaray_vkpipeline_trace.log", "a");
    if (!f) return;
    static auto t0 = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
    fprintf(f, "[%lld ms] ", (long long)ms);
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fprintf(f, "\n");
    fflush(f);
}

// Called from rt_shader.cpp when RT motion blur pipeline is created
extern "C" void smaray_notify_rt_motion_pipeline_created() {
    g_rt_motion_pipeline_created.fetch_add(1);
    cs_trace_log("[RT_MOTION] pipeline CREATED (total created=%d, destroyed=%d)",
                 g_rt_motion_pipeline_created.load(), g_rt_motion_pipeline_destroyed.load());
}
// Called from rt_shader.cpp when RT motion blur pipeline is destroyed
extern "C" void smaray_notify_rt_motion_pipeline_destroyed() {
    g_rt_motion_pipeline_destroyed.fetch_add(1);
    cs_trace_log("[RT_MOTION] pipeline DESTROYED (total created=%d, destroyed=%d)",
                 g_rt_motion_pipeline_created.load(), g_rt_motion_pipeline_destroyed.load());
}

namespace lc::vk {

bool ComputeShader::verify_type_md5(luisa::span<const luisa::compute::Type *const> arg_types, vstd::MD5 md5) {
    return hlsl::CodegenUtility::GetTypeMD5(arg_types) == md5;
}
ComputeShader::ComputeShader(
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
    vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers)
    : Shader{device, ShaderTag::kComputeShader, std::move(captured), std::move(saved_args), binds, use_tex2d_bindless, use_tex3d_bindless, use_buffer_bindless, std::move(printers)}, _block_size(block_size) {
    VkPipelineCacheCreateInfo pso_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (!cache_code.empty()) {
        pso_ci.initialDataSize = cache_code.size();
        pso_ci.pInitialData = cache_code.data();
    }
    VK_CHECK_RESULT(vkCreatePipelineCache(device->logic_device(), &pso_ci, Device::alloc_callbacks(), &_pipe_cache));
    VkShaderModuleCreateInfo module_create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_code.size_bytes(),
        .pCode = spv_code.data()};
    VkShaderModule shader_module;
    VK_CHECK_RESULT(vkCreateShaderModule(device->logic_device(), &module_create_info, Device::alloc_callbacks(), &shader_module));
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(device->logic_device(), shader_module, Device::alloc_callbacks());
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

    int pipeline_id = g_compute_pipeline_create_count.fetch_add(1);
    bool rt_was_used_and_destroyed = (g_rt_motion_pipeline_destroyed.load() > 0);
    cs_trace_log("[vkCreateComputePipelines] ENTER #%d, rt_motion_created=%d, rt_motion_destroyed=%d, "
                 "rt_was_used_and_destroyed=%d",
                 pipeline_id,
                 g_rt_motion_pipeline_created.load(),
                 g_rt_motion_pipeline_destroyed.load(),
                 (int)rt_was_used_and_destroyed);
    VK_CHECK_RESULT(vkCreateComputePipelines(device->logic_device(), _pipe_cache, 1, &pipe_ci, Device::alloc_callbacks(), &_pipeline));
    cs_trace_log("[vkCreateComputePipelines] EXIT #%d, pipeline=%p", pipeline_id, (void*)_pipeline);
}
bool ComputeShader::serialize_pso(vstd::vector<std::byte> &result) const {
    auto last_size = result.size();
    size_t pso_size = 0;
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, nullptr));
    luisa::vector_resize(result, last_size + pso_size);
    if (pso_size <= sizeof(VkPipelineCacheHeaderVersionOne)) return false;
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, result.data() + last_size));
    luisa::vector_resize(result, last_size + pso_size);
    return true;
}
ComputeShader::~ComputeShader() {
    vkDestroyPipeline(device()->logic_device(), _pipeline, Device::alloc_callbacks());
    vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, Device::alloc_callbacks());
}
ComputeShader *ComputeShader::compile(
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
        if (Device::print_code()) {
            auto f = fopen("hlsl_output.hlsl", "ab");
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
        auto comp_result = Device::compiler()->compile_compute(
            str.result.view(),
            true,
            shader_model,
            unsafe_math,
            true,
            false);
        return comp_result.multi_visit_or(
            vstd::UndefEval<ComputeShader *>{},
            [&](hlsl::ComUniquePtr<IDxcBlob> const &buffer) {
                auto shader = new ComputeShader(
                    device,
                    block_size,
                    str.properties,
                    std::move(saved_args),
                    {reinterpret_cast<const uint *>(buffer->GetBufferPointer()), buffer->GetBufferSize() / sizeof(uint)},
                    std::move(bindings),
                    {},
                    str.useTex2DBindless,
                    str.useTex3DBindless,
                    str.useBufferBindless,
                    std::move(str.printers));
                if (write_cache) {
                    ShaderSerializer::serialize_bytecode(
                        shader->binds(),
                        shader->saved_arguments(),
                        md5,
                        vstd::MD5(vstd::MD5::MD5Data{0, 0}),
                        block_size,
                        file_name,
                        {reinterpret_cast<const uint *>(buffer->GetBufferPointer()), buffer->GetBufferSize() / sizeof(uint)},
                        serde_type,
                        bin_io,
                        str.useTex2DBindless,
                        str.useTex3DBindless,
                        str.useBufferBindless,
                        shader->printers());
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
