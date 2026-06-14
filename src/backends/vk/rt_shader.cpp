#include "rt_shader.h"
#include "device.h"
#include "log.h"
#include "../common/hlsl/hlsl_codegen.h"
#include <luisa/core/stl/filesystem.h>
#include "shader_serializer.h"
#include <luisa/core/logging.h>
#include "../common/hlsl/shader_compiler.h"
#include "vk_allocator.h"
#include "spirv_motion_patch.h"

namespace lc::vk {

RayTracingShader::RayTracingShader(
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
    : Shader{device, ShaderTag::kRayTracingShader, std::move(captured), std::move(saved_args), binds, use_tex2d_bindless, use_tex3d_bindless, use_buffer_bindless, std::move(printers)}, _block_size(block_size) {

    VkPipelineCacheCreateInfo pso_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (!cache_code.empty()) {
        pso_ci.initialDataSize = cache_code.size();
        pso_ci.pInitialData = cache_code.data();
    }
    VK_CHECK_RESULT(vkCreatePipelineCache(device->logic_device(), &pso_ci, Device::alloc_callbacks(), &_pipe_cache));

    // Create shader module from SPIR-V
    VkShaderModuleCreateInfo module_create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_code.size_bytes(),
        .pCode = spv_code.data()};
    VkShaderModule shader_module;
    VK_CHECK_RESULT(vkCreateShaderModule(device->logic_device(), &module_create_info, Device::alloc_callbacks(), &shader_module));
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(device->logic_device(), shader_module, Device::alloc_callbacks());
    });

    // Shader stages: raygen, miss, closesthit
    VkPipelineShaderStageCreateInfo stages[3]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = shader_module;
    stages[0].pName = "main_raygen";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = shader_module;
    stages[1].pName = "main_miss";
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = shader_module;
    stages[2].pName = "main_closesthit";

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR rt_pipeline_ci{};
    rt_pipeline_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rt_pipeline_ci.stageCount = 3;
    rt_pipeline_ci.pStages = stages;
    rt_pipeline_ci.groupCount = 3;
    rt_pipeline_ci.pGroups = groups;
    rt_pipeline_ci.maxPipelineRayRecursionDepth = 1;
    rt_pipeline_ci.layout = _pipeline_layout;
    // Allow motion blur in the ray tracing pipeline.
    // VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV is required for
    // OpTraceRayMotionNV to work correctly with motion TLAS/BLAS.
    rt_pipeline_ci.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;

    VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(
        device->logic_device(), VK_NULL_HANDLE, _pipe_cache,
        1, &rt_pipeline_ci, Device::alloc_callbacks(), &_pipeline));

    // Query RT pipeline properties for SBT
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props{};
    rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rt_props;
    vkGetPhysicalDeviceProperties2(device->physical_device(), &props2);

    auto handle_size = rt_props.shaderGroupHandleSize;
    auto handle_alignment = rt_props.shaderGroupHandleAlignment;
    auto base_alignment = rt_props.shaderGroupBaseAlignment;

    auto aligned_handle_size = (handle_size + handle_alignment - 1) & ~(handle_alignment - 1);
    auto raygen_size = (aligned_handle_size + base_alignment - 1) & ~(base_alignment - 1);
    auto miss_size = (aligned_handle_size + base_alignment - 1) & ~(base_alignment - 1);
    auto hit_size = (aligned_handle_size + base_alignment - 1) & ~(base_alignment - 1);
    auto sbt_size = raygen_size + miss_size + hit_size;

    // Get shader group handles
    auto group_count = 3u;
    vstd::vector<uint8_t> handles(handle_size * group_count);
    VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesKHR(
        device->logic_device(), _pipeline, 0, group_count,
        handles.size(), handles.data()));

    // Create SBT buffer with VMA - needs host-visible + shader device address
    VkBufferCreateInfo sbt_buffer_ci{};
    sbt_buffer_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sbt_buffer_ci.size = sbt_size;
    sbt_buffer_ci.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo sbt_alloc_ci{};
    sbt_alloc_ci.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    sbt_alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo sbt_alloc_info{};
    VK_CHECK_RESULT(vmaCreateBuffer(
        device->allocator().allocator(),
        &sbt_buffer_ci, &sbt_alloc_ci,
        &_sbt_vk_buffer, &_sbt_allocation, &sbt_alloc_info));

    // Write SBT data
    auto sbt_mapped = static_cast<uint8_t *>(sbt_alloc_info.pMappedData);
    memset(sbt_mapped, 0, sbt_size);
    memcpy(sbt_mapped, handles.data(), handle_size);
    memcpy(sbt_mapped + raygen_size, handles.data() + handle_size, handle_size);
    memcpy(sbt_mapped + raygen_size + miss_size, handles.data() + handle_size * 2, handle_size);
    vmaFlushAllocation(device->allocator().allocator(), _sbt_allocation, 0, sbt_size);

    VkBufferDeviceAddressInfo sbt_addr_info{};
    sbt_addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    sbt_addr_info.buffer = _sbt_vk_buffer;
    auto sbt_address = vkGetBufferDeviceAddress(device->logic_device(), &sbt_addr_info);
    // For raygen, size must equal stride (only one entry)
    _raygen_region = {sbt_address, aligned_handle_size, aligned_handle_size};
    _miss_region = {sbt_address + raygen_size, aligned_handle_size, miss_size};
    _hit_region = {sbt_address + raygen_size + miss_size, aligned_handle_size, hit_size};
    _callable_region = {0, 0, 0};
}

bool RayTracingShader::serialize_pso(vstd::vector<std::byte> &result) const {
    auto last_size = result.size();
    size_t pso_size = 0;
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, nullptr));
    if (pso_size <= sizeof(VkPipelineCacheHeaderVersionOne)) {
        luisa::vector_resize(result, last_size);
        return false;
    }
    luisa::vector_resize(result, last_size + pso_size);
    VK_CHECK_RESULT(vkGetPipelineCacheData(device()->logic_device(), _pipe_cache, &pso_size, result.data() + last_size));
    luisa::vector_resize(result, last_size + pso_size);
    return true;
}

void RayTracingShader::release_vma_resources() noexcept {
    if (_sbt_vk_buffer) {
        vmaDestroyBuffer(device()->allocator().allocator(), _sbt_vk_buffer, _sbt_allocation);
        _sbt_vk_buffer = VK_NULL_HANDLE;
        _sbt_allocation = VK_NULL_HANDLE;
    }
}

void RayTracingShader::destroy_pipeline_objects() noexcept {
    if (_pipeline) {
        vkDestroyPipeline(device()->logic_device(), _pipeline, Device::alloc_callbacks());
        _pipeline = VK_NULL_HANDLE;
    }
    if (_pipe_cache) {
        vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, Device::alloc_callbacks());
        _pipe_cache = VK_NULL_HANDLE;
    }
}

RayTracingShader::~RayTracingShader() {
    // WORKAROUND: NVIDIA Vulkan driver bug with VK_NV_ray_tracing_motion_blur.
    // After vkDestroyPipeline is called on an RT pipeline that used motion blur,
    // ALL subsequent vkCreateComputePipelines and vkCreateRayTracingPipelinesKHR
    // calls deadlock indefinitely, even after vkDeviceWaitIdle returns VK_SUCCESS.
    //
    // Workaround: intentionally leak the VkPipeline and VkPipelineCache.
    // Only free the SBT buffer (VMA-managed, unrelated to the bug).
    // The pipeline objects will be cleaned up in ~Device() when no more
    // pipeline creation is possible.
    if (_sbt_vk_buffer) {
        vmaDestroyBuffer(device()->allocator().allocator(), _sbt_vk_buffer, _sbt_allocation);
    }
    // DO NOT call vkDestroyPipeline or vkDestroyPipelineCache here.
    // vkDestroyPipeline(device()->logic_device(), _pipeline, Device::alloc_callbacks());
    // vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, Device::alloc_callbacks());
}

RayTracingShader *RayTracingShader::compile(
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
    bool debug) {

    auto str = codegen();

    if (Device::print_code()) {
        auto dump_name = [&]() -> luisa::string {
            if (!file_name.empty()) return luisa::string{file_name.data(), file_name.size()};
            if (code_md5) return luisa::string{code_md5->to_string(false)};
            return luisa::string{"rt_unknown"};
        }();
        auto dump_file_name = luisa::format("hlsl_output_{}.hlsl", dump_name);
        auto f = fopen(dump_file_name.c_str(), "wb");
        if (f) {
            fwrite(str.result.data(), str.result.size(), 1, f);
            fclose(f);
        }
    }

    auto comp_result = Device::compiler()->compile_raytracing(
        str.result.view(),
        !debug,
        shader_model,
        unsafe_math,
        true,
        debug);

    return comp_result.multi_visit_or(
        vstd::UndefEval<RayTracingShader *>{},
        [&](hlsl::ComUniquePtr<IDxcBlob> const &buffer) {
            // Post-process SPIR-V to replace OpTraceRayKHR with OpTraceRayMotionNV
            auto original_spv = vstd::span<uint32_t const>{
                reinterpret_cast<const uint32_t *>(buffer->GetBufferPointer()),
                buffer->GetBufferSize() / sizeof(uint32_t)};

            vstd::span<uint32_t const> final_spv = original_spv;
            auto patched_spv = patch_spirv_for_motion_blur(original_spv);
            if (!patched_spv.empty()) {
                final_spv = {patched_spv.data(), patched_spv.size()};
            }

            auto shader = new RayTracingShader(
                device,
                block_size,
                str.properties,
                std::move(saved_args),
                final_spv,
                std::move(bindings),
                {},
                str.useTex2DBindless,
                str.useTex3DBindless,
                str.useBufferBindless,
                std::move(str.printers));
            return shader;
        },
        [](auto &&err) {
            LUISA_ERROR("RT Shader Compile Error: {}", err);
            return nullptr;
        });
}

}// namespace lc::vk
