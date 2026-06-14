#pragma once
#include "shader.h"
#include "default_buffer.h"
#include <volk.h>
#include "vk_allocator.h"
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

// Ray tracing pipeline shader for motion blur support.
// Uses VK_KHR_ray_tracing_pipeline with VK_NV_ray_tracing_motion_blur
// to support time-parameterized ray tracing (OpTraceMotionNV).
class RayTracingShader : public Shader {
    VkPipelineCache _pipe_cache{};
    VkPipeline _pipeline{};
    uint3 _block_size;
    // Shader Binding Table
    VkBuffer _sbt_vk_buffer{};
    VmaAllocation _sbt_allocation{};
    VkStridedDeviceAddressRegionKHR _raygen_region{};
    VkStridedDeviceAddressRegionKHR _miss_region{};
    VkStridedDeviceAddressRegionKHR _hit_region{};
    VkStridedDeviceAddressRegionKHR _callable_region{};

public:
    auto pipeline() const { return _pipeline; }
    bool serialize_pso(vstd::vector<std::byte> &result) const override;
    auto block_size() const { return _block_size; }
    auto const &raygen_region() const { return _raygen_region; }
    auto const &miss_region() const { return _miss_region; }
    auto const &hit_region() const { return _hit_region; }
    auto const &callable_region() const { return _callable_region; }

    RayTracingShader(
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
        vstd::vector<std::pair<luisa::string, luisa::compute::Type const *>> &&printers);
    ~RayTracingShader();

    static RayTracingShader *compile(
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
        bool debug = false);
};
}// namespace lc::vk
