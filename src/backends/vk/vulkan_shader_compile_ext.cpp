#include "vulkan_shader_compile_ext.h"
#include "device.h"
namespace lc::vk {

luisa::compute::VulkanShaderCompileCapabilities
VkShaderCompileExtImpl::capabilities() const noexcept {
    luisa::compute::VulkanShaderCompileCapabilities capabilities;
    capabilities.parallel_prepare = true;
    capabilities.ordered_publication = true;
    // The compatibility HLSL-to-SPIR-V route keeps the serial path: it is compiled
    // by the process-global DXC instance (`Device::compiler()`), and this backend
    // does not own that object's thread contract. A request that needs it is
    // reported with `serial_fallback` instead of being run concurrently.
    capabilities.parallel_hlsl_compatibility_route = false;
    capabilities.max_workers = detail::ShaderCompilePool::max_workers;
    return capabilities;
}

luisa::compute::VulkanShaderCompileStatus
VkShaderCompileExtImpl::compile_shaders(
    luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
    luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
    uint32_t worker_count) noexcept {
    return _device->shader_compile_pool().compile(requests, results, worker_count);
}

void VkShaderCompileExtImpl::shutdown_pool() noexcept {
    _device->shutdown_shader_compile_pool();
}

luisa::compute::VulkanShaderCompileStats
VkShaderCompileExtImpl::stats() const noexcept {
    return _device->shader_compile_pool().stats();
}
}// namespace lc::vk