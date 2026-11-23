// Backend-private implementation of `VulkanShaderCompileExt` (plan P5.1).
//
// It is a thin view over `Device` and the device-owned
// `detail::ShaderCompilePool`: the pool prepares on its workers, the extension
// publishes in request order through the same two Device entry points a serial
// `create_shader` uses. Nothing is cached here, so a caller that never asks for the
// extension still gets the pool's teardown at device destruction (`~Device` stops
// it) and never pays for its threads (they are created on first use only).
#pragma once
#include <luisa/backends/ext/vk_shader_compile_ext.h>
namespace lc::vk {
class Device;
class VkShaderCompileExtImpl final : public luisa::compute::VulkanShaderCompileExt {
private:
    Device *_device;
public:
    explicit VkShaderCompileExtImpl(Device *device) noexcept
        : _device{device} {}
    [[nodiscard]] luisa::compute::VulkanShaderCompileCapabilities
    capabilities() const noexcept override;
    [[nodiscard]] luisa::compute::VulkanShaderCompileStatus compile_shaders(
        luisa::span<const luisa::compute::VulkanShaderCompileRequest> requests,
        luisa::vector<luisa::compute::VulkanShaderCompileResult> &results,
        uint32_t worker_count) noexcept override;
    void shutdown_pool() noexcept override;
    [[nodiscard]] luisa::compute::VulkanShaderCompileStats stats() const noexcept override;
};
}// namespace lc::vk