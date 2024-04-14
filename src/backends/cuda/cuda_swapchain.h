#pragma once

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/pixel.h>

namespace luisa::compute {
class VulkanSwapchain;
}// namespace luisa::compute

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDATexture;

class CUDASwapchain {

public:
    class Impl;

private:
    luisa::unique_ptr<Impl> _impl;

public:
    CUDASwapchain(CUDADevice *device, SwapchainOption o) noexcept;
    ~CUDASwapchain() noexcept;
    [[nodiscard]] VulkanSwapchain *native_handle() noexcept;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept;
    void present(CUDAStream *stream, CUDATexture *image) noexcept;
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

#endif

