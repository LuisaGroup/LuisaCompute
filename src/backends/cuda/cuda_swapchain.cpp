//
// Created by Mike on 3/24/2023.
//

#ifdef LC_CUDA_ENABLE_VULKAN_SWAPCHAIN

#include <core/platform.h>
#include <vulkan/vulkan.h>

#ifdef LUISA_PLATFORM_WINDOWS
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#else
#error TODO
#endif

#include <backends/common/vulkan_swapchain.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_swapchain.h>

namespace luisa::compute::cuda {

class CUDASwapchain::Impl {

private:
    static constexpr std::array required_extensions{
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
    };

private:
    VulkanSwapchain _base;

public:
    Impl(CUuuid device_uuid, uint64_t window_handle,
         uint width, uint height, bool allow_hdr,
         bool vsync, uint back_buffer_size) noexcept
        : _base{luisa::bit_cast<VulkanSwapchain::DeviceUUID>(device_uuid),
                window_handle,
                width,
                height,
                allow_hdr,
                vsync,
                back_buffer_size,
                required_extensions} {}
    [[nodiscard]] auto pixel_storage() const noexcept { return PixelStorage::BYTE4; }
    [[nodiscard]] auto size() const noexcept { return make_uint2(); }
    void present(CUstream stream, CUarray image) noexcept {}
};

CUDASwapchain::CUDASwapchain(CUDADevice *device, uint64_t window_handle,
                             uint width, uint height, bool allow_hdr,
                             bool vsync, uint back_buffer_size) noexcept
    : _impl{luisa::make_unique<Impl>(device->handle().uuid(),
                                     window_handle, width, height,
                                     allow_hdr, vsync, back_buffer_size)} {}

CUDASwapchain::~CUDASwapchain() noexcept = default;

PixelStorage CUDASwapchain::pixel_storage() const noexcept {
    return _impl->pixel_storage();
}

void CUDASwapchain::present(CUDAStream *stream, CUDAMipmapArray *image) noexcept {
    LUISA_ASSERT(image->storage() == _impl->pixel_storage() &&
                     all(image->size() == make_uint3(_impl->size(), 0u)),
                 "Image size and pixel format must match the swapchain");
    _impl->present(stream->handle(), image->level(0u));
}

}// namespace luisa::compute::cuda

#endif
