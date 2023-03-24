//
// Created by Mike on 3/24/2023.
//

#ifdef LC_CUDA_ENABLE_VULKAN_SWAPCHAIN

#include <backends/common/vulkan_swapchain.h>
#include <backends/cuda/cuda_swapchain.h>

namespace luisa::compute::cuda {

class CUDASwapchain::Impl {

};

CUDASwapchain::CUDASwapchain(CUDADevice *device, uint64_t window_handle,
                             uint width, uint height, bool allow_hdr,
                             bool vsync, uint back_buffer_size) noexcept {}

CUDASwapchain::~CUDASwapchain() noexcept = default;

PixelStorage CUDASwapchain::pixel_storage() const noexcept {
    return PixelStorage::BYTE4;
}

void CUDASwapchain::present(CUDAStream *stream, CUDAMipmapArray *image) noexcept {
}

}// namespace luisa::compute::cuda

#endif
