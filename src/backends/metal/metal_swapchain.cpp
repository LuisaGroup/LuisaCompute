//
// Created by Mike Smith on 2023/4/16.
//

#include <backends/metal/metal_swapchain.h>

namespace luisa::compute::metal {

MetalSwapchain::MetalSwapchain(MetalDevice *device, uint64_t window_handle,
                               uint width, uint height, bool allow_hdr,
                               bool vsync, uint back_buffer_size) noexcept {

}

MetalSwapchain::~MetalSwapchain() noexcept {

}

PixelStorage MetalSwapchain::pixel_storage() const noexcept {

}

void MetalSwapchain::present(MTL::CommandQueue *queue, MTL::Texture *image) noexcept {

}

}// namespace luisa::compute::metal
