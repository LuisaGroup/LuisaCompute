//
// Created by Mike Smith on 2023/4/16.
//

#pragma once

#include <runtime/rhi/pixel.h>
#include <backends/metal/metal_api.h>

namespace luisa::compute::metal {

class MetalDevice;
class MetalTexture;

class MetalSwapchain {

public:
    MetalSwapchain(MetalDevice *device, uint64_t window_handle,
                   uint width, uint height, bool allow_hdr,
                   bool vsync, uint back_buffer_size) noexcept;
    ~MetalSwapchain() noexcept;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept;
    void present(MTL::CommandQueue *queue, MTL::Texture *image) noexcept;
};

}// namespace luisa::compute::metal
