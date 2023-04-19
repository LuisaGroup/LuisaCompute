//
// Created by Mike Smith on 2023/4/16.
//

#pragma once

#include <runtime/rhi/pixel.h>
#include <backends/metal/metal_api.h>

extern "C" CA::MetalLayer *luisa_metal_backend_create_layer(
    MTL::Device *device, uint64_t window_handle,
    bool hdr, bool vsync, uint32_t back_buffer_count) noexcept;

namespace luisa::compute::metal {

class MetalDevice;
class MetalTexture;

class MetalSwapchain {

private:
    MetalDevice *_device;
    CA::MetalLayer *_layer;
    MTL::RenderPassDescriptor *_render_pass_desc;

public:
    MetalSwapchain(MetalDevice *device, uint64_t window_handle,
                   uint width, uint height, bool allow_hdr,
                   bool vsync, uint back_buffer_size) noexcept;
    ~MetalSwapchain() noexcept;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept;
    void present(MTL::CommandQueue *queue, MTL::Texture *image) noexcept;
};

}// namespace luisa::compute::metal
