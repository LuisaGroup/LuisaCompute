#pragma once

#include <luisa/runtime/rhi/pixel.h>
#include "metal_api.h"

extern "C" CA::MetalLayer *luisa_metal_backend_create_layer(
    MTL::Device *device, uint64_t window_handle,
    uint32_t width, uint32_t height,
    bool hdr, bool vsync,
    uint32_t back_buffer_count) noexcept;

namespace luisa::compute::metal {

class MetalDevice;
class MetalTexture;

class MetalSwapchain {

private:
    CA::MetalLayer *_layer;
    MTL::RenderPipelineState *_pipeline;
    MTL::RenderPassDescriptor *_render_pass_desc;
    NS::String *_command_label;

public:
    MetalSwapchain(MetalDevice *device, uint64_t window_handle,
                   uint width, uint height, bool allow_hdr,
                   bool vsync, uint back_buffer_size) noexcept;
    ~MetalSwapchain() noexcept;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept;
    [[nodiscard]] auto layer() const noexcept { return _layer; }
    void present(MTL::CommandQueue *queue, MTL::Texture *image) noexcept;
    void set_name(luisa::string_view name) noexcept;
};

}// namespace luisa::compute::metal

