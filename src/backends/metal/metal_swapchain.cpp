//
// Created by Mike Smith on 2023/4/16.
//

#include <core/logging.h>
#include <backends/metal/metal_device.h>
#include <backends/metal/metal_swapchain.h>

namespace luisa::compute::metal {

MetalSwapchain::MetalSwapchain(MetalDevice *device, uint64_t window_handle,
                               uint width, uint height, bool allow_hdr,
                               bool vsync, uint back_buffer_size) noexcept
    : _device{device},
      _layer{luisa_metal_backend_create_layer(
          device->handle(), window_handle,
          width, height, allow_hdr,
          vsync, back_buffer_size)},
      _render_pass_desc{MTL::RenderPassDescriptor::alloc()->init()} {
    auto attachment_desc = _render_pass_desc->colorAttachments()->object(0);
    attachment_desc->setLoadAction(MTL::LoadActionDontCare);
    attachment_desc->setStoreAction(MTL::StoreActionStore);
}

MetalSwapchain::~MetalSwapchain() noexcept {
    _render_pass_desc->release();
}

PixelStorage MetalSwapchain::pixel_storage() const noexcept {
    return _layer->pixelFormat() == MTL::PixelFormatRGBA16Float ?
               PixelStorage::HALF4 :
               PixelStorage::BYTE4;
}

void MetalSwapchain::present(MTL::CommandQueue *queue, MTL::Texture *image) noexcept {
    if (auto drawable = _layer->nextDrawable()) {
        auto attachment_desc = _render_pass_desc->colorAttachments()->object(0);
        attachment_desc->setTexture(drawable->texture());
        auto command_buffer = queue->commandBufferWithUnretainedReferences();
        auto command_encoder = command_buffer->renderCommandEncoder(_render_pass_desc);
        constexpr std::array vertices{make_float4(-1.f, 1.f, 0.f, 1.f), make_float4(-1.f, -1.f, 0.f, 1.f),
                                      make_float4(1.f, 1.f, 0.f, 1.f), make_float4(1.f, -1.f, 0.f, 1.f)};
        auto pso = _layer->pixelFormat() == MTL::PixelFormatRGBA16Float ?
                       _device->builtin_swapchain_present_hdr() :
                       _device->builtin_swapchain_present_ldr();
        command_encoder->setRenderPipelineState(pso);
        command_encoder->setVertexBytes(&vertices, sizeof(vertices), 0);
        command_encoder->setFragmentTexture(image, 0);
        command_encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                                        static_cast<NS::Integer>(0u),
                                        static_cast<NS::Integer>(4u));
        command_encoder->endEncoding();
        command_buffer->presentDrawable(drawable);
        command_buffer->commit();
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to acquire next drawable from swapchain.");
    }
}

}// namespace luisa::compute::metal
