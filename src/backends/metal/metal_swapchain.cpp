#include <luisa/core/logging.h>
#include "metal_device.h"
#include "metal_swapchain.h"

namespace luisa::compute::metal {

MetalSwapchain::MetalSwapchain(MetalDevice *device, uint64_t window_handle,
                               uint width, uint height, bool allow_hdr,
                               bool vsync, uint back_buffer_size) noexcept
    : _layer{luisa_metal_backend_create_layer(
          device->handle(), window_handle,
          width, height, allow_hdr,
          vsync, back_buffer_size)},
      _pipeline{allow_hdr ?
                    device->builtin_swapchain_present_hdr() :
                    device->builtin_swapchain_present_ldr()},
      _render_pass_desc{MTL::RenderPassDescriptor::alloc()->init()} {
    _layer->retain();
    auto attachment_desc = _render_pass_desc->colorAttachments()->object(0);
    attachment_desc->setLoadAction(MTL::LoadActionDontCare);
    attachment_desc->setStoreAction(MTL::StoreActionStore);
}

MetalSwapchain::~MetalSwapchain() noexcept {
    if (_command_label) { _command_label->release(); }
    _render_pass_desc->release();
    _layer->release();
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
        constexpr std::array vertices{make_float2(-1.f, 1.f), make_float2(-1.f, -1.f),
                                      make_float2(1.f, 1.f), make_float2(1.f, -1.f)};
        command_encoder->setRenderPipelineState(_pipeline);
        command_encoder->setVertexBytes(&vertices, sizeof(vertices), 0);
        command_encoder->setFragmentTexture(image, 0);
        command_encoder->drawPrimitives(MTL::PrimitiveTypeTriangleStrip,
                                        static_cast<NS::Integer>(0u),
                                        static_cast<NS::Integer>(4u));
        command_encoder->endEncoding();
        command_buffer->presentDrawable(drawable);
        if (_command_label) { command_buffer->setLabel(_command_label); }
        command_buffer->commit();
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to acquire next drawable from swapchain.");
    }
}

void MetalSwapchain::set_name(luisa::string_view name) noexcept {
    if (_command_label) {
        _command_label->release();
        _command_label = nullptr;
    }
    if (!name.empty()) {
        auto label = luisa::format("{}::present", name);
        _command_label = NS::String::alloc()->init(
            label.data(), label.size(),
            NS::UTF8StringEncoding, false);
    }
}

}// namespace luisa::compute::metal

