#include <luisa/core/logging.h>

#include "metal_depth_buffer.h"

namespace luisa::compute::metal {

namespace {

[[nodiscard]] MTL::PixelFormat depth_pixel_format(DepthFormat format) noexcept {
    switch (format) {
        case DepthFormat::D16: return MTL::PixelFormatDepth16Unorm;
        case DepthFormat::D32: return MTL::PixelFormatDepth32Float;
        case DepthFormat::D24S8:
        case DepthFormat::D32S8A24:
            LUISA_ERROR_WITH_LOCATION(
                "Stencil-bearing depth formats are not supported by Metal raster AIR yet.");
        case DepthFormat::None: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid Metal depth format 0x{:02x}.",
                              luisa::to_underlying(format));
}

}// namespace

MetalDepthBuffer::MetalDepthBuffer(
    MTL::Device *device, DepthFormat format, uint width, uint height) noexcept
    : _handle{nullptr}, _format{format}, _size{width, height} {
    LUISA_ASSERT(width != 0u && height != 0u,
                 "Metal depth-buffer dimensions must be non-zero.");
    auto descriptor = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
    descriptor->setTextureType(MTL::TextureType2D);
    descriptor->setPixelFormat(depth_pixel_format(format));
    descriptor->setWidth(width);
    descriptor->setHeight(height);
    descriptor->setDepth(1u);
    descriptor->setMipmapLevelCount(1u);
    descriptor->setStorageMode(MTL::StorageModePrivate);
    descriptor->setHazardTrackingMode(MTL::HazardTrackingModeTracked);
    descriptor->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    descriptor->setAllowGPUOptimizedContents(true);
    _handle = device->newTexture(descriptor.get());
    LUISA_ASSERT(_handle != nullptr,
                 "Failed to create {}x{} Metal depth buffer.", width, height);
}

MetalDepthBuffer::~MetalDepthBuffer() noexcept {
    _handle->release();
}

MTL::Texture *MetalDepthBuffer::handle(uint level) const noexcept {
    LUISA_ASSERT(level == 0u,
                 "Metal depth buffers only expose mip level zero, got {}.", level);
    return _handle;
}

void MetalDepthBuffer::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _handle->setLabel(nullptr);
    } else {
        auto label = NS::TransferPtr(NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false));
        _handle->setLabel(label.get());
    }
}

}// namespace luisa::compute::metal
