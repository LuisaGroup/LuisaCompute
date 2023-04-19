//
// Created by Mike Smith on 2023/4/19.
//

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

extern "C" CAMetalLayer *luisa_metal_backend_create_layer(id<MTLDevice> device, uint64_t window_handle,
                                                          bool hdr, bool vsync, uint32_t back_buffer_count) noexcept {
    auto window = (__bridge NSWindow *)(reinterpret_cast<void *>(window_handle));
    auto layer = [CAMetalLayer layer];
    window.contentView.layer = layer;
    window.contentView.wantsLayer = YES;
    window.contentView.layer.contentsScale = window.backingScaleFactor;
    layer.device = device;
    layer.pixelFormat = hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA8Unorm;
    layer.wantsExtendedDynamicRangeContent = hdr;
    layer.displaySyncEnabled = vsync;
    layer.maximumDrawableCount = back_buffer_count > 3u ? 3u :
                                 back_buffer_count < 2u ? 2u :
                                                          back_buffer_count;
    return layer;
}
