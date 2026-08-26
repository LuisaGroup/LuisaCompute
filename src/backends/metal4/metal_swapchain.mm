#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

extern "C" CAMetalLayer *luisa_metal_backend_create_layer(id<MTLDevice> device, uint64_t window_handle,
                                                          uint32_t width, uint32_t height,
                                                          bool hdr, bool vsync,
                                                          uint32_t back_buffer_count) noexcept {

    NSView *view = nullptr;
    auto window_or_view = (__bridge NSObject *)(reinterpret_cast<void *>(window_handle));
    if ([window_or_view isKindOfClass:[NSWindow class]]) {
        view = static_cast<NSWindow *>(window_or_view).contentView;
    } else {
        if (![window_or_view isKindOfClass:[NSView class]]) {
            NSLog(@"Invalid window handle %llu of class %@. "
                   "Expected NSWindow or NSView.",
                  window_handle, [window_or_view class]);
        }
        view = static_cast<NSView *>(window_or_view);
    }
    auto layer = [CAMetalLayer layer];
    view.layer = layer;
    view.wantsLayer = YES;
    view.layer.contentsScale = view.window.backingScaleFactor;
    layer.device = device;
    layer.pixelFormat = hdr ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
    layer.wantsExtendedDynamicRangeContent = hdr;
    layer.displaySyncEnabled = vsync;
    layer.maximumDrawableCount = back_buffer_count > 3u ? 3u :
                                 back_buffer_count < 2u ? 2u :
                                                          back_buffer_count;
    layer.drawableSize = CGSizeMake(width, height);
    return layer;
}
