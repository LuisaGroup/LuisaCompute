#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

namespace luisa::compute {

void *cocoa_window_content_view(uint64_t window_handle) noexcept {
    @autoreleasepool {
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
        return view;
    }
}

}// namespace luisa::compute
