//
// Created by Mike Smith on 2023/4/15.
//

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

namespace luisa::compute {

void *cocoa_window_content_view(uint64_t window_handle) noexcept {
    @autoreleasepool {
        auto window = (__bridge NSWindow *)(reinterpret_cast<void *>(window_handle));
        auto layer = [CAMetalLayer layer];
        window.contentView.layer = layer;
        window.contentView.wantsLayer = YES;
        window.contentView.layer.contentsScale = window.backingScaleFactor;
        return window.contentView;
    }
}

}// namespace luisa::compute
