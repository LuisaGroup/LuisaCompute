//
// Created by Mike Smith on 2021/4/8.
//

#import <future>

#import <core/logging.h>
#import <backends/metal/metal_event.h>

namespace luisa::compute::metal {

MetalEvent::MetalEvent(id<MTLSharedEvent> handle) noexcept
    : _handle{handle}, _counter{0u} { _handle.signaledValue = 0u; }

void MetalEvent::wait(id<MTLCommandBuffer> cb) const noexcept {
    [cb encodeWaitForEvent:_handle value:_counter];
}

void MetalEvent::synchronize(MTLSharedEventListener *listener) const noexcept {
    std::condition_variable cv;
    auto p = &cv;
    [_handle notifyListener:listener
                    atValue:_counter
                      block:^(id<MTLSharedEvent>, uint64_t) { p->notify_one(); }];
    std::mutex mutex;
    std::unique_lock lock{mutex};
    cv.wait(lock);
}

}
