//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

namespace luisa::compute::metal {

class MetalEvent {

private:
    id<MTLSharedEvent> _handle;
    uint64_t _counter{0u};
    

public:
    explicit MetalEvent(id<MTLSharedEvent> handle) noexcept
        : _handle{handle} {}
    
    void signal(id<MTLCommandBuffer> cb) noexcept {
        [cb encodeSignalEvent:_handle value:++_counter];
    }
    
    void wait(id<MTLCommandBuffer> cb) const noexcept {
        [cb encodeWaitForEvent:_handle value:_counter];
    }
    
    void synchronize() const noexcept {
    
    }
};

}// namespace luisa::compute::metal
