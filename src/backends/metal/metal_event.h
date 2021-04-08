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
    uint64_t _counter;

private:
    friend class MetalDevice;
    explicit MetalEvent(id<MTLSharedEvent> handle) noexcept;

public:
    void wait(id<MTLCommandBuffer> cb) const noexcept;
    void synchronize(MTLSharedEventListener *listener) const noexcept;
};

}// namespace luisa::compute::metal
