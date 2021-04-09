//
// Created by Mike Smith on 2021/4/8.
//

#pragma once

#import <Metal/Metal.h>

namespace luisa::compute::metal {

struct MetalEvent {

    id<MTLSharedEvent> handle;
    MTLSharedEventListener *listener{nullptr};
    uint64_t counter{0u};

    explicit MetalEvent(id<MTLSharedEvent> h) noexcept
        : handle{h} { handle.signaledValue = 0u; }
};

}// namespace luisa::compute::metal
