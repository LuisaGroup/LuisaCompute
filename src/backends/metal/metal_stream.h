//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#import <mutex>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/spin_mutex.h>

namespace luisa::compute::metal {

class MetalStream {

private:
    id<MTLCommandQueue> _handle;
    __weak id<MTLCommandBuffer> _last{nullptr};
    spin_mutex _mutex;

public:
    explicit MetalStream(id<MTLCommandQueue> handle) noexcept
        : _handle{handle} {}
    ~MetalStream() noexcept { _handle = nullptr; }
    
    template<typename Encode>
    void with_command_buffer(Encode &&encode) noexcept {
        auto command_buffer = [_handle commandBuffer];
        {
            std::scoped_lock lock{_mutex};
            [command_buffer enqueue];
            _last = command_buffer;
        }
        encode(command_buffer);
        [command_buffer commit];
    }
    
    void synchronize() noexcept {
        auto last = [this] {
            std::scoped_lock lock{_mutex};
            __strong id<MTLCommandBuffer> cb = _last;
            return cb;
        }();
        if (last != nullptr) {
            [last waitUntilCompleted];
        }
    }
};

}
