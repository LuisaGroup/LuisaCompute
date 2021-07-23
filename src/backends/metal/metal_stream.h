//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#import <mutex>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/spin_mutex.h>
#import <backends/metal/metal_ring_buffer.h>

namespace luisa::compute::metal {

class MetalStream {

public:
    static constexpr auto ring_buffer_size = 32u * 1024u * 1024u;

private:
    id<MTLCommandQueue> _handle;
    id<MTLCommandBuffer> _last{nullptr};
    MetalRingBuffer _upload_ring_buffer;
    MetalRingBuffer _download_ring_buffer;
    spin_mutex _mutex;

public:
    explicit MetalStream(id<MTLCommandQueue> handle) noexcept
        : _handle{handle},
          _upload_ring_buffer{handle.device, ring_buffer_size, true},
          _download_ring_buffer{handle.device, ring_buffer_size, false} {}

    ~MetalStream() noexcept { _handle = nullptr; }

    [[nodiscard]] auto command_buffer() noexcept { return [_handle commandBuffer]; }
    [[nodiscard]] auto &upload_ring_buffer() noexcept { return _upload_ring_buffer; }
    [[nodiscard]] auto &download_ring_buffer() noexcept { return _download_ring_buffer; }

    template<typename Dispatch>
    void dispatch(Dispatch d) noexcept {
        @autoreleasepool {
            auto command_buffer = d(this);
            {
                std::scoped_lock lock{_mutex};
                [command_buffer commit];
                _last = command_buffer;
            }
        }
    }

    void synchronize() noexcept {
        if (auto last = [this]() noexcept
            -> id<MTLCommandBuffer> {
                std::scoped_lock lock{_mutex};
                auto last_cmd = _last;
                _last = nullptr;
                return last_cmd;
            }()) { [last waitUntilCompleted]; }
    }
};

}// namespace luisa::compute::metal
