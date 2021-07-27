//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#import <mutex>
#import <semaphore>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/logging.h>
#import <core/spin_mutex.h>
#import <backends/metal/metal_ring_buffer.h>

namespace luisa::compute::metal {

class MetalStream {

public:
    static constexpr auto ring_buffer_size = 32u * 1024u * 1024u;

private:
    id<MTLCommandQueue> _handle;
    __weak id<MTLCommandBuffer> _last{nullptr};
    MetalRingBuffer _upload_ring_buffer;
    MetalRingBuffer _download_ring_buffer;
    std::counting_semaphore<> _sem;
    spin_mutex _mutex;

public:
    explicit MetalStream(id<MTLDevice> device, uint max_command_buffers) noexcept;
    ~MetalStream() noexcept;
    [[nodiscard]] id<MTLCommandBuffer> command_buffer() noexcept;
    void dispatch(id<MTLCommandBuffer> command_buffer) noexcept;
    void synchronize() noexcept;
    [[nodiscard]] auto &upload_ring_buffer() noexcept { return _upload_ring_buffer; }
    [[nodiscard]] auto &download_ring_buffer() noexcept { return _download_ring_buffer; }
};

}// namespace luisa::compute::metal
