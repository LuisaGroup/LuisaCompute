//
// Created by Mike Smith on 2021/4/13.
//

#pragma once

#import <mutex>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/logging.h>
#import <core/spin_mutex.h>
#import <backends/metal/metal_host_buffer_pool.h>

namespace luisa::compute::metal {

class MetalStream {

public:
    static constexpr auto ring_buffer_size = 32u * 1024u * 1024u;

private:
    id<MTLCommandQueue> _handle;
    __weak id<MTLCommandBuffer> _last{nullptr};
    MetalHostBufferPool _upload_host_buffer_pool;
    MetalHostBufferPool _download_host_buffer_pool;
    dispatch_semaphore_t _sem;

public:
    explicit MetalStream(id<MTLDevice> device, uint max_command_buffers) noexcept;
    ~MetalStream() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] id<MTLCommandBuffer> command_buffer() noexcept;
    void dispatch(id<MTLCommandBuffer> command_buffer) noexcept;
    void synchronize() noexcept;
    [[nodiscard]] auto &upload_host_buffer_pool() noexcept { return _upload_host_buffer_pool; }
    [[nodiscard]] auto &download_host_buffer_pool() noexcept { return _download_host_buffer_pool; }
};

}// namespace luisa::compute::metal
