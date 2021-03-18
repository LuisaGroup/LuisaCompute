//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#import <vector>
#import <thread>

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import <core/spin_mutex.h>
#import <runtime/device.h>

namespace luisa::compute::metal {

class MetalDevice : public Device {

private:
    id<MTLDevice> _handle{nullptr};
    
    // for buffers
    mutable spin_mutex _buffer_mutex;
    std::vector<id<MTLBuffer>> _buffer_slots;
    std::vector<size_t> _available_buffer_slots;
    
    // for streams
    mutable spin_mutex _stream_mutex;
    std::vector<id<MTLCommandQueue>> _stream_slots;
    std::vector<size_t> _available_stream_slots;

private:
    uint64_t _create_buffer(size_t size_bytes) noexcept override;
    void _dispose_buffer(uint64_t handle) noexcept override;
    uint64_t _create_stream() noexcept override;
    void _dispose_stream(uint64_t handle) noexcept override;
    void _synchronize_stream(uint64_t stream_handle) noexcept override;
    void _dispatch(uint64_t stream_handle, std::unique_ptr<CommandBuffer> ptr) noexcept override;

public:
    explicit MetalDevice(uint32_t index) noexcept;
    ~MetalDevice() noexcept override;
    [[nodiscard]] id<MTLDevice> handle() const noexcept;
    [[nodiscard]] id<MTLBuffer> buffer(uint64_t handle) const noexcept;
    [[nodiscard]] id<MTLCommandQueue> stream(uint64_t handle) const noexcept;
};

}

