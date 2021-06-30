//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#import <core/spin_mutex.h>
#import <backends/metal/metal_buffer_view.h>

namespace luisa::compute::metal {

class MetalRingBuffer {

private:
    id<MTLBuffer> _buffer{nullptr};
    id<MTLDevice> _device;
    size_t _size;
    size_t _free_begin;
    size_t _free_end;
    bool _optimize_write;
    spin_mutex _mutex;

public:
    MetalRingBuffer(id<MTLDevice> device, size_t size, bool optimize_write) noexcept
        : _device{device}, _size{size}, _free_begin{0u}, _free_end{size}, _optimize_write{optimize_write} {}
    ~MetalRingBuffer() noexcept { _buffer = nullptr; }
    [[nodiscard]] MetalBufferView allocate(size_t size) noexcept;
    void recycle(const MetalBufferView &view) noexcept;
};

}// namespace luisa::compute::metal
