//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#import <core/spin_mutex.h>
#import <backends/metal/metal_buffer_view.h>

namespace luisa::compute::metal {

class MetalRingBuffer {

private:
    id<MTLBuffer> _buffer;
    size_t _size;
    size_t _alloc_begin;
    size_t _alloc_end;
    spin_mutex _mutex;

public:
    MetalRingBuffer(id<MTLDevice> device, size_t size, bool optimize_write) noexcept
        : _size{size}, _alloc_begin{0u}, _alloc_end{0u} {
        auto buffer_options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        if (optimize_write) { buffer_options |= MTLResourceCPUCacheModeWriteCombined; }
        _buffer = [device newBufferWithLength:size options:buffer_options];
    }
    ~MetalRingBuffer() noexcept { _buffer = nullptr; }
    [[nodiscard]] MetalBufferView allocate(size_t size) noexcept;
    void recycle(const MetalBufferView &view) noexcept;
};

}// namespace luisa::compute::metal
