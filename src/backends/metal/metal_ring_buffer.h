//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#import <core/spin_mutex.h>
#import <core/mathematics.h>
#import <backends/metal/metal_buffer_view.h>

namespace luisa::compute::metal {

class MetalRingBuffer {

    static constexpr auto alignment = static_cast<size_t>(16u);

private:
    id<MTLBuffer> _buffer{nullptr};
    id<MTLDevice> _device;
    size_t _size;
    size_t _free_begin;
    size_t _free_end;
    uint _alloc_count;
    bool _optimize_write;

public:
    MetalRingBuffer(id<MTLDevice> device, size_t size, bool optimize_write) noexcept
        : _device{device},
          _size{std::max(next_pow2(size), alignment)},
          _free_begin{0u},
          _free_end{0u},
          _alloc_count{0u},
          _optimize_write{optimize_write} {}
    MetalRingBuffer(MetalRingBuffer &&) noexcept = delete;
    MetalRingBuffer(const MetalRingBuffer &) noexcept = delete;
    MetalRingBuffer &operator=(MetalRingBuffer &&) noexcept = delete;
    MetalRingBuffer &operator=(const MetalRingBuffer &) noexcept = delete;
    ~MetalRingBuffer() noexcept { _buffer = nullptr; }
    [[nodiscard]] MetalBufferView allocate(size_t size) noexcept;
    void recycle(const MetalBufferView &view) noexcept;
};

}// namespace luisa::compute::metal
