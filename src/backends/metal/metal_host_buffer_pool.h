//
// Created by Mike Smith on 2021/6/30.
//

#pragma once

#import <core/spin_mutex.h>
#import <core/mathematics.h>
#import <core/first_fit.h>
#import <backends/metal/metal_buffer_view.h>

namespace luisa::compute::metal {

class MetalHostBufferPool {

    static constexpr auto alignment = static_cast<size_t>(16u);

private:
    std::mutex _mutex;
    FirstFit _first_fit;
    id<MTLBuffer> _buffer{nullptr};
    id<MTLDevice> _device;
    bool _optimize_write;

public:
    MetalHostBufferPool(id<MTLDevice> device, size_t size, bool optimize_write) noexcept
        : _first_fit{size, alignment},
          _device{device},
          _optimize_write{optimize_write} {}
    MetalHostBufferPool(MetalHostBufferPool &&) noexcept = delete;
    MetalHostBufferPool(const MetalHostBufferPool &) noexcept = delete;
    MetalHostBufferPool &operator=(MetalHostBufferPool &&) noexcept = delete;
    MetalHostBufferPool &operator=(const MetalHostBufferPool &) noexcept = delete;
    ~MetalHostBufferPool() noexcept { _buffer = nullptr; }
    [[nodiscard]] MetalBufferView allocate(size_t size) noexcept;
    void recycle(const MetalBufferView &view) noexcept;
};

}// namespace luisa::compute::metal
