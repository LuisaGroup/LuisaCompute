//
// Created by Mike Smith on 2021/3/26.
//

#pragma once

#import <vector>

#import <core/spin_mutex.h>
#import <core/allocator.h>
#import <backends/metal/metal_buffer_view.h>

namespace luisa::compute::metal {

class MetalSharedBufferPool {

private:
    __weak id<MTLDevice> _device;
    luisa::vector<MetalBufferView> _available_buffers;
    size_t _block_size;
    size_t _trunk_size;
    bool _optimize_for_write;
    spin_mutex _mutex;

private:
    void _create_new_trunk_if_empty() noexcept;

public:
    MetalSharedBufferPool(
        id<MTLDevice> device,
        size_t block_size, size_t trunk_size,
        bool optimize_for_write) noexcept;
    [[nodiscard]] MetalBufferView allocate() noexcept;
    void recycle(MetalBufferView buffer) noexcept;
};

}// namespace luisa::compute::metal
