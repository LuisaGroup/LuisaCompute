//
// Created by Mike Smith on 2021/3/26.
//

#import <mutex>

#import <core/mathematics.h>
#import <backends/metal/metal_shared_buffer_pool.h>

namespace luisa::compute::metal {

MetalBufferView MetalSharedBufferPool::allocate() noexcept {
    _create_new_trunk_if_empty();
    std::scoped_lock lock{_mutex};
    auto buffer = _available_buffers.back();
    _available_buffers.pop_back();
    return buffer;
}

void MetalSharedBufferPool::recycle(MetalBufferView buffer) noexcept {
    std::scoped_lock lock{_mutex};
    _available_buffers.emplace_back(std::move(buffer));
}

void MetalSharedBufferPool::_create_new_trunk_if_empty() noexcept {
    std::scoped_lock lock{_mutex};
    if (_available_buffers.empty()) {
        static constexpr auto options = MTLResourceStorageModeShared
                                        | MTLResourceCPUCacheModeWriteCombined
                                        | MTLResourceHazardTrackingModeUntracked;
        auto buffer_size = _block_size * _trunk_size;
        auto buffer = [_device newBufferWithLength:buffer_size
                                           options:options];
        for (auto i = buffer_size; i != 0u; i -= _block_size) {
            _available_buffers.emplace_back(
                buffer, i - _block_size, _block_size);
        }
    }
}

MetalSharedBufferPool::MetalSharedBufferPool(
    id<MTLDevice> device,
    size_t block_size,
    size_t trunk_size) noexcept
    : _device{device},
      _block_size{std::max(next_pow2(block_size), static_cast<size_t>(16u))},
      _trunk_size{std::max(next_pow2(trunk_size), 4096u / _block_size)} {
    _create_new_trunk_if_empty();
}

}
