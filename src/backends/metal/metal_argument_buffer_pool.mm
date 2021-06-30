//
// Created by Mike Smith on 2021/3/26.
//

#import <mutex>
#import <backends/metal/metal_argument_buffer_pool.h>

namespace luisa::compute::metal {

MetalBufferView MetalArgumentBufferPool::allocate() noexcept {
    _create_new_trunk_if_empty();
    std::scoped_lock lock{_mutex};
    auto buffer = _available_buffers.back();
    _available_buffers.pop_back();
    return buffer;
}

void MetalArgumentBufferPool::recycle(MetalBufferView buffer) noexcept {
    std::scoped_lock lock{_mutex};
    _available_buffers.emplace_back(std::move(buffer));
}

void MetalArgumentBufferPool::_create_new_trunk_if_empty() noexcept {
    std::scoped_lock lock{_mutex};
    if (_available_buffers.empty()) {
        static constexpr auto buffer_size = argument_buffer_size * trunk_size;
        static constexpr auto options = MTLResourceStorageModeShared
                                        | MTLResourceCPUCacheModeWriteCombined
                                        | MTLResourceHazardTrackingModeUntracked;
        auto buffer = [_device newBufferWithLength:buffer_size
                                           options:options];
        for (auto i = buffer_size; i != 0u; i -= argument_buffer_size) {
            _available_buffers.emplace_back(
                buffer, i - argument_buffer_size, argument_buffer_size);
        }
    }
}

MetalArgumentBufferPool::MetalArgumentBufferPool(id<MTLDevice> device) noexcept
    : _device{device} { _create_new_trunk_if_empty(); }

}
