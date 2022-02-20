//
// Created by Mike Smith on 2021/6/30.
//

#import <mutex>

#import <core/logging.h>
#import <backends/metal/metal_host_buffer_pool.h>

namespace luisa::compute::metal {

MetalBufferView MetalHostBufferPool::allocate(size_t size) noexcept {
    auto allocate_buffer = [this](size_t size) noexcept {
        auto options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        if (_optimize_write) { options |= MTLResourceCPUCacheModeWriteCombined; }
        auto buffer = [_device newBufferWithLength:size options:options];
        return buffer;
    };
    std::scoped_lock lock{_mutex};
    size = (size + alignment - 1u) / alignment * alignment;
    auto node = _first_fit.allocate(size);
    if (node == nullptr) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate {} bytes "
            "from MetalHostBufferPool.",
            size);
        auto buffer = allocate_buffer(size);
        return {buffer, nullptr};
    }
    if (_buffer == nullptr) {// lazily create the device buffer
        _buffer = allocate_buffer(_first_fit.size());
    }
    return {_buffer, node};
}

void MetalHostBufferPool::recycle(const MetalBufferView &view) noexcept {
    if (view.is_pooled()) {
        std::scoped_lock lock{_mutex};
        _first_fit.free(view.node());
    }
}

}
