//
// Created by Mike Smith on 2021/6/30.
//

#import <mutex>

#import <core/logging.h>
#import <backends/metal/metal_ring_buffer.h>

namespace luisa::compute::metal {

MetalBufferView MetalRingBuffer::allocate(size_t size) noexcept {
    std::scoped_lock lock{_mutex};
    auto offset = _alloc_end;
    _alloc_end += size;
    if (_alloc_end > _size) {
        offset = 0u;
        _alloc_end = size;
    }
    if (_alloc_end > _alloc_begin) {
        return {nullptr, 0u, 0u};
    }
    return {_buffer, offset, size};
}

void MetalRingBuffer::recycle(const MetalBufferView &view) noexcept {
    std::scoped_lock lock{_mutex};
    if (view.offset() == 0u) {
        if (_alloc_begin + view.size() <= _size) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid ring buffer item offset {} for recycling (expected {}).",
                0u, _alloc_begin);
        }
        _alloc_begin = 0u;
    } else if (view.offset() != _alloc_begin) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid ring buffer item offset {} for recycling (expected {}).",
            view.offset(), _alloc_begin);
    }
    _alloc_begin += view.size();
}

}
