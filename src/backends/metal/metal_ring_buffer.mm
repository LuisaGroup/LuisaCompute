//
// Created by Mike Smith on 2021/6/30.
//

#import <mutex>

#import <core/logging.h>
#import <backends/metal/metal_ring_buffer.h>

namespace luisa::compute::metal {

MetalBufferView MetalRingBuffer::allocate(size_t size) noexcept {
    auto alignment = 16u;
    size = (size + alignment - 1u) / alignment * alignment;
    std::scoped_lock lock{_mutex};
    if (_buffer == nullptr) {
        auto buffer_options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        if (_optimize_write) { buffer_options |= MTLResourceCPUCacheModeWriteCombined; }
        _buffer = [_device newBufferWithLength:size options:buffer_options];
    }
    auto offset = _free_begin & (_size - 1u);
    auto free_next = _free_begin + size;
    if (offset + size > _size) {
        offset = 0u;                // wrap
        free_next += _size - offset;// skip tail
        if (free_next > _free_end) {// fails to allocate
            LUISA_WARNING_WITH_LOCATION(
                "Failed to allocate {} bytes from ring buffer "
                "(free_begin = {}, free_end = {}, size = {}).",
                size, _free_begin, _free_end, _size);
            return {nullptr, 0u, 0u};
        }
    }
    _free_end = free_next;
    return {_buffer, offset, size};
}

void MetalRingBuffer::recycle(const MetalBufferView &view) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto end_offset = _free_end & (_size - 1u); end_offset + view.size() > _size) {
        if (view.offset() != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid ring buffer item offset {} for recycling (expected 0).",
                view.offset());
        }
        _free_end += _size - end_offset;
    }
    _free_end += view.size();
}

}
