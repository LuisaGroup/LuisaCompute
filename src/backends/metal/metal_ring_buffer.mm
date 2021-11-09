//
// Created by Mike Smith on 2021/6/30.
//

#import <mutex>

#import <core/logging.h>
#import <backends/metal/metal_ring_buffer.h>

namespace luisa::compute::metal {

MetalBufferView MetalRingBuffer::allocate(size_t size) noexcept {

    // simple check
    if (size > _size) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate {} bytes from "
            "the ring buffer with size {}.",
            size, _size);
        return {nullptr, 0u, 0u};
    }
    size = (size + alignment - 1u) / alignment * alignment;

    auto buffer = [this] {
      if (_buffer == nullptr) {// lazily create the device buffer
          auto buffer_options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
          if (_optimize_write) { buffer_options |= MTLResourceCPUCacheModeWriteCombined; }
          _buffer = [_device newBufferWithLength:_size options:buffer_options];
      }
      return _buffer;
    }();

    // try allocation
    auto offset = [this, size] {
        if (_free_begin == _free_end && _alloc_count != 0u) { return _size; }
        if (_free_end <= _free_begin) {
            if (_free_begin + size <= _size) { return _free_begin; }
            return size <= _free_end ? 0u : _size;
        }
        return _free_begin + size <= _free_end ? _free_begin : _size;
    }();

    if (offset == _size) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate {} bytes from ring "
            "buffer with begin {} and end {}.",
            size, _free_begin, _free_end);
        return {nullptr, 0u, 0u};
    }
    _alloc_count++;
    _free_begin = (offset + size) & (_size - 1u);
    return {buffer, offset, size};
}

void MetalRingBuffer::recycle(const MetalBufferView &view) noexcept {
    if (_free_end + view.size() > _size) { _free_end = 0u; }
    if (view.offset() != _free_end) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid ring buffer item offset {} "
            "for recycling (expected {}).",
            view.offset(), _free_end);
    }
    _free_end = (view.offset() + view.size()) & (_size - 1u);
    _alloc_count--;
}

}
