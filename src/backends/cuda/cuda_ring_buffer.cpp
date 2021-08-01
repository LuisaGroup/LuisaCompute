//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_ring_buffer.h>

namespace luisa::compute::cuda {

CUDARingBuffer::CUDARingBuffer(size_t size, bool write_combined) noexcept
    : _size{std::max(next_pow2(size), alignment)},
      _free_begin{0u},
      _free_end{0u},
      _alloc_count{0u},
      _write_combined{write_combined} {}

CUDARingBuffer::~CUDARingBuffer() noexcept {
    LUISA_CHECK_CUDA(cuMemFreeHost(_memory));
}

std::span<std::byte> CUDARingBuffer::allocate(size_t size) noexcept {

    // simple check
    if (size > _size) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate {} bytes from "
            "the ring memory with size {}.",
            size, _size);
        return {};
    }
    size = (size + alignment - 1u) / alignment * alignment;

    auto memory = [this] {
        std::scoped_lock lock{_mutex};
        if (_memory == nullptr) {// lazily create the device memory
            auto flags = _write_combined ? CU_MEMHOSTALLOC_WRITECOMBINED : 0;
            LUISA_CHECK_CUDA(cuMemHostAlloc(reinterpret_cast<void **>(&_memory), _size, flags));
        }
        return _memory;
    }();

    // try allocation
    std::scoped_lock lock{_mutex};
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
            "memory with begin {} and end {}.",
            size, _free_begin, _free_end);
        return {};
    }
    _alloc_count++;
    _free_begin = (offset + size) & (_size - 1u);
    return {memory + offset, size};
}

void CUDARingBuffer::recycle(std::span<std::byte> view) noexcept {
    std::scoped_lock lock{_mutex};
    if (_free_end + view.size() > _size) { _free_end = 0u; }
    if (view.data() != _memory + _free_end) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid ring buffer item offset {} "
            "for recycling (expected {}).",
            view.data() - _memory, _free_end);
    }
    _free_end = (view.data() - _memory + view.size()) & (_size - 1u);
    _alloc_count--;
}

}// namespace luisa::compute::cuda
