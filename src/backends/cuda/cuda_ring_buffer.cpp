//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_ring_buffer.h>

namespace luisa::compute::cuda {

CUDARingBuffer::CUDARingBuffer(size_t size, bool write_combined) noexcept
    : _size{std::max(next_pow2(size), static_cast<size_t>(4096u))},
      _free_begin{0u},
      _free_end{0u},
      _alloc_count{0u},
      _write_combined{write_combined} {}

CUDARingBuffer::~CUDARingBuffer() noexcept {
    LUISA_CHECK_CUDA(cuMemFreeHost(reinterpret_cast<void *>(_memory)));
}

CUDARingBuffer::View CUDARingBuffer::allocate(size_t size) noexcept {

    auto allocate_ad_hoc = [size, this] {
        auto buffer = luisa::detail::allocator_allocate(size, 16u);
        return View{static_cast<std::byte *>(buffer), size, false};
    };

    std::scoped_lock lock{_mutex};

    // simple check
    if (size > _size) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate {} bytes from "
            "the ring memory with size {}.",
            size, _size);
        return allocate_ad_hoc();
    }
    size = (size + alignment - 1u) / alignment * alignment;

    auto memory = [this] {
        if (_memory == nullptr) {// lazily create the device memory
            auto flags = _write_combined ? CU_MEMHOSTALLOC_WRITECOMBINED : 0;
            LUISA_CHECK_CUDA(cuMemHostAlloc(reinterpret_cast<void **>(&_memory), _size, flags));
        }
        return _memory;
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
            "memory with begin {} and end {}.",
            size, _free_begin, _free_end);
        return allocate_ad_hoc();
    }
    _alloc_count++;
    _free_begin = (offset + size) & (_size - 1u);
    return {memory + offset, size, true};
}

void CUDARingBuffer::recycle(View view) noexcept {
    if (view.is_pooled()) {
        std::scoped_lock lock{_mutex};
        if (_free_end + view.size() > _size) { _free_end = 0u; }
        if (view.address() != _memory + _free_end) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid ring buffer item offset {} "
                "for recycling (expected {}).",
                view.address() - _memory, _free_end);
        }
        _free_end = (view.address() - _memory + view.size()) & (_size - 1u);
        _alloc_count--;
    } else {
        luisa::detail::allocator_deallocate(view.address(), 16u);
    }
}

[[nodiscard]] auto &ring_buffer_recycle_context_pool() noexcept {
    static Pool<CUDARingBuffer::RecycleContext> pool;
    return pool;
}

inline CUDARingBuffer::RecycleContext::RecycleContext(CUDARingBuffer::View buffer, CUDARingBuffer *pool) noexcept
    : _buffer{buffer}, _pool{pool} {}

void CUDARingBuffer::RecycleContext::recycle() noexcept {
    _pool->recycle(_buffer);
    ring_buffer_recycle_context_pool().recycle(this);
}

CUDARingBuffer::RecycleContext *CUDARingBuffer::RecycleContext::create(
    CUDARingBuffer::View buffer, CUDARingBuffer *pool) noexcept {
    return ring_buffer_recycle_context_pool().create(buffer, pool);
}

}// namespace luisa::compute::cuda
