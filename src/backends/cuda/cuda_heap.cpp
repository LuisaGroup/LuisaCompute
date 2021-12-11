//
// Created by Mike on 2021/12/10.
//

#include <mutex>

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_heap.h>

namespace luisa::compute::cuda {

namespace detail {

[[nodiscard]] inline constexpr auto is_small_buffer_handle(uint64_t handle) noexcept {
    return static_cast<bool>(handle & 1u);
}

[[nodiscard]] static auto &buffer_free_context_pool() noexcept {
    static Pool<CUDAHeap::BufferFreeContext> pool;
    return pool;
}

}

uint64_t CUDAHeap::allocate(size_t size) noexcept {
    static thread_local luisa::vector<CUdeviceptr> buffers_to_free;
    {
        std::scoped_lock lock{_mutex};
        buffers_to_free = _native_buffers_to_free;
        _native_buffers_to_free.clear();
    }
    for (auto b : buffers_to_free) {
        LUISA_CHECK_CUDA(cuMemFree(b));
    }
    if (size > small_buffer_size_threshold) {
        // allocate native buffers
        auto buffer = 0ull;
        LUISA_CHECK_CUDA(cuMemAlloc(&buffer, size));
        return buffer;
    }
    static Pool<SmallBuffer> pool;
    auto buffer = pool.create();
    buffer->pool = &pool;
    std::scoped_lock lock{_mutex};
    if (!_lists.empty()) {
        for (auto last = _last_tried_list;
             _last_tried_list != last + _lists.size();
             _last_tried_list++) {
            auto index = _last_tried_list < _lists.size() ?
                             _last_tried_list :
                             _last_tried_list - _lists.size();
            auto list = _lists[index].get();
            auto dump = list->dump_free_list();
            if (auto node = _lists[index]->allocate(size)) {
                buffer->node = node;
                buffer->list = _lists[index].get();
                buffer->address = _pool_buffers[index] + node->offset();
                return reinterpret_cast<uint64_t>(buffer) | 1u;
            }
        }
    }
    // create new list
    auto pool_buffer = 0ull;
    LUISA_CHECK_CUDA(cuMemAlloc(&pool_buffer, small_buffer_pool_size));
    _pool_buffers.emplace_back(pool_buffer);
    _last_tried_list = _lists.size();
    auto list = _lists.emplace_back(luisa::make_unique<FirstFit>(small_buffer_pool_size, small_buffer_alignment)).get();
    auto node = list->allocate(size);
    buffer->node = node;
    buffer->list = list;
    buffer->address = pool_buffer + node->offset();
    return reinterpret_cast<uint64_t>(buffer) | 1u;
}

void CUDAHeap::free(uint64_t handle) noexcept {
    if (handle != 0u) {
        if (detail::is_small_buffer_handle(handle)) {
            auto buffer = reinterpret_cast<SmallBuffer *>(handle & ~1ull);
            auto list = buffer->list;
            auto node = buffer->node;
            auto pool = buffer->pool;
            pool->recycle(buffer);
            std::scoped_lock lock{_mutex};
            list->free(node);
        } else {
            std::scoped_lock lock{_mutex};
            _native_buffers_to_free.emplace_back(handle);
        }
    }
}

CUdeviceptr CUDAHeap::buffer_address(uint64_t handle) noexcept {
    return detail::is_small_buffer_handle(handle) ?
               reinterpret_cast<SmallBuffer *>(handle & ~1ull)->address :
               handle;
}

CUDAHeap::~CUDAHeap() noexcept {
    for (auto b : _native_buffers_to_free) {
        LUISA_CHECK_CUDA(cuMemFree(b));
    }
    for (auto b : _pool_buffers) {
        LUISA_CHECK_CUDA(cuMemFree(b));
    }
}

CUDAHeap::BufferFreeContext *CUDAHeap::BufferFreeContext::create(CUDAHeap *heap, uint64_t buffer) noexcept {
    return detail::buffer_free_context_pool().create(BufferFreeContext{heap, buffer});
}

void CUDAHeap::BufferFreeContext::recycle() noexcept {
    _heap->free(_buffer);
    detail::buffer_free_context_pool().recycle(this);
}

}
