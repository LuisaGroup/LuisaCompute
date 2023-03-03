//
// Created by Mike on 8/1/2021.
//

#include "backends/cuda/cuda_error.h"
#include <backends/cuda/cuda_host_buffer_pool.h>

namespace luisa::compute::cuda {

CUDAHostBufferPool::CUDAHostBufferPool(size_t size, bool write_combined) noexcept
    : _first_fit{std::max(next_pow2(size), static_cast<size_t>(4096u)), alignment},
      _write_combined{write_combined} {
    auto flags = _write_combined ? CU_MEMHOSTALLOC_WRITECOMBINED : 0;
    LUISA_CHECK_CUDA(cuMemHostAlloc(
        reinterpret_cast<void **>(&_memory),
        _first_fit.size(), flags));
}

CUDAHostBufferPool::~CUDAHostBufferPool() noexcept {
    LUISA_CHECK_CUDA(cuMemFreeHost(reinterpret_cast<void *>(_memory)));
}

CUDAHostBufferPool::View *CUDAHostBufferPool::allocate(size_t size) noexcept {
    auto view = [this, size] {
        std::scoped_lock lock{_mutex};
        auto node = _first_fit.allocate(size);
        return node ? View::create(node, this) : nullptr;
    }();
    if (view == nullptr) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to allocate {} bytes from "
            "CUDAHostBufferPool. Falling back "
            "to ad-hoc allocation.",
            size);
        view = View::create(luisa::allocate_with_allocator<std::byte>(size));
    }
    return view;
}

void CUDAHostBufferPool::recycle(FirstFit::Node *node) noexcept {
    std::scoped_lock lock{_mutex};
    _first_fit.free(node);
}

[[nodiscard]] auto &host_buffer_recycle_context_pool() noexcept {
    static Pool<CUDAHostBufferPool::View> pool;
    return pool;
}

inline CUDAHostBufferPool::View::View(std::byte *handle) noexcept
    : _handle{handle} {}

inline CUDAHostBufferPool::View::View(FirstFit::Node *node, CUDAHostBufferPool *pool) noexcept
    : _handle{node}, _pool{pool} {}

std::byte *CUDAHostBufferPool::View::address() const noexcept {
    return is_pooled() ?
               _pool->memory() + node()->offset() :
               static_cast<std::byte *>(_handle);
}

void CUDAHostBufferPool::View::recycle() noexcept {
    if (is_pooled()) [[likely]] {
        _pool->recycle(node());
    } else {
        luisa::deallocate_with_allocator(static_cast<std::byte *>(_handle));
    }
    host_buffer_recycle_context_pool().destroy(this);
}

CUDAHostBufferPool::View *CUDAHostBufferPool::View::create(std::byte *handle) noexcept {
    return host_buffer_recycle_context_pool().create(handle);
}

CUDAHostBufferPool::View *CUDAHostBufferPool::View::create(FirstFit::Node *node, CUDAHostBufferPool *pool) noexcept {
    return host_buffer_recycle_context_pool().create(node, pool);
}

}// namespace luisa::compute::cuda
