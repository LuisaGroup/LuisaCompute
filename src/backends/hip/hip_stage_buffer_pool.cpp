//
// Created by mike on 1/9/26.
//

#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_stage_buffer_pool.h"

namespace luisa::compute::hip {

namespace {
[[nodiscard]] auto &host_buffer_recycle_context_pool() noexcept {
    static Pool<HIPStageBufferPool::View> pool;
    return pool;
}
}// namespace

HIPStageBufferPool::View::View(std::byte *handle) noexcept
    : _handle{handle} {}

HIPStageBufferPool::View::View(FirstFit::Node *node, HIPStageBufferPool *pool) noexcept
    : _handle{node}, _pool{pool} {}

std::byte *HIPStageBufferPool::View::address() const noexcept {
    return is_pooled() ? _pool->memory() + node()->offset() :
                         static_cast<std::byte *>(_handle);
}

HIPStageBufferPool::View *HIPStageBufferPool::View::create(std::byte *handle) noexcept {
    return host_buffer_recycle_context_pool().create(handle);
}

HIPStageBufferPool::View *HIPStageBufferPool::View::create(FirstFit::Node *node, HIPStageBufferPool *pool) noexcept {
    return host_buffer_recycle_context_pool().create(node, pool);
}

void HIPStageBufferPool::View::recycle() noexcept {
    if (is_pooled()) [[likely]] {
        _pool->recycle(node());
    } else {
        LUISA_CHECK_HIP(hipHostFree(static_cast<std::byte *>(_handle)));
    }
    host_buffer_recycle_context_pool().destroy(this);
}

HIPStageBufferPool::HIPStageBufferPool(size_t size, bool write_combined) noexcept
    : _first_fit{std::max(std::bit_ceil(size), static_cast<size_t>(4096u)), alignment} {
    auto flags = write_combined ? hipHostMallocMapped | hipHostMallocWriteCombined :
                                  hipHostMallocMapped;
    LUISA_CHECK_HIP(hipHostMalloc(reinterpret_cast<void **>(&_memory), _first_fit.size(), flags));
}

HIPStageBufferPool::~HIPStageBufferPool() noexcept {
    LUISA_CHECK_HIP(hipHostFree(_memory));
}

HIPStageBufferPool::View *HIPStageBufferPool::allocate(size_t size, bool fallback_if_failed) noexcept {
    auto view = [this, size] {
        std::scoped_lock lock{_mutex};
        auto node = _first_fit.allocate(size);
        return node ? View::create(node, this) : nullptr;
    }();
    if (view == nullptr) [[unlikely]] {
        static thread_local bool warned = false;
        if (!warned) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to allocate {} bytes from HIPStageBufferPool. "
                "Falling back to ad-hoc allocation. Consecutive warnings "
                "would be suppressed to avoid flooding.",
                size);
            warned = true;
        }
        if (fallback_if_failed) {
            std::byte *p = nullptr;
            LUISA_CHECK_HIP(hipHostMalloc(reinterpret_cast<void **>(&p), size, hipHostMallocMapped));
            view = View::create(p);
        }
    }
    return view;
}

void HIPStageBufferPool::recycle(FirstFit::Node *node) noexcept {
    std::scoped_lock lock{_mutex};
    _first_fit.free(node);
}

}// namespace luisa::compute::hip
