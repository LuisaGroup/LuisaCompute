//
// Created by mike on 1/9/26.
//

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_stage_buffer_pool.h"

namespace luisa::compute::hip {

namespace {

struct HIPStageProfileStats {
    std::atomic<uint64_t> total_allocs{0u};
    std::atomic<uint64_t> adhoc_count{0u};
    std::atomic<uint64_t> adhoc_bytes{0u};
    std::atomic<uint64_t> adhoc_ns{0u};
};

[[nodiscard]] bool hip_stage_profile_enabled() noexcept {
    static const bool enabled = [] {
        using namespace std::string_view_literals;
        auto env = std::getenv("LUISA_HIP_PROFILE_SCENE_BUILD");
        return env != nullptr && std::string_view{env} == "1"sv;
    }();
    return enabled;
}

[[nodiscard]] HIPStageProfileStats &hip_stage_profile_stats() noexcept {
    static HIPStageProfileStats stats;
    static const bool registered = [] {
        if (hip_stage_profile_enabled()) {
            std::atexit([] {
                auto total = stats.total_allocs.load(std::memory_order_relaxed);
                auto adhoc = stats.adhoc_count.load(std::memory_order_relaxed);
                if (total == 0u && adhoc == 0u) { return; }
                LUISA_INFO(
                    "[HIP_SCENE_PROFILE] stage_allocs={} adhoc_fallbacks={} adhoc_bytes={} adhoc_hipHostMalloc={:.2f}ms",
                    total, adhoc,
                    stats.adhoc_bytes.load(std::memory_order_relaxed),
                    static_cast<double>(stats.adhoc_ns.load(std::memory_order_relaxed)) * 1e-6);
            });
        }
        return true;
    }();
    static_cast<void>(registered);
    return stats;
}

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
    const auto profile = hip_stage_profile_enabled();
    if (profile) {
        hip_stage_profile_stats().total_allocs.fetch_add(1u, std::memory_order_relaxed);
    }
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
            using ProfileClock = std::chrono::steady_clock;
            auto t0 = profile ? ProfileClock::now() : ProfileClock::time_point{};
            std::byte *p = nullptr;
            LUISA_CHECK_HIP(hipHostMalloc(reinterpret_cast<void **>(&p), size, hipHostMallocMapped));
            if (profile) {
                auto t1 = ProfileClock::now();
                auto &s = hip_stage_profile_stats();
                s.adhoc_count.fetch_add(1u, std::memory_order_relaxed);
                s.adhoc_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
                s.adhoc_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                    std::memory_order_relaxed);
            }
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
