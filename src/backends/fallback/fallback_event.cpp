#include <chrono>
#include <cstdlib>
#include <string_view>
#include <thread>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(_M_ARM64) && defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#include "fallback_event.h"

namespace luisa::compute::fallback {

namespace {
inline void event_spin_pause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_pause();
#elif defined(_M_ARM64) || defined(__aarch64__)
#if defined(_MSC_VER) && !defined(__clang__)
    __yield();
#else
    asm volatile("yield" ::: "memory");
#endif
#endif
}
[[nodiscard]] bool use_orig_event_wait() noexcept {
    static const bool orig = []() noexcept {
        if (const auto *s = std::getenv("LUISA_FALLBACK_BACKEND_SCHEDULER")) {
            if (std::string_view{s} == "akari_orig") { return true; }
        }
        if (const auto *q = std::getenv("LUISA_FALLBACK_QUEUE_MODE")) {
            if (std::string_view{q} == "orig") { return true; }
        }
        return false;
    }();
    return orig;
}
}// namespace

void FallbackEvent::signal(uint64_t fence_value) noexcept {
#if defined(__has_builtin) && __has_builtin(__c11_atomic_fetch_max)
    using A = _Atomic(uint64_t);
    auto p = reinterpret_cast<A *>(&_fence_value);
    __c11_atomic_fetch_max(p, fence_value, __ATOMIC_RELEASE);
#else
    // use a CAS loop to ensure the fence value is monotonically increasing
    auto current = _fence_value.load(std::memory_order_acquire);
    while (current < fence_value) {
        if (_fence_value.compare_exchange_weak(current, fence_value, std::memory_order_release, std::memory_order_acquire)) {
            break;
        }
    }
#endif
    if (!use_orig_event_wait()) {
        _fence_value.notify_all();
    }
}

void FallbackEvent::wait(uint64_t fence_value) const noexcept {
    if (use_orig_event_wait()) {
        while (!is_completed(fence_value)) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(50us);
        }
        return;
    }
    for (uint32_t spin = 0u; spin < 64u; ++spin) {
        if (is_completed(fence_value)) { return; }
        event_spin_pause();
    }
    auto cur = _fence_value.load(std::memory_order_acquire);
    while (cur < fence_value) {
        _fence_value.wait(cur, std::memory_order_acquire);
        cur = _fence_value.load(std::memory_order_acquire);
    }
}

}// namespace luisa::compute::fallback
