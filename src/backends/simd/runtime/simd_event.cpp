#include "simd_event.h"

#include <chrono>
#include <thread>

namespace luisa::compute::simd {

void SIMDEvent::signal(uint64_t value) noexcept {
    auto current = _value.load(std::memory_order_acquire);
    while (current < value &&
           !_value.compare_exchange_weak(
               current, value, std::memory_order_release,
               std::memory_order_acquire)) {}
}

void SIMDEvent::wait(uint64_t value) const noexcept {
    while (!is_completed(value)) {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(50us);
    }
}

}// namespace luisa::compute::simd
