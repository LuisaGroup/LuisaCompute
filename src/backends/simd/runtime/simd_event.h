#pragma once

#include <atomic>
#include <cstdint>

namespace luisa::compute::simd {

class SIMDEvent {

private:
    std::atomic_uint64_t _value{0u};

public:
    void signal(uint64_t value) noexcept;
    void wait(uint64_t value) const noexcept;
    [[nodiscard]] bool is_completed(uint64_t value) const noexcept {
        return _value.load(std::memory_order_acquire) >= value;
    }
    [[nodiscard]] auto native_handle() noexcept { return &_value; }
};

}// namespace luisa::compute::simd
