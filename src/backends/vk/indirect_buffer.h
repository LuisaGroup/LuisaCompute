#pragma once

#include "default_buffer.h"
#include "../common/indirect_dispatch_layout.h"

namespace lc::vk {

class IndirectBuffer final : public DefaultBuffer {
private:
    size_t _capacity;
    mutable std::atomic_bool _header_initialization_claimed{false};

    [[nodiscard]] static size_t _checked_size(size_t capacity) noexcept {
        LUISA_ASSERT(capacity > 0u,
                     "Vulkan indirect-dispatch buffer capacity must be positive.");
        LUISA_ASSERT(capacity <= std::numeric_limits<uint32_t>::max(),
                     "Vulkan indirect-dispatch buffer capacity {} exceeds the "
                     "32-bit record-index ABI.",
                     capacity);
        size_t size = 0u;
        LUISA_ASSERT(IndirectDispatchLayout::try_total_size(capacity, size),
                     "Vulkan indirect-dispatch buffer size overflows for "
                     "capacity {}.",
                     capacity);
        return size;
    }

public:
    IndirectBuffer(Device *device, size_t capacity)
        : DefaultBuffer{device, _checked_size(capacity)},
          _capacity{capacity} {}

    [[nodiscard]] bool is_indirect_dispatch_buffer() const noexcept override {
        return true;
    }
    [[nodiscard]] size_t indirect_dispatch_capacity() const noexcept override {
        return _capacity;
    }
    [[nodiscard]] bool claim_indirect_header_initialization() const noexcept override {
        return !_header_initialization_claimed.exchange(
            true, std::memory_order_acq_rel);
    }
    [[nodiscard]] bool indirect_header_initialization_claimed() const noexcept override {
        return _header_initialization_claimed.load(
            std::memory_order_acquire);
    }
};

}// namespace lc::vk
