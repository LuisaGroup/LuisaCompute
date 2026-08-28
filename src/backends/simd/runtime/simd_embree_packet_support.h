#pragma once

#include <cstdint>

namespace luisa::compute::simd {

struct SIMDEmbreeNativeRayPacketSupport {
    bool w4{false};
    bool w8{false};
    bool w16{false};

    [[nodiscard]] constexpr bool supports(
        uint32_t logical_width) const noexcept {
        switch (logical_width) {
            case 1u: return true;
            case 2u:
            case 4u: return w4;
            case 8u: return w8;
            case 16u: return w16;
            default: return false;
        }
    }
};

}// namespace luisa::compute::simd
