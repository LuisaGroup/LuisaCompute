#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace luisa::ref {

struct NBodyWinnerEncoding {
    static constexpr uint32_t kParticleIndexBits = 11u;
    static constexpr uint32_t kMaxParticleCount = 1u << kParticleIndexBits;
    static constexpr uint32_t kParticleIndexMask = kMaxParticleCount - 1u;
    static constexpr uint32_t kDepthMask = ~kParticleIndexMask;
    static constexpr uint32_t kInvalid = std::numeric_limits<uint32_t>::max();
    static constexpr float kMinimumVisibleDistance = 0.1f;

    [[nodiscard]] static uint32_t pack(float distance, uint32_t particle_index) noexcept {
        static_assert(sizeof(float) == sizeof(uint32_t));
        static_assert(std::numeric_limits<float>::is_iec559);
        if (!(distance > kMinimumVisibleDistance) ||
            !std::isfinite(distance) ||
            particle_index >= kMaxParticleCount) {
            return kInvalid;
        }
        // Positive IEEE-754 floats have the same ordering as their unsigned
        // bit patterns. Replacing the low mantissa bits with the particle
        // index gives atomic-min a stable index tie-break without requiring
        // 64-bit buffer atomics.
        auto depth_bits = std::bit_cast<uint32_t>(distance);
        return (depth_bits & kDepthMask) | particle_index;
    }

    [[nodiscard]] static constexpr bool valid(uint32_t packed) noexcept {
        return packed != kInvalid;
    }

    [[nodiscard]] static constexpr uint32_t particle_index(uint32_t packed) noexcept {
        return packed & kParticleIndexMask;
    }

    [[nodiscard]] static constexpr uint32_t depth_bucket(uint32_t packed) noexcept {
        return packed & kDepthMask;
    }

    [[nodiscard]] static constexpr uint32_t select(uint32_t a, uint32_t b) noexcept {
        return a < b ? a : b;
    }
};

struct NBodyFootprint {
    int32_t min_x{0};
    int32_t min_y{0};
    int32_t max_x{-1};
    int32_t max_y{-1};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return min_x <= max_x && min_y <= max_y;
    }

    [[nodiscard]] constexpr uint32_t pixel_count() const noexcept {
        return valid() ?
                   static_cast<uint32_t>(max_x - min_x + 1) *
                       static_cast<uint32_t>(max_y - min_y + 1) :
                   0u;
    }

    [[nodiscard]] constexpr bool contains(int32_t x, int32_t y) const noexcept {
        return valid() && x >= min_x && x <= max_x &&
               y >= min_y && y <= max_y;
    }
};

inline constexpr int32_t kNBodyGlowRadius = 2;

[[nodiscard]] constexpr NBodyFootprint plan_nbody_footprint(
    int32_t center_x, int32_t center_y,
    int32_t width, int32_t height) noexcept {
    // Match the renderer's visibility rule: the projected particle center
    // must be in the viewport, then its 5x5 footprint is clipped per pixel.
    if (width <= 0 || height <= 0 ||
        center_x < 0 || center_x >= width ||
        center_y < 0 || center_y >= height) {
        return {};
    }
    return {
        center_x < kNBodyGlowRadius ? 0 : center_x - kNBodyGlowRadius,
        center_y < kNBodyGlowRadius ? 0 : center_y - kNBodyGlowRadius,
        center_x + kNBodyGlowRadius >= width ? width - 1 : center_x + kNBodyGlowRadius,
        center_y + kNBodyGlowRadius >= height ? height - 1 : center_y + kNBodyGlowRadius,
    };
}

}// namespace luisa::ref
