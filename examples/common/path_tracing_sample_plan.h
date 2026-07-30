#pragma once

#include <cstdint>

namespace luisa::ref {

inline constexpr uint32_t DEFAULT_PATH_TRACING_SPP = 1024u;

struct PathTracingSamplePassPlan {
    uint64_t total_spp{0u};
    uint32_t max_spp_per_dispatch{1u};
    bool infinite{false};

    [[nodiscard]] constexpr bool has_next(uint64_t completed_spp) const noexcept {
        return max_spp_per_dispatch != 0u &&
               (infinite || completed_spp < total_spp);
    }

    [[nodiscard]] constexpr uint32_t next_dispatch_spp(uint64_t completed_spp) const noexcept {
        if (!has_next(completed_spp)) { return 0u; }
        if (infinite) { return max_spp_per_dispatch; }
        auto remaining = total_spp - completed_spp;
        return remaining < max_spp_per_dispatch ?
                   static_cast<uint32_t>(remaining) :
                   max_spp_per_dispatch;
    }
};

}// namespace luisa::ref
