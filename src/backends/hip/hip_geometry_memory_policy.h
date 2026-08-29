#pragma once

#include <cstddef>

namespace luisa::compute::hip {

// High-quality construction is a preference, not a license to consume the
// memory needed by the scene itself. Reserving at most one quarter of current
// free VRAM for one BLAS scratch arena leaves room for the output and scene.
inline constexpr auto hiprt_high_quality_scratch_budget_denominator = size_t{4u};

[[nodiscard]] constexpr bool hiprt_high_quality_scratch_exceeds_budget(
    size_t scratch_bytes,
    size_t free_bytes) noexcept {
    // Overflow-free scratch/free > 1/4. The strict inequality deliberately
    // keeps the exact quarter boundary eligible for HighQualityBuild.
    return scratch_bytes != 0u &&
           (free_bytes == 0u ||
            scratch_bytes >
                free_bytes /
                    hiprt_high_quality_scratch_budget_denominator);
}

}// namespace luisa::compute::hip
