#pragma once

// Private, bounded realization of a proved closed Tile IR prefix dataflow.
#include <luisa/tile/collective_plan.h>
#include "cuda_tile_codegen.h"

namespace luisa::compute::cuda::native_tile {

struct StreamingScanPlan {
    luisa::string error;
    uint32_t input_slot{0u};
    uint32_t output_slot{0u};
    tile::ScalarType storage{tile::ScalarType::INVALID};
    uint64_t rows{0u}, columns{0u}, padded_columns{0u};
    uint32_t rows_per_program{0u}, chunk_extent{0u};
    uint64_t minimum_bytes{0u};
    [[nodiscard]] bool ok() const noexcept { return error.empty() && minimum_bytes != 0u; }
};

// A successful existing artifact is mandatory. The implementation performs
// shared collective admission itself: callers cannot forge an analysis receipt.
[[nodiscard]] StreamingScanPlan match_streaming_scan(
    const tile::Function &function, const Artifact &original, uint32_t chunk) noexcept;

// Returns only the independent entry, to append after the unchanged successful
// original source. The original generic/aligned entries are never rewritten.
[[nodiscard]] luisa::string emit_streaming_scan_entry(
    const StreamingScanPlan &plan, const Artifact &original,
    uint32_t worker_warps = 0u, uint32_t target_sm = 0u) noexcept;

// Ineligibility leaves the successful original artifact/source unchanged.
void append_streaming_scan(Artifact &original, const tile::Function &function,
                           uint32_t chunk, uint32_t worker_warps = 0u, uint32_t target_sm = 0u) noexcept;

}// namespace luisa::compute::cuda::native_tile
