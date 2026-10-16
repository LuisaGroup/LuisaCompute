#pragma once

#include <luisa/tile/collective_prefix.h>
#include "cuda_tile_codegen.h"

namespace luisa::compute::cuda::native_tile {

// Independent ordinary CUDA source for an optional realization of a proved
// existing prefix. It never modifies or replaces the original Tile source.
// Compile with strict FP32 flags and without --enable-tile or -restrict.
struct CubScanArtifact {
    luisa::string error;
    luisa::string source;
    luisa::string entry{"luisa_tile_cub_scan"};
    std::array<uint32_t, 3u> grid{1u, 1u, 1u};
    std::array<uint32_t, 3u> block{1u, 1u, 1u};
    StreamingScanGuard guard{};
    uint32_t alignment_mask{0u};
    uint32_t threads{0u};
    uint32_t chunk_extent{0u};
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !source.empty(); }
};

// The caller must retain a successfully compiled original for every fallback.
// Fresh shared semantic proof is followed by this realizer's ABI/shape gates:
// equal F16/BF16 storage, one full row/program, width divisible by 8*threads.
// Threads may be 128/256/512/1024. Every thread owns eight contiguous elements.
// Actual final pointers must satisfy alignment_mask's 16B requirement AND
// guard's complete disjoint byte intervals before selecting this module/block.
[[nodiscard]] CubScanArtifact generate_cub_scan(
    const tile::Function &function, const Artifact &original, uint32_t threads) noexcept;

}// namespace luisa::compute::cuda::native_tile
