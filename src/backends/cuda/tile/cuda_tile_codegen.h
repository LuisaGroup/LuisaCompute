#pragma once

#include <array>
#include <luisa/tile/ir.h>

namespace luisa::compute::cuda::native_tile {

struct BufferArgument {
    tile::ScalarType element;
    uint64_t minimum_size_bytes;
    bool read{false};
    bool written{false};
};

// The source is CUDA Tile C++, not SIMT CUDA C++. Compile with NVRTC
// --enable-tile --tile-only --ftz=false, get TileIR, then invoke tileiras.
// It must be launched with block (1,1,1), regardless of logical Tile sizes.
struct Artifact {
    luisa::string source;
    luisa::string error;
    luisa::string entry{"luisa_tile_main"};
    std::array<uint32_t, 3u> grid{1u, 1u, 1u};
    std::array<uint32_t, 3u> block{1u, 1u, 1u};
    luisa::vector<BufferArgument> arguments;
    [[nodiscard]] bool ok() const noexcept { return error.empty() && !source.empty(); }
};

// A deliberately bounded, source-producing emitter. Unsupported IR is an
// error, never a fallback to kernel-name recognition, SIMT, or reduced FP32.
// Buffers and values support bool, i32/u32/i64/u64, and strict FP32, with
// explicit FP16/BF16 storage/conversion/MMA. Pure maps,
// named-axis broadcasts and closed unordered add/min/max reducers use native
// Tiles; ordered/custom reductions retain their scalar contribution order.
// Closed unordered prefix sums and proven bitonic partner permutations use
// native scans and rearrangements. MMA permits shared singleton batch axes.
// Other lane-dependent gathers, nested maps and explicit layout constraints reject.
// Buffer parameters are contiguous row-major views with static extents.
[[nodiscard]] Artifact generate(const tile::Function &function) noexcept;

}// namespace luisa::compute::cuda::native_tile
