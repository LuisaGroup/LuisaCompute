#pragma once

#include <array>
#include <luisa/tile/ir.h>
#include "cuda_tile_streaming_guard.h"

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
    // Optional independent entry. The original entry never contains alignment
    // assumptions; only a host-side test of the final device pointers may
    // select this specialization. Bits index the direct device buffer ABI.
    luisa::string aligned16_entry;
    uint32_t aligned16_buffer_mask{0u};
    // Full, chunk-aligned loads represented as partition_view only in the
    // independently selected aligned16 entry. The original source is intact.
    uint32_t aligned16_partition_loads{0u};
    // Diagnostic opt-ins: immutable-value realizations, never grid changes.
    uint32_t scan_chunk_extent{0u};
    uint32_t chunked_scan_operations{0u};
    uint32_t independent_axis_extent{0u};
    uint32_t partitioned_collective_operations{0u};
    // Optional streaming entry keeps the original source prefix byte-identical.
    luisa::string streaming_scan_entry;
    luisa::string streaming_scan_diagnostic;
    size_t streaming_scan_source_offset{0u};
    uint32_t streaming_scan_chunk_extent{0u};
    StreamingScanGuard streaming_scan_guard{};
    // A separate row-partitioned realization has its own physical grid.
    // Selection additionally requires disjoint final input/output intervals.
    luisa::string partition_entry;
    luisa::string partition_diagnostic;
    size_t partition_source_offset{0u};
    uint32_t partition_rows{0u};
    uint32_t partition_original_rows{0u};
    std::array<uint32_t, 3u> partition_grid{1u, 1u, 1u};
    StreamingScanGuard partition_guard{};
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
[[nodiscard]] Artifact generate(const tile::Function &function, bool enable_fast_math = false,
                                bool enable_aligned16 = false,
                                uint32_t worker_warps = 0u, uint32_t target_sm = 0u,
                                uint32_t scan_chunk_extent = 0u,
                                uint32_t independent_axis_extent = 0u) noexcept;

}// namespace luisa::compute::cuda::native_tile
