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
// This first runtime slice accepts FP32 buffers and integer/bool/FP32 values.
// Buffer parameters are contiguous row-major views with static extents.
[[nodiscard]] Artifact generate(const tile::Function &function) noexcept;

}// namespace luisa::compute::cuda::native_tile
