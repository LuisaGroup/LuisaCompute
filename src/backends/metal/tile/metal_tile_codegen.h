#pragma once

#include <luisa/tile/runtime.h>

namespace luisa::compute::metal {

struct MetalTileCode {
    uint3 block_size{0u};
    tile::KernelMetadata metadata;
    [[nodiscard]] bool ok() const noexcept { return metadata.error.empty(); }
};

// Read-only, bounded-family legalization. No Metal/TVM runtime is needed to
// inspect a plan, diagnose unsupported TileIR, or test the generated MSL.
[[nodiscard]] MetalTileCode lower_tile_to_mpp(const tile::Function &function,
                                              const tile::CompileOptions &options,
                                              uint32_t max_threads) noexcept;

}// namespace luisa::compute::metal
