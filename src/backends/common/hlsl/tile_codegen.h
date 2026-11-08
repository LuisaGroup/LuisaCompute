#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/tile/runtime.h>

namespace lc::hlsl {

using namespace luisa;
using namespace luisa::compute;

struct HlslTileTargetConfig {
    uint32_t warp_size{32u};
    uint32_t max_block_size{1024u};
    size_t max_shared_memory{32u * 1024u};
    luisa::string_view backend_label{"HLSL"};
};

struct HlslTileCode {
    uint3 block_size{0u};
    tile::KernelMetadata metadata;
    // True once the input has been recognized as part of the bounded native
    // MMA family. DX/VK use this to retain the general XIR fallback for other
    // Tile programs without hiding errors in a native candidate.
    bool native_candidate{false};
    [[nodiscard]] bool ok() const noexcept { return metadata.error.empty(); }
};

// Read-only TileIR legalization and HLSL source generation. The current native
// family is the same dense FP32 MMA/store subset as the Metal bootstrap, but is
// realized as explicitly distributed scalar multiply-accumulate loops. Eight-
// output whole-group atoms use direct 4x2 register tiles; smaller evenly
// distributed atoms may stage operands in shared memory within the target
// budget. Neither path needs tensor-core or cooperative-matrix extensions.
[[nodiscard]] HlslTileCode lower_tile_to_hlsl(
    const tile::Function &function,
    const tile::CompileOptions &options,
    const HlslTileTargetConfig &target) noexcept;

// Compile a lowered native candidate through the ordinary AST/HLSL resource
// ABI. native_selected is false only when the caller should use its general
// Tile XIR->AST fallback instead.
[[nodiscard]] ShaderCreationInfo create_tile_kernel_via_hlsl(
    DeviceInterface *device, const ShaderOption &option,
    const tile::Function &kernel, const tile::CompileOptions &tile_options,
    tile::KernelMetadata &metadata, const HlslTileTargetConfig &target,
    bool &native_selected) noexcept;

}// namespace lc::hlsl
