#include "LCDevice.h"

#include "../../common/tile_xir_kernel.h"
#include "../../common/hlsl/tile_codegen.h"

namespace lc::dx {

// Bounded native MMA programs lower directly to HLSL. Other Tile programs use
// TileIR -> XIR -> xir2ast -> the ordinary HLSL/DXIL create_shader entry.
ShaderCreationInfo LCDevice::create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                                const tile::CompileOptions &tile_options,
                                                tile::KernelMetadata &metadata) noexcept {
    auto warp_size = static_cast<uint32_t>(compute_warp_size());
    auto native_selected = false;
    auto info = hlsl::create_tile_kernel_via_hlsl(
        this, option, kernel, tile_options, metadata,
        {.warp_size = warp_size, .max_block_size = 1024u, .max_shared_memory = 32u * 1024u, .backend_label = "HLSL/DXIL"},
        native_selected);
    if (native_selected) { return info; }
    auto config = backend_detail::GPUTileTargetConfig{
        .warp_size = warp_size,
        .max_block_size = 1024u,
        .max_local_bytes = 64u * 1024u,
        .backend_label = "HLSL/DXIL"};
    return backend_detail::create_tile_kernel_via_ast(this, option, kernel, tile_options, metadata, config);
}

}// namespace lc::dx
