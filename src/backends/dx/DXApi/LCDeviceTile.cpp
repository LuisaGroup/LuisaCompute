#include "LCDevice.h"

#include "../../common/tile_xir_kernel.h"

namespace lc::dx {

// Tile fallback: TileIR -> XIR bridge -> xir2ast -> the ordinary HLSL/DXIL
// create_shader entry. D3D12 guarantees a 32-lane wave and 1024 threads/group.
ShaderCreationInfo LCDevice::create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                                const tile::CompileOptions &tile_options,
                                                tile::KernelMetadata &metadata) noexcept {
    auto config = backend_detail::GPUTileTargetConfig{
        .warp_size = static_cast<uint32_t>(compute_warp_size()),
        .max_block_size = 1024u,
        .max_local_bytes = 64u * 1024u,
        .backend_label = "HLSL/DXIL"};
    return backend_detail::create_tile_kernel_via_ast(this, option, kernel, tile_options, metadata, config);
}

}// namespace lc::dx
