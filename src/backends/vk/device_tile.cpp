#include "device.h"

#include "../../common/tile_xir_kernel.h"

#include <algorithm>

namespace lc::vk {

// Tile fallback: TileIR -> XIR bridge -> xir2ast -> the ordinary Vulkan
// create_shader entry (native XIR->SPIR-V when compiled in, HLSL->SPIR-V
// otherwise). The subgroup size is the XIR packet width; the block width is
// bounded by the physical device limit.
ShaderCreationInfo Device::create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                              const tile::CompileOptions &tile_options,
                                              tile::KernelMetadata &metadata) noexcept {
    auto max_block_size = std::min<uint32_t>(properties().limits.maxComputeWorkGroupSize[0], 1024u);
    auto config = backend_detail::GPUTileTargetConfig{
        .warp_size = compute_warp_size(),
        .max_block_size = max_block_size == 0u ? 1024u : max_block_size,
        .max_local_bytes = 64u * 1024u,
        .backend_label = "SPIR-V"};
    return backend_detail::create_tile_kernel_via_ast(this, option, kernel, tile_options, metadata, config);
}

}// namespace lc::vk
