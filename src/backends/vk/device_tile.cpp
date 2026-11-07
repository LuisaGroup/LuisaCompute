#include "device.h"

#include "../common/tile_xir_kernel.h"
#include "../common/hlsl/tile_codegen.h"

#include <algorithm>

namespace lc::vk {

// With DXC compatibility, bounded native MMA programs lower directly to HLSL.
// Other programs use TileIR -> XIR -> xir2ast -> the ordinary Vulkan
// create_shader entry. The XIR packet and block widths follow the device.
ShaderCreationInfo Device::create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                              const tile::CompileOptions &tile_options,
                                              tile::KernelMetadata &metadata) noexcept {
    auto max_block_size = std::min<uint32_t>(properties().limits.maxComputeWorkGroupSize[0], 1024u);
    max_block_size = max_block_size == 0u ? 1024u : max_block_size;
    auto warp_size = compute_warp_size();
#ifndef LC_NO_HLSL_BUILTIN
    auto native_selected = false;
    auto info = hlsl::create_tile_kernel_via_hlsl(
        this, option, kernel, tile_options, metadata,
        {.warp_size = warp_size, .max_block_size = max_block_size, .backend_label = "HLSL/SPIR-V"},
        native_selected);
    if (native_selected) { return info; }
#endif
    auto config = backend_detail::GPUTileTargetConfig{
        .warp_size = warp_size,
        .max_block_size = max_block_size,
        .max_local_bytes = 64u * 1024u,
        .backend_label = "SPIR-V"};
    return backend_detail::create_tile_kernel_via_ast(this, option, kernel, tile_options, metadata, config);
}

}// namespace lc::vk
