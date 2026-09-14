#pragma once

#include <array>

#include <cuda.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/ast/usage.h>

#include "cuda_shader.h"

namespace luisa::compute::cuda {

class CUDADevice;

// A statically shaped, direct-buffer Tile shader launched on CUDA.
//
// Unlike the ordinary DSL CUDAShaderNative, a Tile device artifact has:
//   * one __global__ entry whose parameters are plain typed buffer pointers
//     (no trailing uint4 launch-size parameter, no cudadevrt/kernel_launcher),
//   * a static grid/block configuration carried by the artifact, and
//   * a device binding order that may differ from the original host argument
//     order (buffer_arguments[device_slot] -> host parameter index).
//
// _launch therefore reorders ShaderDispatchCommand arguments into the device
// binding order and encodes each buffer view as a raw CUdeviceptr (base +
// offset), the exact by-value parameter the generated CUDA kernel expects.
class CUDAShaderTile final : public CUDAShader {

private:
    CUmodule _module{};
    CUfunction _function{};
    luisa::string _entry;
    std::array<uint32_t, 3u> _grid{1u, 1u, 1u};
    uint3 _block_size{0u, 0u, 0u};
    // Indexed by device buffer slot; each value indexes the original host
    // ShaderDispatchCommand argument (see DeviceArtifact::buffer_arguments).
    luisa::vector<uint32_t> _buffer_arguments;
    // The loaded (possibly version-patched) PTX text. Retained for Vulkan
    // VK_NV_cuda_kernel_launch interop parity with CUDAShaderNative.
    luisa::vector<std::byte> _module_image;

private:
    void _launch(CUDACommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept override;

public:
    // Loads PTX and resolves `entry` inside the caller's active CUDA context.
    // Retries with CUDAShader::_patch_ptx_version on old-driver
    // CUDA_ERROR_UNSUPPORTED_PTX_VERSION (mirroring CUDAShaderNative).
    CUDAShaderTile(CUDADevice *device, luisa::vector<std::byte> ptx,
                   luisa::string entry,
                   const std::array<uint32_t, 3u> &grid,
                   uint3 block_size,
                   luisa::vector<uint32_t> buffer_arguments,
                   luisa::vector<Usage> argument_usages) noexcept;
    ~CUDAShaderTile() noexcept override;
    [[nodiscard]] bool is_graph_compatible() const noexcept override { return true; }
    [[nodiscard]] void *handle() const noexcept override { return _function; }
    [[nodiscard]] luisa::span<const std::byte> module_image() const noexcept override { return _module_image; }
    [[nodiscard]] luisa::string_view entry() const noexcept override { return _entry; }
    [[nodiscard]] uint3 block_size() const noexcept override { return _block_size; }
    [[nodiscard]] size_t bound_argument_count() const noexcept override { return 0u; }
    [[nodiscard]] auto grid() const noexcept { return _grid; }
};

}// namespace luisa::compute::cuda
