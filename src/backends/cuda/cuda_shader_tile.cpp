#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>

#include "cuda_buffer.h"
#include "cuda_command_encoder.h"
#include "cuda_error.h"
#include "cuda_shader_printer.h"
#include "cuda_shader_tile.h"

namespace luisa::compute::cuda {

namespace {

// Loads a plain PTX module (Tile artifacts never use cudadevrt/kernel_launcher;
// indirect dispatch is out of scope for the CUDA Tile route). Returns the CUDA
// error so the caller can distinguish an unsupported PTX version from other
// module load failures.
CUresult load_tile_ptx(CUmodule *module, CUfunction *function,
                       const void *ptx, size_t ptx_size,
                       luisa::string_view entry) noexcept {
    if (auto ret = cuModuleLoadData(module, ptx); ret != CUDA_SUCCESS) {
        return ret;
    }
    auto entry_name = luisa::string{entry};
    if (auto ret = cuModuleGetFunction(function, *module, entry_name.c_str());
        ret != CUDA_SUCCESS) {
        return ret;
    }
    return CUDA_SUCCESS;
}

}// namespace

CUDAShaderTile::CUDAShaderTile(CUDADevice *device, luisa::vector<std::byte> ptx,
                               luisa::string entry,
                               const std::array<uint32_t, 3u> &grid,
                               uint3 block_size,
                               luisa::vector<uint32_t> buffer_arguments,
                               luisa::vector<Usage> argument_usages) noexcept
    : CUDAShader{CUDAShaderPrinter::create(luisa::span<const std::pair<luisa::string, luisa::string>>{}), std::move(argument_usages)},
      _entry{std::move(entry)},
      _grid{grid},
      _block_size{block_size},
      _buffer_arguments{std::move(buffer_arguments)} {
    static_cast<void>(device);
    auto ret = load_tile_ptx(&_module, &_function, ptx.data(), ptx.size(), _entry);
    if (ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
        CUDAShader::_patch_ptx_version(ptx);
        ret = load_tile_ptx(&_module, &_function, ptx.data(), ptx.size(), _entry);
    }
    LUISA_CHECK_CUDA(ret);
    // Retain the loaded (possibly version-patched) PTX for cross-backend
    // import, matching CUDAShaderNative's plain-PTX path.
    _module_image = std::move(ptx);
}

CUDAShaderTile::~CUDAShaderTile() noexcept {
    LUISA_CHECK_CUDA(cuModuleUnload(_module));
}

void CUDAShaderTile::_launch(CUDACommandEncoder &encoder,
                             ShaderDispatchCommand *command) const noexcept {

    LUISA_ASSERT(!command->is_indirect() && !command->is_multiple_dispatch(),
                 "Direct-buffer Tile shaders require a single static dispatch.");
    auto args = command->arguments();
    LUISA_ASSERT(args.size() == argument_count() && bound_argument_count() == 0u && printer() == nullptr,
                 "Direct-buffer Tile shader ABI mismatch.");
    auto dispatch_size = command->dispatch_size();
    if (any(dispatch_size == 0u)) { return; }

    auto parameter_count = _buffer_arguments.size();
    LUISA_ASSERT(parameter_count <= 31u,
                 "CUDA Tile shader exceeds the supported kernel parameter "
                 "envelope ({} device buffers).",
                 parameter_count);
    luisa::vector<CUdeviceptr> pointers(parameter_count);
    luisa::vector<void *> kernel_parameters(parameter_count);
    for (auto slot = size_t{0u}; slot < parameter_count; slot++) {
        auto index = _buffer_arguments[slot];
        LUISA_ASSERT(index < args.size() && args[index].tag == ShaderDispatchCommand::Argument::Tag::BUFFER,
                     "Direct-buffer Tile shader expects buffer arguments.");
        auto &arg = args[index].buffer;
        auto base = reinterpret_cast<const CUDABufferBase *>(arg.handle);
        LUISA_ASSERT(!base->is_indirect(), "Dispatch buffers are not Tile tensor arguments.");
        LUISA_ASSERT(arg.offset <= base->size_bytes() && arg.size <= base->size_bytes() - arg.offset,
                     "Direct-buffer Tile argument range exceeds its resource.");
        auto buffer = static_cast<const CUDABuffer *>(base);
        auto binding = buffer->binding(arg.offset, arg.size);
        // The generated CUDA kernel takes plain typed pointers. A nonzero view
        // offset is folded into the raw device address by CUDABuffer::binding.
        pointers[slot] = binding.handle;
        kernel_parameters[slot] = &pointers[slot];
    }

    auto block = _block_size;
    auto stream = encoder.stream()->handle();
    LUISA_CHECK_CUDA(cuLaunchKernel(
        _function,
        _grid[0], _grid[1], _grid[2],
        block.x, block.y, block.z,
        0u, stream, kernel_parameters.data(), nullptr));
}

}// namespace luisa::compute::cuda
