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
namespace cuda_shader_tile_detail {
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

}  // namespace cuda_shader_tile_detail
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
    auto ret = cuda_shader_tile_detail::load_tile_ptx(&_module, &_function, ptx.data(), ptx.size(), _entry);
    if (ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
        CUDAShader::_patch_ptx_version(ptx);
        ret = cuda_shader_tile_detail::load_tile_ptx(&_module, &_function, ptx.data(), ptx.size(), _entry);
    }
    LUISA_CHECK_CUDA(ret);
    // Retain the loaded (possibly version-patched) PTX for cross-backend
    // import, matching CUDAShaderNative's plain-PTX path.
    _module_image = std::move(ptx);
}

CUDAShaderTile::CUDAShaderTile(CUmodule module, CUfunction function, luisa::string entry,
                               const std::array<uint32_t, 3u> &grid,
                               luisa::vector<uint32_t> buffer_arguments,
                               luisa::vector<Usage> argument_usages,
                               CUfunction aligned16_function,
                               uint32_t aligned16_buffer_mask,
                               CUfunction streaming_scan_function,
                               native_tile::StreamingScanGuard streaming_scan_guard,
                               CUfunction partition_function,
                               std::array<uint32_t, 3u> partition_grid,
                               native_tile::StreamingScanGuard partition_guard) noexcept
    : CUDAShader{nullptr, std::move(argument_usages)},
      _module{module}, _function{function}, _aligned16_function{aligned16_function},
      _aligned16_buffer_mask{aligned16_buffer_mask},
      _streaming_scan_function{streaming_scan_function}, _streaming_scan_guard{streaming_scan_guard},
      _partition_function{partition_function}, _partition_grid{partition_grid}, _partition_guard{partition_guard},
      _entry{std::move(entry)},
      _grid{grid}, _block_size{1u, 1u, 1u},
      _buffer_arguments{std::move(buffer_arguments)} {
    // module_image remains empty: Vulkan interop uses the DSL parameter ABI.
    // CUDA graphs encode Tile's direct pointer list explicitly.
    LUISA_ASSERT(_module != nullptr && _function != nullptr, "Native Tile requires an owned module and entry.");
    LUISA_ASSERT(_buffer_arguments.size() <= 31u &&
                     ((_aligned16_function == nullptr) == (_aligned16_buffer_mask == 0u)) &&
                     (_aligned16_buffer_mask >> _buffer_arguments.size()) == 0u,
                 "Native Tile aligned entry has an invalid device binding mask.");
    LUISA_ASSERT(_streaming_scan_function == nullptr ||
                     (_streaming_scan_guard.input_slot < _buffer_arguments.size() &&
                      _streaming_scan_guard.output_slot < _buffer_arguments.size() &&
                      _streaming_scan_guard.input_slot != _streaming_scan_guard.output_slot &&
                      _streaming_scan_guard.input_bytes != 0u && _streaming_scan_guard.output_bytes != 0u),
                 "Native Tile streaming entry has an invalid static range descriptor.");
    LUISA_ASSERT(_partition_function == nullptr ||
                     (_streaming_scan_function == nullptr && _aligned16_function == nullptr &&
                      _partition_guard.input_slot < _buffer_arguments.size() &&
                      _partition_guard.output_slot < _buffer_arguments.size() &&
                      _partition_guard.input_slot != _partition_guard.output_slot &&
                      _partition_guard.input_bytes != 0u && _partition_guard.output_bytes != 0u &&
                      _partition_grid[0u] != 0u && _partition_grid[1u] == 1u && _partition_grid[2u] == 1u),
                 "Native Tile partition entry has an invalid launch or static range descriptor.");
}

CUDAShaderTile::~CUDAShaderTile() noexcept {
    if (_cub_scan_module != nullptr) { LUISA_CHECK_CUDA(cuModuleUnload(_cub_scan_module)); }
    LUISA_CHECK_CUDA(cuModuleUnload(_module));
}

bool CUDAShaderTile::install_cub_scan(CUmodule module, CUfunction function,
                                      std::array<uint32_t, 3u> grid, uint3 block,
                                      native_tile::StreamingScanGuard guard,
                                      uint32_t alignment_mask,
                                      luisa::string source, luisa::string identity) noexcept {
    if (module == nullptr || function == nullptr || module == _module || _cub_scan_module != nullptr ||
        _aligned16_function != nullptr || _streaming_scan_function != nullptr || _partition_function != nullptr || _buffer_arguments.size() > 31u ||
        guard.input_slot >= _buffer_arguments.size() || guard.output_slot >= _buffer_arguments.size() ||
        guard.input_slot == guard.output_slot || guard.input_bytes == 0u || guard.output_bytes == 0u ||
        grid[0u] == 0u || grid[1u] != 1u || grid[2u] != 1u || block.y != 1u || block.z != 1u ||
        (block.x != 128u && block.x != 256u && block.x != 512u && block.x != 1024u) ||
        source.empty() || identity.empty()) { return false; }
    auto expected_mask = (uint32_t{1u} << guard.input_slot) | (uint32_t{1u} << guard.output_slot);
    if (alignment_mask != expected_mask) { return false; }
    _cub_scan_module = module;
    _cub_scan_function = function;
    _cub_scan_grid = grid;
    _cub_scan_block = block;
    _cub_scan_guard = guard;
    _cub_scan_alignment_mask = alignment_mask;
    _cub_scan_source = std::move(source);
    _cub_scan_identity = std::move(identity);
    return true;
}

bool CUDAShaderTile::encode_buffer_pointers(luisa::span<const Argument> args,
                                            luisa::span<CUdeviceptr> pointers) const noexcept {
    if (args.size() != argument_count() || pointers.size() != _buffer_arguments.size() || pointers.size() > 31u) { return false; }
    // Validate unused host parameters too, before graph dependency analysis.
    for (auto &&arg : args) {
        if (arg.tag != Argument::Tag::BUFFER || arg.buffer.handle == ~uint64_t{0}) { return false; }
        auto base = reinterpret_cast<const CUDABufferBase *>(arg.buffer.handle);
        if (base == nullptr || base->is_indirect() || arg.buffer.offset > base->size_bytes() ||
            arg.buffer.size > base->size_bytes() - arg.buffer.offset) { return false; }
    }
    for (auto slot = size_t{0u}; slot < pointers.size(); slot++) {
        auto index = _buffer_arguments[slot];
        if (index >= args.size()) { return false; }
        auto &&arg = args[index].buffer;
        auto buffer = reinterpret_cast<const CUDABuffer *>(arg.handle);
        pointers[slot] = buffer->binding(arg.offset, arg.size).handle;
    }
    return true;
}

CUDAShaderTile::Launch CUDAShaderTile::select_launch(luisa::span<const CUdeviceptr> pointers) const noexcept {
    auto launch = Launch{_function, _grid, _block_size};
    if (pointers.size() != _buffer_arguments.size()) { return launch; }
    if (_cub_scan_function != nullptr && native_tile::streaming_scan_disjoint(_cub_scan_guard, pointers)) {
        CUdeviceptr alignment_bits = 0u;
        for (auto slot = size_t{0u}; slot < pointers.size(); slot++) {
            if ((_cub_scan_alignment_mask & (uint32_t{1u} << slot)) != 0u) { alignment_bits |= pointers[slot]; }
        }
        if ((alignment_bits & 15u) == 0u) { return Launch{_cub_scan_function, _cub_scan_grid, _cub_scan_block}; }
    }
    if (_partition_function != nullptr && native_tile::streaming_scan_disjoint(_partition_guard, pointers)) {
        return Launch{_partition_function, _partition_grid, _block_size};
    }
    if (_streaming_scan_function != nullptr && native_tile::streaming_scan_disjoint(_streaming_scan_guard, pointers)) {
        launch.function = _streaming_scan_function;
        return launch;
    }
    if (_aligned16_function == nullptr) { return launch; }
    CUdeviceptr alignment_bits = 0u;
    for (auto slot = size_t{0u}; slot < pointers.size(); slot++) {
        if ((_aligned16_buffer_mask & (uint32_t{1u} << slot)) != 0u) { alignment_bits |= pointers[slot]; }
    }
    if ((alignment_bits & 15u) == 0u) { launch.function = _aligned16_function; }
    return launch;
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
    std::array<CUdeviceptr, 31u> pointers{};
    std::array<void *, 31u> kernel_parameters{};
    LUISA_ASSERT(encode_buffer_pointers(args, {pointers.data(), parameter_count}),
                 "Direct-buffer Tile arguments have an invalid type, count, or range.");
    for (auto slot = size_t{0u}; slot < parameter_count; slot++) {
        kernel_parameters[slot] = &pointers[slot];
    }

    auto launch = select_launch({pointers.data(), parameter_count});
    auto block = launch.block;
    auto stream = encoder.stream()->handle();
    LUISA_CHECK_CUDA(cuLaunchKernel(
        launch.function,
        launch.grid[0], launch.grid[1], launch.grid[2],
        block.x, block.y, block.z,
        0u, stream, kernel_parameters.data(), nullptr));
}

}// namespace luisa::compute::cuda
