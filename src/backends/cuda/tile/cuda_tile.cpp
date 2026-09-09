// CUDA native-Tile factory: lowers a Tile kernel through the shared TIRx
// bridge into a CUDA device artifact (CUDA C source or NVPTX text) and then
// compiles/loads it for direct, statically shaped cuLaunchKernel launches.
//
// Mirror of src/backends/metal/tile/metal_tile.cpp for the CUDA backend.

#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>

#include <luisa/core/binary_io.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/string.h>
#include <luisa/tile/runtime.h>

#include "cuda_tile.h"
#include "../cuda_buffer.h"
#include "../cuda_device.h"
#include "../cuda_error.h"
#include "../cuda_shader_metadata.h"
#include "../cuda_shader_tile.h"

#ifdef LUISA_CUDA_TILE_TIRX
#include <luisa/tile/bridge/tirx/compiler.h>
#include <luisa/tile/bridge/tirx/lower.h>
#endif

namespace luisa::compute::cuda {

namespace {

// The ABI marker participates in cache identity. Bump it whenever the
// generated-kernel or launch ABI changes so stale PTX files are never reused.
constexpr auto cuda_tile_direct_buffer_abi = "cuda-tile-direct-buffers-v1";

[[nodiscard]] bool end_with_ptx(luisa::string_view name) noexcept {
    return name.ends_with(".ptx") || name.ends_with(".PTX");
}

[[nodiscard]] luisa::optional<CUDAShaderMetadata>
parse_tile_shader_metadata(luisa::string_view data) noexcept {
    constexpr luisa::string_view metadata_prefix = "// METADATA: ";
    if (!data.starts_with(metadata_prefix)) { return luisa::nullopt; }
    auto m = data.substr(metadata_prefix.size());
    return deserialize_cuda_shader_metadata(m);
}

// Reads the PTX cache pair produced by CUDADevice::create_tile_kernel. The
// sidecar uses the ordinary CUDA "// METADATA: ..." convention so a generic
// future loader can recognize a Tile artifact. A missing or mismatching entry
// simply returns an empty PTX vector (the caller recompiles).
[[nodiscard]] luisa::vector<std::byte> read_tile_shader_ptx(
    const BinaryIO *io, luisa::string_view name,
    const CUDAShaderMetadata &expected,
    bool use_user_path, bool use_cache) noexcept {
    luisa::vector<std::byte> ptx_data;
    auto metadata_name = luisa::format("{}.metadata", name);
    luisa::unique_ptr<BinaryStream> ptx_stream;
    luisa::unique_ptr<BinaryStream> metadata_stream;
    if (use_user_path) {
        ptx_stream = io->read_shader_bytecode(name);
        metadata_stream = io->read_shader_bytecode(metadata_name);
    } else if (use_cache) {
        ptx_stream = io->read_shader_cache(name);
        metadata_stream = io->read_shader_cache(metadata_name);
    }
    if (ptx_stream == nullptr || metadata_stream == nullptr ||
        ptx_stream->length() == 0u || metadata_stream->length() == 0u) {
        LUISA_VERBOSE("CUDA Tile shader '{}' is not found in cache; it will be compiled.", name);
        return {};
    }
    luisa::string metadata_text;
    metadata_text.resize(metadata_stream->length());
    metadata_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(metadata_text.data()),
        metadata_text.size() * sizeof(char)});
    ptx_data.resize(ptx_stream->length());
    ptx_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(ptx_data.data()),
        ptx_data.size() * sizeof(char)});
    auto cached = parse_tile_shader_metadata(metadata_text);
    if (!cached || *cached != expected) {
        LUISA_WARNING_WITH_LOCATION(
            "CUDA Tile shader '{}' cache entry is stale; it will be recompiled.", name);
        return {};
    }
    return ptx_data;
}

void write_tile_shader_ptx(
    const BinaryIO *io, luisa::string_view name,
    const CUDAShaderMetadata &metadata,
    luisa::span<const std::byte> ptx_data,
    bool use_user_path, bool use_cache) noexcept {
    auto serialized = luisa::format(
        "// METADATA: {}\n\n",
        serialize_cuda_shader_metadata(metadata));
    auto metadata_name = luisa::format("{}.metadata", name);
    luisa::span metadata_span{
        reinterpret_cast<const std::byte *>(serialized.data()),
        serialized.size()};
    if (use_user_path) {
        static_cast<void>(io->write_shader_bytecode(name, ptx_data));
        static_cast<void>(io->write_shader_bytecode(metadata_name, metadata_span));
    } else if (use_cache) {
        static_cast<void>(io->write_shader_cache(name, ptx_data));
        static_cast<void>(io->write_shader_cache(metadata_name, metadata_span));
    }
}

// Returns a loader result distinguishing "unsupported PTX version" from other
// load failures so the factory can patch/retry exactly like builtin kernels.
enum class TileLoadResult : uint8_t {
    OK,
    UNSUPPORTED_PTX_VERSION,
    FAILED,
};

TileLoadResult probe_tile_ptx(luisa::string_view entry,
                              luisa::span<const std::byte> ptx,
                              CUresult *error_out) noexcept {
    if (error_out) { *error_out = CUDA_SUCCESS; }
    CUmodule module{};
    auto ret = cuModuleLoadData(&module, ptx.data());
    if (ret != CUDA_SUCCESS) {
        if (error_out) { *error_out = ret; }
        return ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION ?
                   TileLoadResult::UNSUPPORTED_PTX_VERSION :
                   TileLoadResult::FAILED;
    }
    CUfunction function{};
    auto entry_name = luisa::string{entry};
    ret = cuModuleGetFunction(&function, module, entry_name.c_str());
    LUISA_CHECK_CUDA(cuModuleUnload(module));
    if (ret != CUDA_SUCCESS) {
        if (error_out) { *error_out = ret; }
        return ret == CUDA_ERROR_UNSUPPORTED_PTX_VERSION ?
                   TileLoadResult::UNSUPPORTED_PTX_VERSION :
                   TileLoadResult::FAILED;
    }
    return TileLoadResult::OK;
}

}// namespace

ShaderCreationInfo CUDADevice::create_tile_kernel(const ShaderOption &option,
                                                  const tile::Function &kernel,
                                                  const tile::CompileOptions &tile_options,
                                                  tile::KernelMetadata &metadata) noexcept {
    metadata = {};
    if (tile_options.xir != nullptr) {
        metadata.error = "CUDA cannot use the CPU XIR execution planner";
        return ShaderCreationInfo::make_invalid();
    }
    if (option.compile_only) {
        metadata.error = "Tile compile-only archives are not supported yet";
        return ShaderCreationInfo::make_invalid();
    }
    if (tile_options.lowering == tile::Lowering::TIRX) {
#ifdef LUISA_CUDA_TILE_TIRX
        auto fail = [&metadata](luisa::string_view message) noexcept {
            metadata.error = message;
            return ShaderCreationInfo::make_invalid();
        };
        Clock codegen_clock;
        auto attached = false;
        if (auto parent = kernel.parent_module()) {
            for (auto function : parent->functions()) { attached |= function == &kernel; }
        }
        if (!attached) { return fail("Tile function must belong to its owning module"); }

        auto lowered = tile::bridge::tirx::lower(kernel);
        if (!lowered) { return fail(lowered.error); }
        auto root = kernel.body().block(0u);
        for (auto &arg : root->arguments()) {
            auto volume = arg->type().index_space()->static_volume();
            if (arg->type().scalar_type() != tile::ScalarType::FLOAT32 || !volume || *volume == 0u ||
                *volume > INT32_MAX || *volume > SIZE_MAX / sizeof(float)) {
                return fail("CUDA TIRx Runtime currently requires nonempty, static, int32-addressable FP32 buffers");
            }
            auto usage = Usage::NONE;
            for (auto use : arg->use_list()) {
                if (use->index() != 0u) { return fail("Unknown Tile view argument effect"); }
                switch (use->user()->kind()) {
                    case tile::OperationKind::VIEW_LOAD:
                        usage = static_cast<Usage>(to_underlying(usage) | to_underlying(Usage::READ));
                        break;
                    case tile::OperationKind::VIEW_STORE:
                        usage = static_cast<Usage>(to_underlying(usage) | to_underlying(Usage::WRITE));
                        break;
                    default: return fail("Unknown Tile view argument effect");
                }
            }
            metadata.arguments.emplace_back(
                tile::KernelArgument{tile::ScalarType::FLOAT32, *volume * sizeof(float), usage});
        }

        int max_threads = 0;
        LUISA_CHECK_CUDA(cuDeviceGetAttribute(
            &max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            _handle.device()));
        auto shared_bytes = compute_max_shared_memory_size();
        auto options = tile_options.tirx ? *tile_options.tirx : tile::bridge::tirx::CompileOptions{};
        // The CUDA device-artifact route realizes every Tile tensor operator
        // through the reference SIMT expansion. Cooperative matrices and the
        // Metal-only planning knobs are hard errors, never silently ignored.
        if (options.cooperative_matrix) { return fail("CUDA Tile kernels do not support cooperative matrices; Tile MMA uses the reference multiply/add realization"); }
        if (options.metal_mpp) { return fail("Metal MPP memory atoms are not available on CUDA Tile kernels"); }
        if (options.planner.metal_subgroup_reductions ||
            options.planner.reduction_programs_per_group != 0u ||
            options.planner.reduction_unroll_factor != 1u ||
            options.planner.reduction_lane_elements != 1u ||
            options.planner.cache_reduction_inputs) {
            return fail("Metal SIMD-group reduction policies are not available on CUDA Tile kernels; REDUCE keeps the reference realization");
        }
        if (options.planner.program_order_rows != 1u || options.planner.program_order_columns != 1u) {
            return fail("program-order traversal is a Metal group-program option and is not available on CUDA Tile kernels");
        }
        if (options.planner.threads_per_group != 0u && tile_options.threads_per_group == 0u) {
            return fail("Exact TIRx planner thread counts are not available on the CUDA reference Tile mapper; pass threads_per_group through tile::CompileOptions and match the generated block");
        }
        options.cooperative_matrix = false;
        options.target = luisa::format(
            R"({{"kind":"cuda","thread_warp_size":32,"max_num_threads":{},"max_shared_memory_per_block":{}}})",
            max_threads, shared_bytes);
        // SplitHostDevice's pointer ABI requires disjoint writable args.
        // The Runtime wrapper checks actual BufferView ranges at launch.
        options.noalias = true;
        metadata.disjoint_writes = true;
        if (tile_options.threads_per_group != 0u) {
            if (options.planner.threads_per_group != 0u &&
                options.planner.threads_per_group != tile_options.threads_per_group) {
                return fail("Conflicting TIRx and Runtime thread constraints");
            }
            options.planner.threads_per_group = tile_options.threads_per_group;
        }
        auto compiled = tile::bridge::tirx::compile_device(std::move(lowered.value), kernel.name(), options);
        if (!compiled) { return fail(compiled.error); }
        auto &artifact = compiled.artifact;
        if (artifact.format != tile::bridge::tirx::DeviceArtifact::Format::CUDA_SOURCE &&
            artifact.format != tile::bridge::tirx::DeviceArtifact::Format::PTX) {
            return fail("Unsupported CUDA device artifact format");
        }
        if (artifact.buffer_arguments.size() > 31u) {
            return fail("CUDA Tile kernel exceeds the supported direct-buffer binding capacity");
        }
        auto block = make_uint3(artifact.block[0], artifact.block[1], artifact.block[2]);
        auto threads = static_cast<uint64_t>(block.x) * block.y * block.z;
        if (threads > static_cast<uint64_t>(max_threads) || threads == 0u ||
            threads % 32u != 0u ||
            (tile_options.threads_per_group && threads != tile_options.threads_per_group)) {
            return fail("TIRx device launch exceeds CUDA capacity, is not warp aligned, or conflicts with the exact thread constraint");
        }
        for (auto i = 0u; i < 3u; i++) {
            if (artifact.grid[i] > UINT32_MAX / artifact.block[i]) {
                return fail("TIRx dispatch extent overflows Runtime uint32 ABI");
            }
            metadata.dispatch_size[i] = artifact.grid[i] * artifact.block[i];
        }
        metadata.source = artifact.source;
        auto codegen_ms = codegen_clock.toc();
        auto is_cuda_source = artifact.format == tile::bridge::tirx::DeviceArtifact::Format::CUDA_SOURCE;
        metadata.realization = luisa::format(
            "{}; {} threads/group; {} group plans; direct-buffer ABI; fast_math={}",
            is_cuda_source ?
                "TIRx -> CUDA C -> NVRTC PTX -> Luisa Runtime" :
                "TIRx -> NVPTX PTX -> Luisa Runtime",
            threads, compiled.plans.size(), option.enable_fast_math);

        // ---- CUDA compiler/cache metadata (shares the DSL sidecar format) ----
        auto use_user_path = !option.name.empty();
        auto checksum = luisa::hash_combine({
            luisa::hash_value(metadata.source),
            luisa::hash_value(artifact.entry),
            luisa::hash_value(option.enable_fast_math),
            luisa::hash_value(block.x),
            luisa::hash_value(block.y),
            luisa::hash_value(block.z),
            luisa::hash_value(artifact.grid[0]),
            luisa::hash_value(artifact.grid[1]),
            luisa::hash_value(artifact.grid[2]),
            luisa::hash_value(luisa::string_view{cuda_tile_direct_buffer_abi}),
        });
        auto name = use_user_path ?
                        option.name :
                        luisa::format("kernel_{:016x}.tile.ptx", checksum);
        if (!end_with_ptx(name)) { name.append(".ptx"); }

        CUDAShaderMetadata shader_metadata{};
        shader_metadata.checksum = checksum;
        shader_metadata.kind = CUDAShaderMetadata::Kind::TILE;
        shader_metadata.enable_debug = option.enable_debug_info;
        shader_metadata.max_register_count = std::clamp(option.max_registers, 0u, 255u);
        shader_metadata.block_size = block;
        for (auto &arg : metadata.arguments) {
            shader_metadata.argument_usages.emplace_back(arg.usage);
        }

        // Try the in-memory NVRTC LRU first; it is filled by CUDACompiler on
        // every compile (including the ordinary DSL path) so a repeated Tile
        // compile in this process never re-runs the standalone compiler.
        luisa::vector<std::byte> ptx;
        if (option.enable_cache || use_user_path) {
            ptx = read_tile_shader_ptx(
                _io, name, shader_metadata,
                use_user_path, option.enable_cache);
        }

        // A PTX-format artifact already contains the final module text: skip
        // NVRTC entirely and use the code generator's output directly. Only the
        // CUDA_SOURCE artifact goes through the standalone-NVRTC pipeline.
        auto can_compile_cuda_source = artifact.format == tile::bridge::tirx::DeviceArtifact::Format::CUDA_SOURCE;
        if (ptx.empty() && !can_compile_cuda_source) {
            ptx.reserve(metadata.source.size() + 1u);
            auto first = reinterpret_cast<const std::byte *>(metadata.source.data());
            ptx.assign(first, first + metadata.source.size());
            ptx.push_back(std::byte{0});// cuModuleLoadData expects NUL-terminated PTX
            if (option.enable_cache || use_user_path) {
                write_tile_shader_ptx(
                    _io, name, shader_metadata, ptx,
                    use_user_path, option.enable_cache);
            }
        }

        auto compile_with_arch = [&](uint32_t arch) noexcept {
            luisa::vector<luisa::string> option_storage;
            option_storage.emplace_back(luisa::format("-arch=compute_{}", arch));
            option_storage.emplace_back("--std=c++17");
            option_storage.emplace_back("-default-device");
            option_storage.emplace_back("-restrict");
            option_storage.emplace_back("-extra-device-vectorization");
            option_storage.emplace_back("-w");
            option_storage.emplace_back("-ewp");
            if (option.enable_fast_math) { option_storage.emplace_back("--use_fast_math"); }
            if (option.enable_debug_info) { option_storage.emplace_back("-lineinfo"); }
            if (option.max_registers != 0u) {
                option_storage.emplace_back(luisa::format(
                    "-maxrregcount={}", std::clamp(option.max_registers, 0u, 255u)));
            }
            luisa::vector<const char *> nvrtc_options;
            nvrtc_options.reserve(option_storage.size());
            for (auto &s : option_storage) { nvrtc_options.emplace_back(s.c_str()); }

            luisa::filesystem::path src_dump_path;
            auto dump_source = option.enable_debug_info || std::getenv("LUISA_DUMP_SOURCE") != nullptr;
            luisa::string src_filename;
            if (dump_source) {
                luisa::span src_span{
                    reinterpret_cast<const std::byte *>(metadata.source.data()),
                    metadata.source.size()};
                auto src_name = luisa::format("cuda_tile_{:016x}.cu", checksum);
                if (use_user_path) {
                    src_dump_path = _io->write_shader_bytecode(src_name, src_span);
                } else if (option.enable_cache) {
                    src_dump_path = _io->write_shader_source(src_name, src_span);
                }
            }
            src_filename = luisa::string{src_dump_path.string()};
            Clock compile_clock;
            auto result = _compiler->compile(
                metadata.source, src_filename, nvrtc_options);
            if (!result.empty()) { LUISA_VERBOSE("CUDA Tile NVRTC took {} ms (PTX {} B).", compile_clock.toc(), result.size()); }
            return result;
        };

        if (ptx.empty() && can_compile_cuda_source) {
            ptx = compile_with_arch(_handle.compute_capability());
            if (!ptx.empty() && (option.enable_cache || use_user_path)) {
                write_tile_shader_ptx(
                    _io, name, shader_metadata, ptx,
                    use_user_path, option.enable_cache);
            }
        }
        if (ptx.empty()) {
            metadata.error = "CUDA Tile NVRTC compilation returned no PTX";
            return ShaderCreationInfo::make_invalid();
        }

        // Probe-load and construct inside the device context so cuModuleLoadData
        // has a current context. A failing module retries once with a patched
        // PTX version and, if that is not enough, falls back to a compute_60
        // recompile (mirroring the builtin-kernel path).
        Clock load_clock;
        luisa::vector<Usage> argument_usages;
        argument_usages.reserve(metadata.arguments.size());
        for (auto &arg : metadata.arguments) { argument_usages.emplace_back(arg.usage); }
        luisa::string load_failure;
        auto shader = with_handle([&]() noexcept -> CUDAShader * {
            CUresult load_error = CUDA_SUCCESS;
            auto load_result = probe_tile_ptx(artifact.entry, ptx, &load_error);
            if (load_result == TileLoadResult::UNSUPPORTED_PTX_VERSION) {
                CUDAShader::_patch_ptx_version(ptx);
                load_result = probe_tile_ptx(artifact.entry, ptx, &load_error);
            }
            if (load_result != TileLoadResult::OK && can_compile_cuda_source &&
                _handle.compute_capability() != 60u) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to load CUDA Tile PTX at the device compute capability; "
                    "recompiling for compute_60.");
                ptx = compile_with_arch(60u);
                if (ptx.empty()) {
                    load_failure = "CUDA Tile NVRTC fallback compilation returned no PTX";
                    return nullptr;
                }
                load_result = probe_tile_ptx(artifact.entry, ptx, &load_error);
                if (load_result == TileLoadResult::UNSUPPORTED_PTX_VERSION) {
                    CUDAShader::_patch_ptx_version(ptx);
                    load_result = probe_tile_ptx(artifact.entry, ptx, &load_error);
                }
                if (load_result == TileLoadResult::OK && (option.enable_cache || use_user_path)) {
                    write_tile_shader_ptx(
                        _io, name, shader_metadata, ptx,
                        use_user_path, option.enable_cache);
                }
            }
            if (load_result != TileLoadResult::OK) {
                const char *error_name = nullptr;
                const char *error_string = nullptr;
                cuGetErrorName(load_error, &error_name);
                cuGetErrorString(load_error, &error_string);
                load_failure = luisa::format(
                    "Failed to load CUDA Tile PTX module ({}: {})",
                    error_name ? error_name : "unknown",
                    error_string ? error_string : "unknown error");
                return nullptr;
            }
            return new_with_allocator<CUDAShaderTile>(
                this, std::move(ptx), artifact.entry, artifact.grid, block,
                std::move(artifact.buffer_arguments), std::move(argument_usages));
        });
        if (shader == nullptr) {
            metadata.error = std::move(load_failure);
            return ShaderCreationInfo::make_invalid();
        }
        auto load_ms = load_clock.toc();
        LUISA_VERBOSE("CUDA Tile module load took {} ms.", load_ms);
        ShaderCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->handle();
        info.block_size = block;
        return info;
#else
        metadata.error = "CUDA backend was built without the optional TIRx bridge";
        return ShaderCreationInfo::make_invalid();
#endif
    }
    if (tile_options.lowering != tile::Lowering::NATIVE || tile_options.tirx != nullptr) {
        metadata.error = "Invalid Tile lowering choice or TIRx options passed to native lowering";
        return ShaderCreationInfo::make_invalid();
    }
    metadata.error = "CUDA native Tile lowering is not implemented; use Lowering::TIRX";
    return ShaderCreationInfo::make_invalid();
}

}// namespace luisa::compute::cuda
