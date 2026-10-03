// CUDA Tile factory: the TIRx route produces SIMT CUDA source/NVPTX;
// an explicitly enabled native route emits CUDA Tile C++ and compiles Tile IR
// to cubin. Both use the static, direct-buffer CUDAShaderTile launch ABI.
//
// Mirror of src/backends/metal/tile/metal_tile.cpp for the CUDA backend.

#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>

#include <luisa/core/binary_io.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/hash.h>
#include <luisa/core/stl/string.h>
#include <luisa/tile/runtime.h>
#include <luisa/tile/collective_plan.h>
#include "cuda_tile_collective_cost.h"
#include "cuda_tile_partition_cost.h"

#include "cuda_tile.h"
#include "cuda_tile_codegen.h"
#include "cuda_tile_cub_scan.h"
#include "cuda_tile_streaming_scan.h"
#include "cuda_tile_partition_codegen.h"
#include "cuda_tile_ir.h"
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
constexpr auto cuda_tile_direct_buffer_abi = "cuda-tile-direct-buffers-v2-typed";

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

// Test-only seam (following the env-var pattern used elsewhere in the CUDA
// backend, e.g. LUISA_XIR_NORMALIZE_CFG): when set to "1", the first probe of
// a Tile module is reported as CUDA_ERROR_UNSUPPORTED_PTX_VERSION so the
// patched-PTX retry and cache write-back paths execute on drivers that would
// otherwise load the module directly. The subsequent (patched) probe always
// runs for real.
[[nodiscard]] bool force_first_tile_probe_unsupported() noexcept {
    auto value = luisa::get_environment_variable("LUISA_CUDA_TILE_FORCE_UNSUPPORTED_PTX");
    return value && luisa::string_view{*value} == "1";
}

}// namespace

ShaderCreationInfo CUDADevice::create_tile_kernel(const ShaderOption &option,
                                                  const tile::Function &kernel,
                                                  const tile::CompileOptions &tile_options,
                                                  tile::KernelMetadata &metadata) noexcept {
    metadata = {};
    // CUDA Tile reference-realization invariants (see also the TIRx mapper):
    //  * blocks are warp aligned: the GPU mapper rounds partial worker and
    //    elementwise domains up to the 32-thread warp and rounds an unaligned
    //    per-block cap down, so the post-codegen %32 check below is a
    //    consistency assertion, not the first line of defence;
    //  * a successful old-driver PTX patch is persisted back to the
    //    user/disk cache so cold processes load the compatible bytes instead of
    //    re-patching the stale cache entry on every launch;
    //  * a Format::PTX (NVPTX) artifact that fails to load cannot be recompiled
    //    by this backend; the final diagnostic names that constraint and
    //    recommends the CUDA_SOURCE ("cuda") TIRx target for this device.
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
            auto element = arg->type().scalar_type();
            auto element_bytes = tile::scalar_type_size(element);
            auto supported = element == tile::ScalarType::FLOAT32 || element == tile::ScalarType::FLOAT16 ||
                             element == tile::ScalarType::BFLOAT16 || element == tile::ScalarType::INT64 ||
                             element == tile::ScalarType::UINT32;
            if (!supported || !volume || *volume == 0u || *volume > INT32_MAX || *volume > SIZE_MAX / element_bytes) {
                return fail("CUDA TIRx Runtime requires nonempty, static, int32-addressable FP32/FP16/BF16/INT64/UINT32 buffers");
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
                tile::KernelArgument{element, *volume * element_bytes, usage});
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
            // CUDA blocks must be warp aligned. Reject a user-supplied exact
            // width before codegen so the mapper never receives an impossible
            // constraint; the value forwarded to the planner is then already a
            // multiple of 32.
            if (tile_options.threads_per_group % 32u != 0u) {
                return fail(luisa::format(
                    "CUDA Tile kernels require threads_per_group to be a "
                    "multiple of the 32-thread warp; got {}",
                    tile_options.threads_per_group));
            }
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
            // Preserve the public storage ABI in the shared CUDA sidecar.
            // BF16 uses its two-byte storage wrapper, not an FP32 stand-in.
            const Type *element = nullptr;
            switch (arg.element) {
                case tile::ScalarType::FLOAT32: element = Type::of<float>(); break;
                case tile::ScalarType::FLOAT16: element = Type::of<half>(); break;
                case tile::ScalarType::BFLOAT16: element = Type::of<tile::bfloat16>(); break;
                case tile::ScalarType::INT64: element = Type::of<int64_t>(); break;
                case tile::ScalarType::UINT32: element = Type::of<uint32_t>(); break;
                default: return fail("Unsupported CUDA TIRx sidecar argument type");
            }
            shader_metadata.argument_types.emplace_back(Type::buffer(element)->description());
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
            option_storage.emplace_back("--include-path=" LUISA_CUDA_TILE_TOOLKIT_INCLUDE_DIR);
            // CUDA 13 moved libcu++ into the toolkit's include/cccl directory.
            // Older SDKs retain the include-root layout and ignore this extra
            // search directory when it does not exist.
            option_storage.emplace_back("--include-path=" LUISA_CUDA_TILE_TOOLKIT_INCLUDE_DIR "/cccl");
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
            auto dump_source = option.enable_debug_info || luisa::get_environment_variable("LUISA_DUMP_SOURCE").has_value();
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
            src_filename = luisa::to_string(src_dump_path);
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
            auto load_result = TileLoadResult::OK;
            if (force_first_tile_probe_unsupported()) {
                load_error = CUDA_ERROR_UNSUPPORTED_PTX_VERSION;
                load_result = TileLoadResult::UNSUPPORTED_PTX_VERSION;
            } else {
                load_result = probe_tile_ptx(artifact.entry, ptx, &load_error);
            }
            if (load_result == TileLoadResult::UNSUPPORTED_PTX_VERSION) {
                auto pre_patch = ptx;
                CUDAShader::_patch_ptx_version(ptx);
                load_result = probe_tile_ptx(artifact.entry, ptx, &load_error);
                // Persist the successfully patched bytes (PTX and sidecar) back
                // to the same cache/user path used above, but only when the
                // patch actually changed the image. Without this a cold process
                // keeps loading the stale version and re-patches every launch.
                if (load_result == TileLoadResult::OK && ptx != pre_patch &&
                    (option.enable_cache || use_user_path)) {
                    write_tile_shader_ptx(
                        _io, name, shader_metadata, ptx,
                        use_user_path, option.enable_cache);
                }
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
                auto driver_error = luisa::format(
                    "{}: {}",
                    error_name ? error_name : "unknown",
                    error_string ? error_string : "unknown error");
                if (can_compile_cuda_source) {
                    load_failure = luisa::format(
                        "Failed to load CUDA Tile PTX module ({})",
                        driver_error);
                } else {
                    // Format::PTX artifacts are the NVPTX code generator's own
                    // output. There is no CUDA C source in this artifact that
                    // this backend could recompile for another architecture, so
                    // surface that constraint and point at the CUDA_SOURCE
                    // ("cuda") TIRx target instead of leaving only a raw error.
                    luisa::string_view requested_target;
                    constexpr luisa::string_view target_marker = ".target ";
                    auto target_pos = luisa::string_view{metadata.source}.find(target_marker);
                    if (target_pos != luisa::string_view::npos) {
                        auto line_end = luisa::string_view{metadata.source}.find('\n', target_pos);
                        auto directive = luisa::string_view{metadata.source}.substr(
                            target_pos, line_end == luisa::string_view::npos ?
                                            luisa::string_view::npos :
                                            line_end - target_pos);
                        if (!directive.empty() && directive.back() == '\r') {
                            directive = directive.substr(0u, directive.size() - 1u);
                        }
                        requested_target = directive;
                    }
                    load_failure = luisa::format(
                        "Failed to load CUDA Tile NVPTX PTX module ({})"
                        "{}. The artifact is NVPTX-generated PTX and cannot be "
                        "recompiled for a different CUDA architecture by the "
                        "Luisa CUDA backend; select the \"cuda\" (CUDA C source) "
                        "TIRx target for this device (compute capability sm_{}).",
                        driver_error,
                        requested_target.empty() ?
                            luisa::string_view{} :
                            luisa::string_view{"; the PTX requests "},
                        requested_target.empty() ?
                            luisa::string_view{} :
                            requested_target,
                        _handle.compute_capability());
                }
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
    auto opt_in = luisa::get_environment_variable("LUISA_CUDA_TILE_IR");
    if (!opt_in || luisa::string_view{*opt_in} != "1") {
        metadata.error = "CUDA native Tile IR is experimental; set LUISA_CUDA_TILE_IR=1 or use Lowering::TIRX";
        return ShaderCreationInfo::make_invalid();
    }
    auto fail = [&metadata](luisa::string_view message) noexcept {
        metadata.error = message;
        return ShaderCreationInfo::make_invalid();
    };
    if (!native_tile_ir_compiler_available()) {
        return fail("CUDA Tile IR is unavailable: configure CMake with CUDA 13.4+ Tile headers and tileiras");
    }
    if (tile_options.threads_per_group != 0u) {
        return fail("CUDA Tile IR does not support an explicit threads_per_group constraint");
    }
    if (option.max_registers != 0u) {
        return fail("CUDA Tile IR does not support an explicit max_registers constraint");
    }
    if (!option.name.empty()) {
        return fail("CUDA Tile IR does not support named shader archives");
    }
    if (!option.native_include.empty()) {
        return fail("CUDA Tile IR does not support native_include");
    }
    auto aligned16_option = luisa::get_environment_variable("LUISA_CUDA_TILE_IR_ALIGNED16");
    auto aligned16_requested = aligned16_option && luisa::string_view{*aligned16_option} == "1";
    auto worker_warps = 0u;
    if (auto workers = luisa::get_environment_variable("LUISA_CUDA_TILE_WORKER_WARPS")) {
        auto value = luisa::string_view{*workers};
        if (value == "4" || value == "8") {
            worker_warps = static_cast<uint32_t>(value.front() - '0');
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_WORKER_WARPS requires 0, 4 or 8");
        }
    }
    auto scan_chunk_extent = 0u;
    if (auto chunk = luisa::get_environment_variable("LUISA_CUDA_TILE_SCAN_CHUNK")) {
        auto value = luisa::string_view{*chunk};
        if (value == "1024") {
            scan_chunk_extent = 1024u;
        } else if (value == "2048") {
            scan_chunk_extent = 2048u;
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_SCAN_CHUNK requires 0, 1024 or 2048");
        }
    }
    auto independent_axis_extent = 0u;
    if (auto extent = luisa::get_environment_variable("LUISA_CUDA_TILE_INDEPENDENT_AXIS")) {
        auto value = luisa::string_view{*extent};
        if (value == "1" || value == "2" || value == "4") {
            independent_axis_extent = static_cast<uint32_t>(value.front() - '0');
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_INDEPENDENT_AXIS requires 0, 1, 2 or 4");
        }
    }
    auto streaming_scan_chunk = 0u;
    if (auto chunk = luisa::get_environment_variable("LUISA_CUDA_TILE_STREAMING_SCAN")) {
        auto value = luisa::string_view{*chunk};
        if (value == "1024") {
            streaming_scan_chunk = 1024u;
        } else if (value == "2048") {
            streaming_scan_chunk = 2048u;
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_STREAMING_SCAN requires 0, 1024 or 2048");
        }
    }
    if (streaming_scan_chunk != 0u && (scan_chunk_extent != 0u || independent_axis_extent != 0u)) {
        return fail("CUDA Tile streaming calibration cannot combine structural transforms");
    }
    if (scan_chunk_extent != 0u && independent_axis_extent != 0u) {
        return fail("CUDA Tile collective calibration permits only one structural transform at a time");
    }
    auto program_rows = 0u;
    if (auto rows = luisa::get_environment_variable("LUISA_CUDA_TILE_PROGRAM_ROWS")) {
        auto value = luisa::string_view{*rows};
        if (value == "1" || value == "2" || value == "4") {
            program_rows = static_cast<uint32_t>(value.front() - '0');
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_PROGRAM_ROWS requires 0, 1, 2 or 4");
        }
    }
    if (program_rows != 0u && (worker_warps != 0u || scan_chunk_extent != 0u ||
                               independent_axis_extent != 0u || streaming_scan_chunk != 0u || aligned16_requested)) {
        return fail("CUDA Tile program partition must be measured separately from explicit schedule hints");
    }
    auto collective_cost_requested = false;
    if (auto cost = luisa::get_environment_variable("LUISA_CUDA_TILE_COLLECTIVE_COST")) {
        auto value = luisa::string_view{*cost};
        if (value == "1") {
            collective_cost_requested = true;
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_COLLECTIVE_COST requires 0 or 1");
        }
    }
    if (collective_cost_requested && (worker_warps != 0u || scan_chunk_extent != 0u ||
                                      independent_axis_extent != 0u || streaming_scan_chunk != 0u || aligned16_requested || program_rows != 0u)) {
        return fail("CUDA Tile collective cost experiment must be measured separately from explicit schedule hints");
    }
    auto partition_cost_requested = false;
    if (auto cost = luisa::get_environment_variable("LUISA_CUDA_TILE_PARTITION_COST")) {
        auto value = luisa::string_view{*cost};
        if (value == "1") {
            partition_cost_requested = true;
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_PARTITION_COST requires 0 or 1");
        }
    }
    if (partition_cost_requested && (collective_cost_requested || worker_warps != 0u || scan_chunk_extent != 0u ||
                                     independent_axis_extent != 0u || streaming_scan_chunk != 0u || aligned16_requested || program_rows != 0u)) {
        return fail("CUDA Tile partition cost experiment must be measured separately from other costs and explicit schedule hints");
    }
    auto cub_scan_threads = 0u;
    if (auto threads = luisa::get_environment_variable("LUISA_CUDA_TILE_CUB_SCAN")) {
        auto value = luisa::string_view{*threads};
        if (value == "128") {
            cub_scan_threads = 128u;
        } else if (value == "256") {
            cub_scan_threads = 256u;
        } else if (value == "512") {
            cub_scan_threads = 512u;
        } else if (value == "1024") {
            cub_scan_threads = 1024u;
        } else if (value != "0") {
            return fail("LUISA_CUDA_TILE_CUB_SCAN requires 0, 128, 256, 512 or 1024");
        }
    }
    if (cub_scan_threads != 0u && (partition_cost_requested || collective_cost_requested || worker_warps != 0u ||
                                   scan_chunk_extent != 0u || independent_axis_extent != 0u || streaming_scan_chunk != 0u ||
                                   aligned16_requested || program_rows != 0u)) {
        return fail("CUDA Tile CUB scan must be measured separately from other experimental realizations");
    }
    auto collective_work = tile::analyze_collective_work(kernel);
    native_tile::CollectiveScheduleChoice collective_choice;
    if (collective_cost_requested) {
        int processors{}, warp_size{}, resident_threads{};
        if (cuDeviceGetAttribute(&processors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _handle.device()) == CUDA_SUCCESS &&
            cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _handle.device()) == CUDA_SUCCESS &&
            cuDeviceGetAttribute(&resident_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, _handle.device()) == CUDA_SUCCESS &&
            processors > 0 && warp_size > 0 && resident_threads > 0) {
            collective_choice = native_tile::choose_collective_schedule(collective_work,
                                                                        _handle.compute_capability(), static_cast<uint32_t>(processors), static_cast<uint32_t>(warp_size),
                                                                        static_cast<uint32_t>(resident_threads), _handle.driver_version(), CUDA_VERSION, option.enable_fast_math);
        }
        worker_warps = collective_choice.worker_warps;
    }
    native_tile::ProgramPartitionChoice partition_choice;
    if (partition_cost_requested) {
        int processors{}, warp_size{}, resident_threads{};
        if (cuDeviceGetAttribute(&processors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, _handle.device()) == CUDA_SUCCESS &&
            cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _handle.device()) == CUDA_SUCCESS &&
            cuDeviceGetAttribute(&resident_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, _handle.device()) == CUDA_SUCCESS &&
            processors > 0 && warp_size > 0 && resident_threads > 0) {
            auto rows1 = tile::plan_independent_collective(kernel, {.target_extent_per_program = 1u});
            auto rows2 = tile::plan_independent_collective(kernel, {.target_extent_per_program = 2u});
            partition_choice = native_tile::choose_program_partition(rows1, rows2,
                                                                     _handle.compute_capability(), static_cast<uint32_t>(processors), static_cast<uint32_t>(warp_size),
                                                                     static_cast<uint32_t>(resident_threads), _handle.driver_version(), CUDA_VERSION, option.enable_fast_math);
        }
        program_rows = partition_choice.target_rows;
    }
    auto artifact = native_tile::generate(kernel, option.enable_fast_math, aligned16_requested, worker_warps,
                                          _handle.compute_capability(), scan_chunk_extent, independent_axis_extent);
    if (!artifact.ok()) { return fail(artifact.error); }
    native_tile::append_streaming_scan(artifact, kernel, streaming_scan_chunk,
                                       worker_warps, _handle.compute_capability());
    native_tile::append_program_partition(artifact, kernel, program_rows);
    native_tile::CubScanArtifact cub_scan;
    if (cub_scan_threads != 0u) {
        if (option.enable_fast_math) {
            cub_scan.error = "requires-strict-math";
        } else {
            cub_scan = native_tile::generate_cub_scan(kernel, artifact, cub_scan_threads);
        }
    }
    auto block = make_uint3(1u, 1u, 1u);
    metadata.dispatch_size = make_uint3(artifact.grid[0u], artifact.grid[1u], artifact.grid[2u]);
    metadata.source = std::move(artifact.source);
    metadata.realization = "CUDA Tile C++ -> NVRTC Tile IR -> tileiras -> cubin; no cache; typed buffers; direct-buffer ABI; block=(1,1,1)";
    if (option.enable_fast_math) { metadata.realization += "; elementwise-fp32-approx-ftz-rsqrt-v2"; }
    if (worker_warps != 0u) { metadata.realization += luisa::format("; worker-warps-hint={}", worker_warps); }
    if (collective_cost_requested) {
        metadata.realization += luisa::format(
            "; collective-cost-profile={}; collective-cost-workers={}; collective-cost-status={}; collective-cost-reason={}",
            native_tile::kCollectiveScheduleProfile, collective_choice.worker_warps,
            collective_choice.status, collective_choice.reason);
        if (collective_choice.has_score) {
            metadata.realization += luisa::format("; collective-cost-log-score={:.17g}", collective_choice.log_score);
        }
    }
    if (streaming_scan_chunk != 0u) {
        metadata.realization += luisa::format("; streaming-scan-chunk={}", streaming_scan_chunk);
    }
    if (program_rows != 0u && !partition_cost_requested) {
        metadata.realization += luisa::format("; program-partition-rows={}", program_rows);
    }
    if (artifact.scan_chunk_extent != 0u) {
        metadata.realization += luisa::format("; scan-chunk={}; chunked-scans={}",
                                              artifact.scan_chunk_extent, artifact.chunked_scan_operations);
    }
    if (artifact.independent_axis_extent != 0u) {
        metadata.realization += luisa::format("; independent-axis-extent={}; partitioned-collectives={}",
                                              artifact.independent_axis_extent, artifact.partitioned_collective_operations);
    }
    if (collective_work.ok()) {
        // These are logical IR facts for schedule calibration, not measured
        // register counts, memory traffic or an occupancy guarantee.
        metadata.realization += luisa::format(
            "; collective-work-v1: programs={}, elementwork={}, read-bytes={}, write-bytes={}, tile-live-bytes={}, largest-tile={}",
            collective_work.programs, collective_work.elementwise_elements_per_program,
            collective_work.global_read_bytes_per_program, collective_work.global_write_bytes_per_program,
            collective_work.materialized_tile_peak_bytes, collective_work.largest_materialized_tile_elements);
        for (auto &&collective : collective_work.collectives) {
            metadata.realization += luisa::format("; collective=kind{}:width{}:independent{}",
                                                  static_cast<uint32_t>(collective.kind),
                                                  collective.contribution_extent, collective.independent_elements);
        }
    }
    if (aligned16_requested) {
        metadata.realization += luisa::format("; aligned16-requested; aligned16-buffer-mask={}; aligned16-partition-loads={}; {}",
                                              artifact.aligned16_buffer_mask,
                                              artifact.aligned16_partition_loads,
                                              artifact.aligned16_entry.empty() ? "aligned16-ineligible" : "host-selected-dual-entry-aligned16-v1");
    }
    // enable_cache is a hint. This experimental route deliberately does not
    // consult/write the PTX cache, a user archive, or an in-memory binary cache.
    luisa::vector<Usage> usages;
    luisa::vector<uint32_t> bindings;
    for (auto i = 0u; i < artifact.arguments.size(); i++) {
        const auto &argument = artifact.arguments[i];
        if (argument.minimum_size_bytes > std::numeric_limits<size_t>::max()) {
            return fail("CUDA Tile IR argument size exceeds the host Runtime ABI");
        }
        auto usage = argument.read ? (argument.written ? Usage::READ_WRITE : Usage::READ) :
                                     (argument.written ? Usage::WRITE : Usage::NONE);
        metadata.arguments.emplace_back(tile::KernelArgument{
            argument.element, static_cast<size_t>(argument.minimum_size_bytes), usage});
        usages.emplace_back(usage);
        bindings.emplace_back(i);
    }
    auto restore_original = [&](luisa::string_view diagnostic) noexcept {
        if (!artifact.partition_entry.empty()) {
            LUISA_ASSERT(artifact.streaming_scan_entry.empty() && artifact.partition_source_offset < metadata.source.size(),
                         "Program partition fallback lost the original source prefix.");
            metadata.source.resize(artifact.partition_source_offset);
            artifact.partition_entry.clear();
            artifact.partition_guard = {};
            artifact.partition_grid = {1u, 1u, 1u};
            artifact.partition_original_rows = 0u;
            artifact.partition_diagnostic.assign(diagnostic.data(), diagnostic.size());
        } else {
            LUISA_ASSERT(!artifact.streaming_scan_entry.empty() &&
                             artifact.streaming_scan_source_offset < metadata.source.size(),
                         "Streaming fallback lost the original source prefix.");
            metadata.source.resize(artifact.streaming_scan_source_offset);
            artifact.streaming_scan_entry.clear();
            artifact.streaming_scan_guard = {};
            artifact.streaming_scan_diagnostic.assign(diagnostic.data(), diagnostic.size());
        }
    };
    auto has_optional_entry = [&]() noexcept { return !artifact.streaming_scan_entry.empty() || !artifact.partition_entry.empty(); };
    auto compile_source = [&]() noexcept {
        return compile_native_tile_ir(context().runtime_directory(), metadata.source,
                                      _handle.compute_capability(), option.enable_debug_info);
    };
    auto binary = compile_source();
    if (!binary.valid() && has_optional_entry()) {
        auto diagnostic = luisa::format("optional {} compilation failed: {}",
                                        artifact.partition_entry.empty() ? "streaming" : "program partition", binary.error);
        restore_original(diagnostic);
        binary = compile_source();
    }
    if (!binary.valid()) { return fail(binary.error); }
    auto load_shader = [&]() noexcept -> CUDAShaderTile * {
        CUmodule module{};
        auto status = cuModuleLoadData(&module, binary.cubin.data());
        if (status != CUDA_SUCCESS) {
            metadata.error = luisa::format("CUDA Tile IR cubin load failed (CUDA {})", static_cast<int>(status));
            return nullptr;
        }
        CUfunction function{};
        status = cuModuleGetFunction(&function, module, artifact.entry.c_str());
        if (status != CUDA_SUCCESS) {
            auto cleanup = cuModuleUnload(module);
            metadata.error = luisa::format("CUDA Tile IR entry lookup failed (CUDA {}, module cleanup {})",
                                           static_cast<int>(status), static_cast<int>(cleanup));
            return nullptr;
        }
        CUfunction aligned16_function{};
        if (!artifact.aligned16_entry.empty()) {
            status = cuModuleGetFunction(&aligned16_function, module, artifact.aligned16_entry.c_str());
            if (status != CUDA_SUCCESS) {
                auto cleanup = cuModuleUnload(module);
                metadata.error = luisa::format("CUDA Tile IR aligned entry lookup failed (CUDA {}, module cleanup {})",
                                               static_cast<int>(status), static_cast<int>(cleanup));
                return nullptr;
            }
        }
        CUfunction streaming_scan_function{};
        if (!artifact.streaming_scan_entry.empty()) {
            status = cuModuleGetFunction(&streaming_scan_function, module, artifact.streaming_scan_entry.c_str());
            if (status != CUDA_SUCCESS) {
                auto cleanup = cuModuleUnload(module);
                metadata.error = luisa::format("CUDA Tile streaming entry lookup failed (CUDA {}, module cleanup {})",
                                               static_cast<int>(status), static_cast<int>(cleanup));
                return nullptr;
            }
        }
        CUfunction partition_function{};
        if (!artifact.partition_entry.empty()) {
            status = cuModuleGetFunction(&partition_function, module, artifact.partition_entry.c_str());
            if (status != CUDA_SUCCESS) {
                auto cleanup = cuModuleUnload(module);
                metadata.error = luisa::format("CUDA Tile partition entry lookup failed (CUDA {}, module cleanup {})",
                                               static_cast<int>(status), static_cast<int>(cleanup));
                return nullptr;
            }
        }
        return new_with_allocator<CUDAShaderTile>(module, function, std::move(artifact.entry),
                                                  artifact.grid, std::move(bindings), std::move(usages),
                                                  aligned16_function, artifact.aligned16_buffer_mask,
                                                  streaming_scan_function, artifact.streaming_scan_guard,
                                                  partition_function, artifact.partition_grid, artifact.partition_guard);
    };
    auto shader = with_handle(load_shader);
    if (shader == nullptr && has_optional_entry()) {
        auto diagnostic = luisa::format("optional {} module load failed: {}",
                                        artifact.partition_entry.empty() ? "streaming" : "program partition", metadata.error);
        restore_original(diagnostic);
        metadata.error.clear();
        binary = compile_source();
        if (!binary.valid()) { return fail(binary.error); }
        shader = with_handle(load_shader);
    }
    if (shader == nullptr) { return ShaderCreationInfo::make_invalid(); }
    if (cub_scan_threads != 0u) {
        // The original Tile shader is already compiled and loaded. Failure of
        // this independent ordinary-CUDA artifact never invalidates it.
        auto available = false;
        uint64_t compile_key = 0u;
        luisa::string cub_source_file;
        // Driver attributes describe the loaded function, not dynamic traffic
        // or Tile's physical worker count. These diagnostics never gate use.
        struct FunctionResources {
            std::array<int, 4u> values{-1, -1, -1, -1};
            luisa::string status{"not-queried"};
        };
        auto query_resources = [](CUfunction function) noexcept {
            constexpr std::array attributes{CU_FUNC_ATTRIBUTE_NUM_REGS, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                            CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK};
            FunctionResources result;
            result.status = "ok";
            for (auto i = size_t{0u}; i < attributes.size(); i++) {
                int value = -1;
                auto status = cuFuncGetAttribute(&value, attributes[i], function);
                if (status == CUDA_SUCCESS && value >= 0) {
                    result.values[i] = value;
                } else if (result.status == "ok") {
                    result.status = status == CUDA_SUCCESS ? "invalid-value" : luisa::format("cuda-{}", static_cast<int>(status));
                }
            }
            return result;
        };
        auto original_resources = with_handle([&]() noexcept { return query_resources(static_cast<CUfunction>(shader->handle())); });
        FunctionResources candidate_resources;
        int resident_capacity = -1;
        luisa::string capacity_status = "not-queried";
        if (cub_scan.ok()) {
            cub_scan.source = luisa::format("// Luisa CUB scan ABI v1; NVRTC {}; SDK {}\n{}",
                                            _compiler->nvrtc_version(), CUDA_VERSION, cub_scan.source);
            luisa::vector<luisa::string> storage{
                "--std=c++20", luisa::format("--gpu-architecture=compute_{}", _handle.compute_capability()),
                "--ftz=false", "--fmad=false", "--prec-div=true", "--prec-sqrt=true",
                "--device-as-default-execution-space",
                "--include-path=" LUISA_CUDA_TILE_TOOLKIT_INCLUDE_DIR,
                "--include-path=" LUISA_CUDA_TILE_TOOLKIT_INCLUDE_DIR "/cccl"};
            if (option.enable_debug_info) { storage.emplace_back("-lineinfo"); }
            luisa::vector<const char *> options;
            options.reserve(storage.size());
            for (auto &&value : storage) { options.emplace_back(value.c_str()); }
            compile_key = CUDACompiler::compute_hash(cub_scan.source, options);
            auto identity = luisa::format("cuda-tile-cub-scan-abi-v1-{:016x}", compile_key);
            luisa::string filename;
            if (option.enable_debug_info || luisa::get_environment_variable("LUISA_DUMP_SOURCE").has_value()) {
                auto name = luisa::format("cuda_tile_cub_scan_{:016x}.cu", compile_key);
                auto path = _io->write_shader_source(name, {reinterpret_cast<const std::byte *>(cub_scan.source.data()), cub_scan.source.size()});
                filename = luisa::to_string(path);
                // Realization is semicolon-delimited. A nonrepresentable dump
                // path remains local and must not masquerade as a valid receipt.
                if (filename.find_first_of(";\r\n") == luisa::string::npos) { cub_source_file = filename; }
            }
            // The ordinary compiler's process errors retain their existing
            // fatal contract. An ordinary compile error returns empty PTX.
            auto ptx = _compiler->compile(cub_scan.source, filename, options);
            if (ptx.empty()) {
                cub_scan.error = "nvrtc-compilation-failed";
            } else {
                if (ptx.back() != std::byte{0}) { ptx.emplace_back(std::byte{0}); }
                available = with_handle([&]() noexcept {
                    CUmodule module{};
                    auto status = cuModuleLoadData(&module, ptx.data());
                    if (status != CUDA_SUCCESS) {
                        cub_scan.error = luisa::format("module-load-failed-{}", static_cast<int>(status));
                        return false;
                    }
                    CUfunction function{};
                    status = cuModuleGetFunction(&function, module, cub_scan.entry.c_str());
                    if (status == CUDA_SUCCESS) {
                        candidate_resources = query_resources(function);
                        // This is ordinary CUDA with an explicit physical block.
                        // Do not apply this query to Tile's logical block=(1,1,1).
                        int capacity = -1;
                        auto capacity_result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                            &capacity, function, static_cast<int>(cub_scan_threads), 0u);
                        if (capacity_result == CUDA_SUCCESS && capacity >= 0) {
                            resident_capacity = capacity;
                            capacity_status = "ok";
                        } else {
                            capacity_status = capacity_result == CUDA_SUCCESS ? "invalid-value" : luisa::format("cuda-{}", static_cast<int>(capacity_result));
                        }
                    }
                    int maximum_threads{}, static_shared{}, device_threads{}, device_shared{};
                    if (status == CUDA_SUCCESS) { status = cuFuncGetAttribute(&maximum_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function); }
                    if (status == CUDA_SUCCESS) { status = cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function); }
                    if (status == CUDA_SUCCESS) { status = cuDeviceGetAttribute(&device_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, _handle.device()); }
                    if (status == CUDA_SUCCESS) { status = cuDeviceGetAttribute(&device_shared, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, _handle.device()); }
                    auto resources_ok = status == CUDA_SUCCESS && maximum_threads >= static_cast<int>(cub_scan_threads) &&
                                        device_threads >= static_cast<int>(cub_scan_threads) && static_shared >= 0 && static_shared <= device_shared;
                    if (resources_ok && shader->install_cub_scan(module, function, cub_scan.grid,
                                                                 make_uint3(cub_scan.block[0u], cub_scan.block[1u], cub_scan.block[2u]),
                                                                 cub_scan.guard, cub_scan.alignment_mask,
                                                                 std::move(cub_scan.source), std::move(identity))) { return true; }
                    auto cleanup = cuModuleUnload(module);
                    cub_scan.error = luisa::format("entry-or-resources-unavailable-{}-cleanup-{}", static_cast<int>(status), static_cast<int>(cleanup));
                    return false;
                });
            }
        }
        metadata.realization += luisa::format("; cub-scan-requested; cub-scan-threads={}; cub-scan-chunk={}; cub-scan-{}; cub-scan-compile-key={:016x}",
                                              cub_scan_threads, cub_scan_threads * 8u, available ? "available" : "unavailable", compile_key);
        if (!cub_source_file.empty()) { metadata.realization += luisa::format("; cub-scan-source-file={}", cub_source_file); }
        auto known_value = [](int value) noexcept { return value < 0 ? luisa::string{"unknown"} : luisa::format("{}", value); };
        auto append_resources = [&](luisa::string_view prefix, const FunctionResources &resources) noexcept {
            constexpr std::array names{"registers", "static-shared-bytes", "local-bytes", "max-threads"};
            for (auto i = size_t{0u}; i < names.size(); i++) {
                metadata.realization += luisa::format("; {}-{}={}", prefix, names[i], known_value(resources.values[i]));
            }
            metadata.realization += luisa::format("; {}-resource-status={}", prefix, resources.status);
        };
        // source-key is a 64-bit source identity, not a SHA or compiler key.
        // Candidate resources use the cub-scan-compile-key recorded above.
        metadata.realization += luisa::format("; cub-original-entry={}; cub-original-source-key={:016x}; cub-scan-resource-entry={}; cub-scan-resource-scope={}",
                                              shader->entry(), hash_value(metadata.source), cub_scan.entry.empty() ? "unknown" : cub_scan.entry,
                                              available ? "installed-entry" : candidate_resources.status == "not-queried" ? "not-queried" : "attempted-entry");
        append_resources("cub-original", original_resources);
        append_resources("cub-scan", candidate_resources);
        metadata.realization += luisa::format("; cub-scan-resident-cta-capacity={}; cub-scan-capacity-status={}; cub-scan-capacity-threads={}; cub-scan-capacity-dynamic-shared-bytes=0",
                                              known_value(resident_capacity), capacity_status, cub_scan_threads);
        if (available) {
            metadata.realization += luisa::format(
                "; cub-scan-input-slot={}; cub-scan-output-slot={}; cub-scan-input-bytes={}; cub-scan-output-bytes={}"
                "; cub-scan-alignment-mask={}; cub-scan-grid-x={}; cub-scan-block-x={}; host-selected-disjoint-aligned16-cub-scan-v1; cub-scan-nvrtc-lru",
                cub_scan.guard.input_slot, cub_scan.guard.output_slot, cub_scan.guard.input_bytes, cub_scan.guard.output_bytes,
                cub_scan.alignment_mask, cub_scan.grid[0u], cub_scan.block[0u]);
        } else {
            metadata.realization += luisa::format("; cub-scan-diagnostic={}", cub_scan.error);
        }
    }
    if (partition_cost_requested) {
        if (partition_choice.target_rows != 0u && artifact.partition_entry.empty()) {
            // A useful prediction is insufficient: unavailable variants retain
            // the original source/function/grid and report the original score.
            partition_choice.target_rows = 0u;
            partition_choice.selected_score = partition_choice.original_score;
            partition_choice.status = "retained";
            partition_choice.reason = "candidate-unavailable";
            program_rows = 0u;
        }
        auto selected_rows = partition_choice.target_rows == 0u ? partition_choice.original_rows : partition_choice.target_rows;
        metadata.realization += luisa::format(
            "; partition-cost-requested=1; partition-cost-profile={}; partition-cost-fit={}; partition-cost-status={}; partition-cost-reason={}"
            "; partition-cost-original-rows={}; partition-cost-selected-rows={}",
            native_tile::kPartitionCostProfile, native_tile::kPartitionCostFit, partition_choice.status, partition_choice.reason,
            partition_choice.original_rows, selected_rows);
        if (partition_choice.has_score) {
            metadata.realization += luisa::format("; partition-cost-original-score={:.17g}; partition-cost-selected-score={:.17g}",
                                                  partition_choice.original_score, partition_choice.selected_score);
        }
        if (program_rows != 0u) {
            metadata.realization += luisa::format("; program-partition-rows={}", program_rows);
        }
    }
    if (streaming_scan_chunk != 0u) {
        if (!artifact.streaming_scan_entry.empty()) {
            metadata.realization += "; streaming-scan-available; host-selected-disjoint-static-views-v1";
            metadata.realization += luisa::format(
                "; streaming-scan-input-slot={}; streaming-scan-output-slot={}; streaming-scan-input-bytes={}; streaming-scan-output-bytes={}",
                artifact.streaming_scan_guard.input_slot, artifact.streaming_scan_guard.output_slot,
                artifact.streaming_scan_guard.input_bytes, artifact.streaming_scan_guard.output_bytes);
        } else {
            metadata.realization += "; streaming-scan-ineligible-or-unavailable: ";
            metadata.realization += artifact.streaming_scan_diagnostic;
        }
    }
    if (program_rows != 0u) {
        if (!artifact.partition_entry.empty()) {
            metadata.realization += "; program-partition-available; host-selected-disjoint-program-rows-v1";
            metadata.realization += luisa::format(
                "; program-partition-input-slot={}; program-partition-output-slot={}; program-partition-input-bytes={}; program-partition-output-bytes={}"
                "; program-partition-original-rows={}; program-partition-grid-x={}; program-partition-original-grid-x={}",
                artifact.partition_guard.input_slot, artifact.partition_guard.output_slot,
                artifact.partition_guard.input_bytes, artifact.partition_guard.output_bytes,
                artifact.partition_original_rows, artifact.partition_grid[0u], artifact.grid[0u]);
        } else {
            metadata.realization += "; program-partition-ineligible-or-unavailable: ";
            metadata.realization += artifact.partition_diagnostic;
        }
    }
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = shader->handle();
    info.block_size = block;
    return info;
}

}// namespace luisa::compute::cuda
