#include <cstring>
#include <fstream>
#include <future>
#include <thread>
#include <cstdlib>

#include <nvtx3/nvToolsExtCuda.h>

#include <luisa/core/clock.h>
#include <luisa/core/binary_io.h>
#include <luisa/core/string_scratch.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/runtime/dispatch_buffer.h>

#ifdef LUISA_ENABLE_IR
#include <luisa/ir/ir2ast.h>
#include <luisa/ir/ast2ir.h>
#include <luisa/ir/transform.h>
#endif

#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_buffer.h"
#include "cuda_mesh.h"
#include "cuda_curve.h"
#include "cuda_procedural_primitive.h"
#include "cuda_motion_instance.h"
#include "cuda_accel.h"
#include "cuda_stream.h"
#include "cuda_event.h"
#include "cuda_codegen_ast.h"
#include "cuda_compiler.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"
#include "cuda_texture.h"
#include "cuda_shader_native.h"
#include "cuda_shader_optix.h"
#include "cuda_shader_metadata.h"
#include "optix_api.h"
#include "cuda_swapchain.h"
#include "cuda_builtin_embedded.h"

#include "extensions/cuda_dstorage.h"
#include "extensions/cuda_denoiser.h"
#include "extensions/cuda_pinned_memory.h"

#ifdef LUISA_COMPUTE_ENABLE_NVTT
#include "extensions/cuda_texture_compression.h"
#endif

#define LUISA_CUDA_KERNEL_DEBUG 1

#ifndef NDEBUG
#define LUISA_CUDA_DUMP_SOURCE 1
#else
static const bool LUISA_CUDA_DUMP_SOURCE = ([] {
    // read env LUISA_DUMP_SOURCE
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    if (env == nullptr) return false;
    return std::string_view{env} == "1";
})();
#endif

static const bool LUISA_CUDA_ENABLE_OPTIX_VALIDATION = ([] {
    // read env LUISA_OPTIX_VALIDATION
    auto env = std::getenv("LUISA_OPTIX_VALIDATION");
    if (env == nullptr) return false;
    return std::string_view{env} == "1";
})();

namespace luisa::compute::cuda {

[[nodiscard]] static auto cuda_array_format(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt: return CU_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::R8UInt: [[fallthrough]];
        case PixelFormat::R8UNorm: return CU_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::RG8SInt: return CU_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::RG8UInt: [[fallthrough]];
        case PixelFormat::RG8UNorm: return CU_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::RGBA8SInt: return CU_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::RGBA8UInt: [[fallthrough]];
        case PixelFormat::RGBA8UNorm: return CU_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::R16SInt: return CU_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::R16UInt: [[fallthrough]];
        case PixelFormat::R16UNorm: return CU_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::RG16SInt: return CU_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::RG16UInt: [[fallthrough]];
        case PixelFormat::RG16UNorm: return CU_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::RGBA16SInt: return CU_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::RGBA16UInt: [[fallthrough]];
        case PixelFormat::RGBA16UNorm: return CU_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::R32SInt: return CU_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::R32UInt: return CU_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::RG32SInt: return CU_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::RG32UInt: return CU_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::RGBA32SInt: return CU_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::RGBA32UInt: return CU_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::R16F: return CU_AD_FORMAT_HALF;
        case PixelFormat::RG16F: return CU_AD_FORMAT_HALF;
        case PixelFormat::RGBA16F: return CU_AD_FORMAT_HALF;
        case PixelFormat::R32F: return CU_AD_FORMAT_FLOAT;
        case PixelFormat::RG32F: return CU_AD_FORMAT_FLOAT;
        case PixelFormat::RGBA32F: return CU_AD_FORMAT_FLOAT;
        case PixelFormat::BC4UNorm: return CU_AD_FORMAT_BC4_UNORM;
        case PixelFormat::BC5UNorm: return CU_AD_FORMAT_BC5_UNORM;
        case PixelFormat::BC6HUF16: return CU_AD_FORMAT_BC6H_UF16;
        case PixelFormat::BC7UNorm: return CU_AD_FORMAT_BC7_UNORM;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid pixel format 0x{:02x}.",
                              luisa::to_underlying(format));
}

CUDADevice::CUDADevice(Context &&ctx,
                       size_t device_id,
                       const BinaryIO *io) noexcept
    : DeviceInterface{std::move(ctx)}, _handle{device_id}, _io{io} {
    // provide a default binary IO
    if (_io == nullptr) {
        _default_io = luisa::make_unique<DefaultBinaryIO>(context());
        _io = _default_io.get();
    }
    _compiler = luisa::make_unique<CUDACompiler>(this);
    auto sm_option = luisa::format("-arch=compute_{}", handle().compute_capability());
    std::array options{sm_option.c_str(),
                       "--std=c++17",
                       "--use_fast_math",
                       "-default-device",
                       "-restrict",
                       "-extra-device-vectorization",
                       "-dw",
                       "-w",
                       "-ewp"};
    luisa::string builtin_kernel_src{
        luisa_cuda_builtin_cuda_builtin_kernels,
        sizeof(luisa_cuda_builtin_cuda_builtin_kernels)};
    auto builtin_kernel_ptx = _compiler->compile(builtin_kernel_src, "luisa_builtin.cu", options);

    // prepare default shaders
    with_handle([&] {
        auto error = cuModuleLoadData(&_builtin_kernel_module, builtin_kernel_ptx.data());
        auto retry_with_ptx_version_patched = [this](auto error, auto &builtin_kernel_ptx) noexcept -> CUresult {
            const char *error_string = nullptr;
            cuGetErrorString(error, &error_string);
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load built-in kernels: {} "
                "Re-trying with patched PTX version...",
                error_string ? error_string : "Unknown error");
            CUDAShader::_patch_ptx_version(builtin_kernel_ptx);
            return cuModuleLoadData(&_builtin_kernel_module, builtin_kernel_ptx.data());
        };
        if (error == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
            error = retry_with_ptx_version_patched(error, builtin_kernel_ptx);
        }
        if (error != CUDA_SUCCESS) {
            const char *error_string = nullptr;
            cuGetErrorString(error, &error_string);
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load built-in kernels: {}. "
                "Re-trying with lower compute capability...",
                error_string ? error_string : "Unknown error");
            _handle.force_compute_capability(60u);
            options.front() = "-arch=compute_60";
            builtin_kernel_ptx = _compiler->compile(builtin_kernel_src, "luisa_builtin.cu", options);
            error = cuModuleLoadData(&_builtin_kernel_module, builtin_kernel_ptx.data());
            if (error == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
                error = retry_with_ptx_version_patched(error, builtin_kernel_ptx);
            }
        }
        LUISA_CHECK_CUDA(error);
        LUISA_CHECK_CUDA(cuModuleGetFunction(
            &_accel_update_function, _builtin_kernel_module,
            "update_accel"));
        LUISA_CHECK_CUDA(cuModuleGetFunction(
            &_bindless_array_update_function, _builtin_kernel_module,
            "update_bindless_array"));
        LUISA_CHECK_CUDA(cuModuleGetFunction(
            &_instance_handle_update_function, _builtin_kernel_module,
            "update_accel_instance_handles"));
    });

    // load cudadevrt
#ifdef LUISA_PLATFORM_WINDOWS
    auto device_runtime_lib_path = context().runtime_directory() / "cudadevrt.lib";
#else
    auto device_runtime_lib_path = context().runtime_directory() / "libcudadevrt.a";
#endif
    std::ifstream devrt_file{device_runtime_lib_path, std::ios::binary};
    if (!devrt_file.is_open()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load CUDA device runtime library '{}'. "
            "Indirect kernel dispatch will not be available.",
            device_runtime_lib_path.string());
    } else {
        _cudadevrt_library = luisa::string{std::istreambuf_iterator<char>{devrt_file},
                                           std::istreambuf_iterator<char>{}};
    }
    // test if the device runtime library is recognized by the driver
    if (!_cudadevrt_library.empty()) {
        // TODO: this check can consume hundreds of milliseconds! Is there a better way?
        // generate some non-sense kernel source with dynamic parallelism
        auto dummy_kernel_src = R"(__global__ void a() {} __global__ void b() { a<<<1024,32>>>(); })";
        auto dummy_ptx = _compiler->compile(dummy_kernel_src, "dummy_devrt_check.cu", options);
        void *output_cubin = nullptr;
        size_t output_cubin_size = 0u;
        with_handle([&] {
            CUlinkState link_state{};
            LUISA_CHECK_CUDA(cuLinkCreate(0u, nullptr, nullptr, &link_state));
            auto report_failure_and_clear_library = [&](auto phase) {
                LUISA_WARNING_WITH_LOCATION(
                    "Found CUDA device runtime library '{}', but the driver does not "
                    "recognize it when {}. Indirect kernel dispatch will not be available.",
                    device_runtime_lib_path.string(), phase);
                _cudadevrt_library.clear();
            };
            if (cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                              dummy_ptx.data(), dummy_ptx.size(),
                              "dummy_kernel", 0u, nullptr, nullptr) != CUDA_SUCCESS) {
                report_failure_and_clear_library("adding the kernel PTX");
            } else if (cuLinkAddData(link_state, CU_JIT_INPUT_LIBRARY,
                                     _cudadevrt_library.data(), _cudadevrt_library.size(),
                                     "cudadevrt", 0u, nullptr, nullptr) != CUDA_SUCCESS) {
                report_failure_and_clear_library("adding the device runtime library");
            } else if (cuLinkComplete(link_state, &output_cubin, &output_cubin_size) != CUDA_SUCCESS) {
                report_failure_and_clear_library("completing linking");
            }
            LUISA_CHECK_CUDA(cuLinkDestroy(link_state));
        });
    }
    if (!_cudadevrt_library.empty()) {
        LUISA_VERBOSE("Successfully loaded CUDA device runtime library. "
                      "Indirect dispatch feature is available.");
    }
}

CUDADevice::~CUDADevice() noexcept {
    with_handle([this] {
        LUISA_CHECK_CUDA(cuCtxSynchronize());
        LUISA_CHECK_CUDA(cuModuleUnload(_builtin_kernel_module));
    });
}

CUDAEventManager *CUDADevice::event_manager() const noexcept {
    std::scoped_lock lock{_event_manager_mutex};
    if (_event_manager == nullptr) [[unlikely]] {
        _event_manager = luisa::make_unique<CUDAEventManager>(handle().uuid());
    }
    return _event_manager.get();
}

BufferCreationInfo CUDADevice::create_buffer(const Type *element,
                                             size_t elem_count,
                                             void *external_memory) noexcept {
    BufferCreationInfo info{};
    elem_count = std::max<size_t>(elem_count, 1u);
    if (element == Type::of<IndirectKernelDispatch>()) {
        LUISA_ASSERT(external_memory == nullptr,
                     "Indirect dispatch buffer cannot "
                     "be created from external memory.");
        auto buffer = with_handle([elem_count] {
            return new_with_allocator<CUDAIndirectDispatchBuffer>(elem_count);
        });
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = reinterpret_cast<void *>(buffer->device_address());
        info.element_stride = sizeof(CUDAIndirectDispatchBuffer::Dispatch);
        info.total_size_bytes = buffer->size_bytes();
    } else if (element == Type::of<void>()) {
        info.element_stride = 1;
        info.total_size_bytes = elem_count;
        auto buffer = with_handle([size = info.total_size_bytes, em = external_memory] {
            return em ? new_with_allocator<CUDABuffer>(reinterpret_cast<CUdeviceptr>(em), size) :
                        new_with_allocator<CUDABuffer>(size);
        });
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = reinterpret_cast<void *>(buffer->device_address());
    } else {
        info.element_stride = CUDACompiler::type_size(element);
        info.total_size_bytes = info.element_stride * elem_count;
        auto buffer = with_handle([size = info.total_size_bytes, em = external_memory] {
            return em ? new_with_allocator<CUDABuffer>(reinterpret_cast<CUdeviceptr>(em), size) :
                        new_with_allocator<CUDABuffer>(size);
        });
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = reinterpret_cast<void *>(buffer->device_address());
    }
    return info;
}

BufferCreationInfo CUDADevice::create_buffer(const ir::CArc<ir::Type> *element,
                                             size_t elem_count,
                                             void *external_memory) noexcept {
#ifdef LUISA_ENABLE_IR
    auto type = IR2AST::get_type(element->get());
    return create_buffer(type, elem_count, external_memory);
#else
    LUISA_ERROR_WITH_LOCATION("CUDA device does not support creating shader from IR types.");
#endif
}

void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
    with_handle([buffer = reinterpret_cast<CUDABufferBase *>(handle)] {
        delete_with_allocator(buffer);
    });
}

ResourceCreationInfo CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept {
    auto p = with_handle([=] {
        auto array_format = cuda_array_format(format);
        auto channels = pixel_format_channel_count(format);
        CUDA_ARRAY3D_DESCRIPTOR array_desc{};
        array_desc.Width = width;
        array_desc.Height = height;
        array_desc.Depth = dimension == 2u ? 0u : depth;
        array_desc.Format = array_format;
        array_desc.NumChannels = channels;
        if (!is_block_compressed(format)) {
            array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
        }
        auto array_handle = [&] {
            if (mipmap_levels == 1u) {
                CUarray handle{nullptr};
                LUISA_CHECK_CUDA(cuArray3DCreate(&handle, &array_desc));
                return reinterpret_cast<uint64_t>(handle);
            }
            CUmipmappedArray handle{nullptr};
            LUISA_CHECK_CUDA(cuMipmappedArrayCreate(&handle, &array_desc, mipmap_levels));
            return reinterpret_cast<uint64_t>(handle);
        }();
        return new_with_allocator<CUDATexture>(
            array_handle, make_uint3(width, height, depth),
            format, mipmap_levels);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = reinterpret_cast<void *>(p->handle())};
}

void CUDADevice::destroy_texture(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDATexture *>(handle)] {
        delete_with_allocator(array);
    });
}

ResourceCreationInfo CUDADevice::create_bindless_array(size_t size) noexcept {
    auto p = with_handle([size] { return new_with_allocator<CUDABindlessArray>(size); });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = reinterpret_cast<void *>(p->handle())};
}

void CUDADevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(handle)] {
        delete_with_allocator(array);
    });
}

ResourceCreationInfo CUDADevice::create_stream(StreamTag stream_tag) noexcept {
#ifndef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    if (stream_tag == StreamTag::GRAPHICS) {
        LUISA_WARNING_WITH_LOCATION("Swapchains are not enabled on CUDA backend, "
                                    "Graphics streams might not work properly.");
    }
#endif
    auto p = with_handle([&] { return new_with_allocator<CUDAStream>(this); });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = p->handle()};
}

void CUDADevice::destroy_stream(uint64_t handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
        delete_with_allocator(stream);
    });
}

void CUDADevice::synchronize_stream(uint64_t stream_handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle)] {
        stream->synchronize();
    });
}

void CUDADevice::set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept {
    reinterpret_cast<CUDAStream *>(stream_handle)->set_log_callback(callback);
}

void CUDADevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    if (!list.empty()) {
        with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle),
                     list = std::move(list)]() mutable noexcept {
            stream->dispatch(std::move(list));
        });
    }
}

SwapchainCreationInfo CUDADevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    auto chain = with_handle([&] {
        return new_with_allocator<CUDASwapchain>(this, option);
    });
    SwapchainCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(chain);
    info.native_handle = chain->native_handle();
    info.storage = chain->pixel_storage();
    return info;
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

void CUDADevice::destroy_swap_chain(uint64_t handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_handle([chain = reinterpret_cast<CUDASwapchain *>(handle)] {
        delete_with_allocator(chain);
    });
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

void CUDADevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle),
                 chain = reinterpret_cast<CUDASwapchain *>(swapchain_handle),
                 image = reinterpret_cast<CUDATexture *>(image_handle)] {
        chain->present(stream, image);
    });
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

[[nodiscard]] inline luisa::optional<CUDAShaderMetadata>
parse_shader_metadata(luisa::string_view data,
                      luisa::string_view name) noexcept {
    if (data.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to parse shader metadata for '{}': "
            "PTX source is empty.",
            name);
        return luisa::nullopt;
    }
    constexpr luisa::string_view metadata_prefix = "// METADATA: ";
    if (!data.starts_with(metadata_prefix)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to parse shader metadata for '{}': "
            "PTX source does not start with metadata prefix.",
            name);
        return luisa::nullopt;
    }
    auto m = data.substr(metadata_prefix.size());
    auto metadata = deserialize_cuda_shader_metadata(m);
    if (!metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to parse shader metadata for '{}': "
            "invalid metadata string '{}'.",
            name, m);
        return luisa::nullopt;
    }
    return metadata;
}

template<bool allow_update_expected_metadata>
[[nodiscard]] inline luisa::vector<std::byte> load_shader_ptx(
    BinaryStream *metadata_stream,
    BinaryStream *ptx_stream,
    luisa::string_view name,
    bool warn_not_found,
    std::conditional_t<allow_update_expected_metadata,
                       CUDAShaderMetadata,
                       const CUDAShaderMetadata> &expected_metadata) noexcept {

    // check if the stream is valid
    if (ptx_stream == nullptr || ptx_stream->length() == 0u ||
        metadata_stream == nullptr || metadata_stream->length() == 0u) {
        if (warn_not_found) {
            LUISA_WARNING_WITH_LOCATION(
                "Shader '{}' is not found in cache. "
                "This may be caused by a mismatch between the shader source and the cached binary. "
                "The shader will be recompiled.",
                name);
        } else {
            LUISA_VERBOSE("Shader '{}' is not found in cache. "
                          "The shader will be recompiled.",
                          name);
        }
        return {};
    }
    // read metadata string from stream
    luisa::string meta_data;
    meta_data.resize(metadata_stream->length());
    metadata_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(meta_data.data()),
        meta_data.size() * sizeof(char)});

    // read ptx
    luisa::vector<std::byte> ptx_data;
    ptx_data.resize(ptx_stream->length());
    ptx_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(ptx_data.data()),
        ptx_data.size() * sizeof(char)});

    // parse metadata
    auto metadata = parse_shader_metadata(meta_data, name);
    if (!metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Shader '{}' is found in cache, but its metadata is invalid. "
            "This may be caused by a mismatch between the shader source and the cached binary. "
            "The shader will be recompiled.",
            name);
        return {};
    }
    // update the empty fields in metadata
    if constexpr (allow_update_expected_metadata) {
        if (expected_metadata.checksum == 0u) { expected_metadata.checksum = metadata->checksum; }
        if (expected_metadata.curve_bases.none()) { expected_metadata.curve_bases = metadata->curve_bases; }
        if (expected_metadata.kind == CUDAShaderMetadata::Kind::UNKNOWN) { expected_metadata.kind = metadata->kind; }
        expected_metadata.enable_debug = metadata->enable_debug;
        expected_metadata.requires_trace_closest = metadata->requires_trace_closest;
        expected_metadata.requires_trace_any = metadata->requires_trace_any;
        expected_metadata.requires_ray_query = metadata->requires_ray_query;
        expected_metadata.requires_printing = metadata->requires_printing;
        expected_metadata.requires_motion_blur = metadata->requires_motion_blur;
        if (expected_metadata.max_register_count == 0u) { expected_metadata.max_register_count = metadata->max_register_count; }
        if (all(expected_metadata.block_size == 0u)) { expected_metadata.block_size = metadata->block_size; }
        if (expected_metadata.argument_types.empty()) { expected_metadata.argument_types = metadata->argument_types; }
        if (expected_metadata.argument_usages.empty()) { expected_metadata.argument_usages = metadata->argument_usages; }
        if (expected_metadata.format_types.empty()) { expected_metadata.format_types = metadata->format_types; }
    }
    // examine the metadata
    if (*metadata != expected_metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Shader '{}' is found in cache, but its metadata '{}' do not match the expected '{}'. "
            "This may be caused by a mismatch between the shader source and the cached binary. "
            "The shader will be recompiled.",
            name, serialize_cuda_shader_metadata(*metadata),
            serialize_cuda_shader_metadata(expected_metadata));
        return {};
    }
    // return the ptx string
    return ptx_data;
}

ShaderCreationInfo CUDADevice::_create_shader(luisa::string name,
                                              const string &source, const ShaderOption &option,
                                              luisa::span<const char *const> nvrtc_options,
                                              const CUDAShaderMetadata &expected_metadata,
                                              luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept {

    // generate a default name if not specified
    auto uses_user_path = !name.empty();
    if (!uses_user_path) { name = luisa::format("kernel_{:016x}.ptx",
                                                expected_metadata.checksum); }
    if (!name.ends_with(".ptx") &&
        !name.ends_with(".PTX")) { name.append(".ptx"); }
    auto metadata_name = luisa::format("{}.metadata", name);

    // try disk cache
    auto ptx = [&] {
        luisa::unique_ptr<BinaryStream> ptx_stream;
        luisa::unique_ptr<BinaryStream> metadata_stream;
        if (uses_user_path) {
            ptx_stream = _io->read_shader_bytecode(name);
            metadata_stream = _io->read_shader_bytecode(metadata_name);
        } else if (option.enable_cache) {
            ptx_stream = _io->read_shader_cache(name);
            metadata_stream = _io->read_shader_cache(metadata_name);
        }
        return load_shader_ptx<false>(
            metadata_stream.get(), ptx_stream.get(),
            name, false, expected_metadata);
    }();

    // compile if not found in cache
    if (ptx.empty()) {
        luisa::filesystem::path src_dump_path;
        if (option.enable_debug_info || LUISA_CUDA_DUMP_SOURCE) {
            luisa::span src_data{reinterpret_cast<const std::byte *>(source.data()), source.size()};
            auto src_name = luisa::format("{}.cu", name);
            if (uses_user_path) {
                src_dump_path = _io->write_shader_bytecode(src_name, src_data);
            } else if (option.enable_cache) {
                src_dump_path = _io->write_shader_source(src_name, src_data);
            }
        }
        luisa::string src_filename{src_dump_path.string()};
        ptx = _compiler->compile(source, src_filename, nvrtc_options, &expected_metadata);
        if (!ptx.empty()) {
            luisa::span ptx_data{reinterpret_cast<const std::byte *>(ptx.data()), ptx.size()};
            auto metadata = luisa::format("// METADATA: {}\n\n", serialize_cuda_shader_metadata(expected_metadata));
            luisa::span metadata_data{reinterpret_cast<const std::byte *>(metadata.data()), metadata.size()};
            if (uses_user_path) {
                static_cast<void>(_io->write_shader_bytecode(name, ptx_data));
                static_cast<void>(_io->write_shader_bytecode(metadata_name, metadata_data));
            } else if (option.enable_cache) {
                static_cast<void>(_io->write_shader_cache(name, ptx_data));
                static_cast<void>(_io->write_shader_cache(metadata_name, metadata_data));
            }
        }
    }

    if (option.compile_only) {// no shader object should be created
        return ShaderCreationInfo::make_invalid();
    }

    // create the shader object
    auto p = with_handle([&]() noexcept -> CUDAShader * {
        if (expected_metadata.kind == CUDAShaderMetadata::Kind::RAY_TRACING) {
            return new_with_allocator<CUDAShaderOptiX>(
                handle().optix_context(), std::move(ptx),
                "__raygen__main", expected_metadata, std::move(bound_arguments));
        }
        return new_with_allocator<CUDAShaderNative>(
            this, std::move(ptx), "kernel_main",
            expected_metadata, std::move(bound_arguments));
    });
#ifndef NDEBUG
    p->set_name(std::move(name));
#endif
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p->handle();
    info.block_size = expected_metadata.block_size;
    return info;
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, Function kernel) noexcept {

    if (kernel.propagated_builtin_callables().test(CallOp::BACKWARD)) {
#ifdef LUISA_ENABLE_IR
        auto ir = AST2IR::build_kernel(kernel);
        ir->get()->module.flags |= ir::ModuleFlags_REQUIRES_REV_AD_TRANSFORM;
        transform_ir_kernel_module_auto(ir->get());
        return create_shader(option, ir->get());
#else
        LUISA_ERROR_WITH_LOCATION("Please enable IR for autodiff support");
#endif
    }

    // codegen

    Clock clk;
    StringScratch scratch;
    CUDACodegenAST codegen{scratch, !_cudadevrt_library.empty()};
    codegen.emit(kernel, _compiler->device_library(), option.native_include);
    LUISA_VERBOSE("Generated CUDA source in {} ms.", clk.toc());

    // process bound arguments
    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments;
    bound_arguments.reserve(kernel.bound_arguments().size());
    for (auto &&arg : kernel.bound_arguments()) {
        luisa::visit(
            [&bound_arguments]<typename T>(T binding) noexcept {
                ShaderDispatchCommand::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BUFFER;
                    argument.buffer.handle = binding.handle;
                    argument.buffer.offset = binding.offset;
                    argument.buffer.size = binding.size;
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::TEXTURE;
                    argument.texture.handle = binding.handle;
                    argument.texture.level = binding.level;
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array.handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::ACCEL;
                    argument.accel.handle = binding.handle;
                } else {
                    LUISA_ERROR_WITH_LOCATION("Unsupported binding type.");
                }
                bound_arguments.emplace_back(argument);
            },
            arg);
    }

    // NVRTC nvrtc_options
    auto sm_option = luisa::format("-arch=compute_{}", _handle.compute_capability());
    auto nvrtc_version_option = luisa::format("-DLC_NVRTC_VERSION={}", _compiler->nvrtc_version());
    auto optix_version_option = luisa::format("-DLC_OPTIX_VERSION={}", optix::VERSION);
    luisa::vector<const char *> nvrtc_options{
        sm_option.c_str(),
        nvrtc_version_option.c_str(),
        optix_version_option.c_str(),
        "--std=c++17",
        "-default-device",
        "-restrict",
        "-extra-device-vectorization",
        "-dw",
        "-w",
        "-ewp",
#if !defined(NDEBUG) && LUISA_CUDA_KERNEL_DEBUG
        "-DLUISA_DEBUG=1",
#endif
    };

    luisa::string max_reg_opt;
    if (option.max_registers != 0u) {
        max_reg_opt = luisa::format(
            "-maxrregcount={}",
            std::clamp(option.max_registers, 0u, 255u));
        nvrtc_options.emplace_back(max_reg_opt.c_str());
    }

    // generate time trace for optimization the compilation time
    if (option.time_trace &&
        _compiler->nvrtc_version() >= 120100 &&
        _handle.driver_version() >= 12010) {
        nvrtc_options.emplace_back("-time=-");
    }

    // multithreaded compilation
    // TODO: the flag seems not working any more
    // if (_compiler->nvrtc_version() >= 120100 &&
    //     _handle.driver_version() >= 12030) {
    //     nvrtc_options.emplace_back("-split-compile=0");
    // }

    if (option.enable_debug_info) {
        nvrtc_options.emplace_back("-lineinfo");
#if defined(NDEBUG) || !LUISA_CUDA_KERNEL_DEBUG
        nvrtc_options.emplace_back("-DLUISA_DEBUG=1");
#endif
    } else if (LUISA_CUDA_DUMP_SOURCE) {
        nvrtc_options.emplace_back("-lineinfo");
    }
    if (option.enable_fast_math) {
        nvrtc_options.emplace_back("-use_fast_math");
    }

    // FIXME: OptiX IR disabled due to many internal compiler errors
    // TODO: use OptiX IR for ray tracing shaders
    //  if (kernel.requires_raytracing()) {
    //      nvrtc_options.emplace_back("--optix-ir");
    //  }

    // compute hash
    auto src_hash = _compiler->compute_hash(scratch.string(), nvrtc_options);

    // create metadata
    CUDAShaderMetadata metadata{
        .checksum = src_hash,
        .curve_bases = kernel.required_curve_bases(),
        .kind = kernel.requires_raytracing() ?
                    CUDAShaderMetadata::Kind::RAY_TRACING :
                    CUDAShaderMetadata::Kind::COMPUTE,
        .enable_debug = option.enable_debug_info,
        .requires_trace_closest = kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_CLOSEST) ||
                                  kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR),
        .requires_trace_any = kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_ANY) ||
                              kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR),
        .requires_ray_query = kernel.propagated_builtin_callables().uses_ray_query(),
        .requires_printing = kernel.requires_printing(),
        .requires_motion_blur = kernel.requires_motion_blur(),
        .max_register_count = std::clamp(option.max_registers, 0u, 255u),
        .block_size = kernel.block_size(),
        .argument_types = [kernel] {
            luisa::vector<luisa::string> types;
            types.reserve(kernel.arguments().size());
            std::transform(kernel.arguments().cbegin(), kernel.arguments().cend(), std::back_inserter(types),
                           [](auto &&arg) noexcept { return luisa::string{arg.type()->description()}; });
            return types; }(),
        .argument_usages = [kernel] {
            luisa::vector<Usage> usages;
            usages.reserve(kernel.arguments().size());
            std::transform(kernel.arguments().cbegin(), kernel.arguments().cend(), std::back_inserter(usages),
                           [kernel](auto &&arg) noexcept { return kernel.variable_usage(arg.uid()); });
            return usages; }(),
        .format_types = [fmt = codegen.print_formats()] {
            luisa::vector<std::pair<luisa::string, luisa::string>> t;
            t.reserve(fmt.size());
            for (auto &&[name, type] : fmt) {
                t.emplace_back(name, type->description());
            }
            return t; }(),
    };
    return _create_shader(option.name, scratch.string(),
                          option, nvrtc_options,
                          metadata, std::move(bound_arguments));
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
#ifdef LUISA_ENABLE_IR
    Clock clk;
    auto function = IR2AST::build(kernel);
    LUISA_VERBOSE("IR2AST done in {} ms.", clk.toc());
    return create_shader(option, function->function());
#else
    LUISA_ERROR_WITH_LOCATION("CUDA device does not support creating shader from IR types.");
    return {};
#endif
}

ShaderCreationInfo CUDADevice::load_shader(luisa::string_view name_in,
                                           luisa::span<const Type *const> arg_types) noexcept {

    luisa::string name{name_in};
    if (!name.ends_with(".ptx") &&
        !name.ends_with(".PTX")) { name.append(".ptx"); }
    auto metadata_name = luisa::format("{}.metadata", name);

    // prepare (incomplete) metadata
    CUDAShaderMetadata metadata{
        .checksum = 0u,
        .kind = CUDAShaderMetadata::Kind::UNKNOWN,
        .max_register_count = 0u,
        .block_size = make_uint3(0u),
        .argument_types = [arg_types] {
            luisa::vector<luisa::string> types;
            types.reserve(arg_types.size());
            std::transform(arg_types.cbegin(), arg_types.cend(), std::back_inserter(types),
                           [](auto &&arg) noexcept { return luisa::string{arg->description()}; });
            return types; }(),
    };

    // load ptx
    auto ptx = [&] {
        auto metadata_stream = _io->read_shader_bytecode(metadata_name);
        auto ptx_stream = _io->read_shader_bytecode(name);
        return load_shader_ptx<true>(metadata_stream.get(), ptx_stream.get(), name, true, metadata);
    }();
    if (ptx.empty()) {
        LUISA_WARNING_WITH_LOCATION("Failed to load shader bytecode from {}.", name);
        return ShaderCreationInfo::make_invalid();
    }

    // check argument count
    if (metadata.argument_types.size() != arg_types.size()) {
        LUISA_WARNING_WITH_LOCATION("Argument count mismatch when loading shader {}.", name);
        return ShaderCreationInfo::make_invalid();
    }

    // create shader
    auto p = with_handle([&]() noexcept -> CUDAShader * {
        if (metadata.kind == CUDAShaderMetadata::Kind::RAY_TRACING) {
            return new_with_allocator<CUDAShaderOptiX>(
                handle().optix_context(), std::move(ptx),
                "__raygen__main", metadata);
        }
        return new_with_allocator<CUDAShaderNative>(
            this, std::move(ptx), "kernel_main", metadata);
    });
#ifndef NDEBUG
    p->set_name(std::move(name));
#endif
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p->handle();
    info.block_size = metadata.block_size;
    return info;
}

Usage CUDADevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return reinterpret_cast<const CUDAShader *>(handle)->argument_usage(index);
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
    with_handle([shader = reinterpret_cast<CUDAShader *>(handle)] {
        delete_with_allocator(shader);
    });
}

ResourceCreationInfo CUDADevice::create_event() noexcept {
    auto event_handle = with_handle([m = this->event_manager()] {
        return m->create();
    });
    return {.handle = reinterpret_cast<uint64_t>(event_handle),
            .native_handle = event_handle->handle()};
}

void CUDADevice::destroy_event(uint64_t handle) noexcept {
    with_handle([m = this->event_manager(), handle] {
        auto event = reinterpret_cast<CUDAEvent *>(handle);
        m->destroy(event);
    });
}

void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUDAEvent *>(handle);
        auto stream = reinterpret_cast<CUDAStream *>(stream_handle);
        stream->signal(event, value);
    });
}

void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUDAEvent *>(handle);
        auto stream = reinterpret_cast<CUDAStream *>(stream_handle);
        stream->wait(event, value);
    });
}

bool CUDADevice::is_event_completed(uint64_t handle, uint64_t value) const noexcept {
    auto event = reinterpret_cast<CUDAEvent *>(handle);
    return event->is_completed(value);
}

void CUDADevice::synchronize_event(uint64_t handle, uint64_t value) noexcept {
    auto event = reinterpret_cast<CUDAEvent *>(handle);
    event->synchronize(value);
}

ResourceCreationInfo CUDADevice::create_mesh(const AccelOption &option) noexcept {
    auto mesh_handle = with_handle([&option] {
        return new_with_allocator<CUDAMesh>(option);
    });
    return {.handle = reinterpret_cast<uint64_t>(mesh_handle),
            .native_handle = const_cast<optix::TraversableHandle *>(mesh_handle->pointer_to_handle())};
}

void CUDADevice::destroy_mesh(uint64_t handle) noexcept {
    with_handle([=] {
        auto mesh = reinterpret_cast<CUDAMesh *>(handle);
        delete_with_allocator(mesh);
    });
}

ResourceCreationInfo CUDADevice::create_curve(const AccelOption &option) noexcept {
    auto curve_handle = with_handle([&option] {
        return new_with_allocator<CUDACurve>(option);
    });
    return {.handle = reinterpret_cast<uint64_t>(curve_handle),
            .native_handle = const_cast<optix::TraversableHandle *>(curve_handle->pointer_to_handle())};
}

void CUDADevice::destroy_curve(uint64_t handle) noexcept {
    with_handle([=] {
        auto curve = reinterpret_cast<CUDACurve *>(handle);
        delete_with_allocator(curve);
    });
}

ResourceCreationInfo CUDADevice::create_procedural_primitive(const AccelOption &option) noexcept {
    auto primitive_handle = with_handle([&option] {
        return new_with_allocator<CUDAProceduralPrimitive>(option);
    });
    return {.handle = reinterpret_cast<uint64_t>(primitive_handle),
            .native_handle = const_cast<optix::TraversableHandle *>(primitive_handle->pointer_to_handle())};
}

void CUDADevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    with_handle([=] {
        auto primitive = reinterpret_cast<CUDAProceduralPrimitive *>(handle);
        delete_with_allocator(primitive);
    });
}

ResourceCreationInfo CUDADevice::create_motion_instance(const AccelMotionOption &option) noexcept {
    auto instance_handle = with_handle([this, &option] {
        return new_with_allocator<CUDAMotionInstance>(this, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(instance_handle),
            .native_handle = const_cast<optix::TraversableHandle *>(instance_handle->pointer_to_handle())};
}

void CUDADevice::destroy_motion_instance(uint64_t handle) noexcept {
    with_handle([=] {
        auto instance = reinterpret_cast<CUDAMotionInstance *>(handle);
        delete_with_allocator(instance);
    });
}

ResourceCreationInfo CUDADevice::create_accel(const AccelOption &option) noexcept {
    auto accel_handle = with_handle([&option] {
        return new_with_allocator<CUDAAccel>(option);
    });
    return {.handle = reinterpret_cast<uint64_t>(accel_handle),
            .native_handle = const_cast<optix::TraversableHandle *>(accel_handle->pointer_to_handle())};
}

void CUDADevice::destroy_accel(uint64_t handle) noexcept {
    with_handle([accel = reinterpret_cast<CUDAAccel *>(handle)] {
        delete_with_allocator(accel);
    });
}

string CUDADevice::query(luisa::string_view property) noexcept {
    if (property == "device_name") {
        return "cuda";
    }
    LUISA_WARNING_WITH_LOCATION("Unknown device property '{}'.", property);
    return {};
}

DeviceExtension *CUDADevice::extension(luisa::string_view name) noexcept {

#define LUISA_COMPUTE_CREATE_CUDA_EXTENSION(ext, v)                         \
    if (name == ext##Ext::name) {                                           \
        std::scoped_lock lock{_ext_mutex};                                  \
        if (v == nullptr) { v = luisa::make_unique<CUDA##ext##Ext>(this); } \
        return v.get();                                                     \
    }
#if LUISA_BACKEND_ENABLE_OIDN
    LUISA_COMPUTE_CREATE_CUDA_EXTENSION(Denoiser, _denoiser_ext)
#endif
    LUISA_COMPUTE_CREATE_CUDA_EXTENSION(DStorage, _dstorage_ext)
    LUISA_COMPUTE_CREATE_CUDA_EXTENSION(PinnedMemory, _pinned_memory_ext)
#ifdef LUISA_COMPUTE_ENABLE_NVTT
    LUISA_COMPUTE_CREATE_CUDA_EXTENSION(TexCompress, _tex_comp_ext)
#endif
#undef LUISA_COMPUTE_CREATE_CUDA_EXTENSION

    LUISA_WARNING_WITH_LOCATION("Unknown device extension '{}'.", name);
    return nullptr;
}

namespace detail {

static constexpr auto required_cuda_version_major = 11;
static constexpr auto required_cuda_version_minor = 7;
static constexpr auto required_cuda_version = required_cuda_version_major * 1000 + required_cuda_version_minor * 10;

static void initialize() {
    // global init
    static std::once_flag flag;
    std::call_once(flag, [] {
        // CUDA
        LUISA_CHECK_CUDA(cuInit(0));
        // check driver version
        auto driver_version = 0;
        LUISA_CHECK_CUDA(cuDriverGetVersion(&driver_version));
        auto driver_version_major = driver_version / 1000;
        auto driver_version_minor = (driver_version % 1000) / 10;
        LUISA_ASSERT(driver_version >= required_cuda_version,
                     "CUDA driver version {}.{} is too old (>= {}.{} required). "
                     "Please update your driver.",
                     driver_version_major, driver_version_minor,
                     required_cuda_version_major, required_cuda_version_minor);
        LUISA_VERBOSE("Successfully initialized CUDA "
                      "backend with driver version {}.{}.",
                      driver_version_major, driver_version_minor);
    });
}

}// namespace detail

CUDADevice::Handle::Handle(size_t index) noexcept {

    detail::initialize();

    // cuda
    auto driver_version = 0;
    LUISA_CHECK_CUDA(cuDriverGetVersion(&driver_version));
    _driver_version = driver_version;

    auto device_count = 0;
    LUISA_CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        LUISA_ERROR_WITH_LOCATION("No available device found for CUDA backend.");
    }
    if (index == std::numeric_limits<size_t>::max()) { index = 0; }
    if (index >= device_count) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid device index {} (device count = {}). Limiting to {}.",
            index, device_count, device_count - 1);
        index = device_count - 1;
    }
    _device_index = index;
    LUISA_CHECK_CUDA(cuDeviceGet(&_device, index));
    auto compute_cap_major = 0;
    auto compute_cap_minor = 0;
    LUISA_CHECK_CUDA(cuDeviceGetUuid(&_uuid, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));

    auto can_map_host_memory = 0;
    auto unified_addressing = 0;
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&can_map_host_memory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, _device));
    LUISA_VERBOSE("Device {} can map host memory: {}", index, can_map_host_memory != 0);
    LUISA_VERBOSE("Device {} supports unified addressing: {}", index, unified_addressing != 0);

    auto format_uuid = [](auto uuid) noexcept {
        luisa::string result;
        result.reserve(36u);
        auto count = 0u;
        for (auto c : uuid.bytes) {
            if (count == 4u || count == 6u || count == 8u || count == 10u) {
                result.append("-");
            }
            result.append(fmt::format("{:02x}", static_cast<uint>(c) & 0xffu));
            count++;
        }
        return result;
    };

    LUISA_INFO("Created CUDA device at index {}: {} "
               "(driver = {}, capability = {}.{}, uuid = {}).",
               index, name(), driver_version,
               compute_cap_major, compute_cap_minor,
               format_uuid(_uuid));
    _compute_capability = 10u * compute_cap_major + compute_cap_minor;
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_context, _device));
}

CUDADevice::Handle::~Handle() noexcept {
    if (_optix_context) {
        LUISA_CHECK_OPTIX(optix::api().deviceContextDestroy(_optix_context));
    }
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRelease(_device));
    LUISA_VERBOSE("Destroyed CUDA device: {}.", name());
}

std::string_view CUDADevice::Handle::name() const noexcept {
    static constexpr auto device_name_length = 1024u;
    static thread_local char device_name[device_name_length];
    LUISA_CHECK_CUDA(cuDeviceGetName(device_name, device_name_length, _device));
    return device_name;
}

optix::DeviceContext CUDADevice::Handle::optix_context() const noexcept {
    std::scoped_lock lock{_mutex};
    if (_optix_context == nullptr) [[unlikely]] {
        optix::DeviceContextOptions optix_options{};
        optix_options.logCallbackLevel = 4u;
        if (LUISA_CUDA_ENABLE_OPTIX_VALIDATION) {
            LUISA_WARNING("OptiX validation is enabled. This may cause significant performance degradation.");
            // Disable due to too much overhead
            optix_options.validationMode = optix::DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        }
        optix_options.logCallbackFunction = [](uint level, const char *tag, const char *message, void *) noexcept {
            auto log = luisa::format("Logs from OptiX ({}): {}", tag, message);
            if (level >= 4) {
                LUISA_VERBOSE("{}", log);
            } else [[unlikely]] {
                LUISA_WARNING("{}", log);
            }
        };
        LUISA_CHECK_OPTIX(optix::api().deviceContextCreate(
            _context, &optix_options, &_optix_context));
    }
    return _optix_context;
}

void CUDADevice::set_name(luisa::compute::Resource::Tag resource_tag,
                          uint64_t resource_handle,
                          luisa::string_view name) noexcept {
    with_handle([tag = resource_tag,
                 handle = resource_handle,
                 name = luisa::string{name}]() mutable noexcept {
        switch (tag) {
            case Resource::Tag::BUFFER:
                reinterpret_cast<CUDABufferBase *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::TEXTURE:
                reinterpret_cast<CUDATexture *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::BINDLESS_ARRAY:
                reinterpret_cast<CUDABindlessArray *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::MESH: [[fallthrough]];
            case Resource::Tag::CURVE: [[fallthrough]];
            case Resource::Tag::PROCEDURAL_PRIMITIVE: [[fallthrough]];
            case Resource::Tag::MOTION_INSTANCE:
                reinterpret_cast<CUDAPrimitiveBase *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::ACCEL:
                reinterpret_cast<CUDAAccel *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::STREAM:
                reinterpret_cast<CUDAStream *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::EVENT:
                break;
            case Resource::Tag::SHADER:
                reinterpret_cast<CUDAShader *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::RASTER_SHADER: break;
            case Resource::Tag::SWAP_CHAIN:
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
                reinterpret_cast<CUDASwapchain *>(handle)->set_name(std::move(name));
#endif
                break;
            case Resource::Tag::DEPTH_BUFFER: break;
            case Resource::Tag::DSTORAGE_FILE: break;
            case Resource::Tag::DSTORAGE_PINNED_MEMORY: break;
            case Resource::Tag::SPARSE_BUFFER: break;
            case Resource::Tag::SPARSE_TEXTURE: break;
            default: break;
        }
    });
}

}// namespace luisa::compute::cuda

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    auto device_id = 0ull;
    auto binary_io = static_cast<const luisa::BinaryIO *>(nullptr);
    if (config != nullptr) {
        device_id = config->device_index;
        binary_io = config->binary_io;
        LUISA_ASSERT(!config->headless,
                     "Headless mode is not implemented yet for CUDA backend.");
    }
    return luisa::new_with_allocator<luisa::compute::cuda::CUDADevice>(
        std::move(ctx), device_id, binary_io);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    auto p = static_cast<luisa::compute::cuda::CUDADevice *>(device);
    // auto p = dynamic_cast<luisa::compute::cuda::CUDADevice *>(device);
    // LUISA_ASSERT(p != nullptr, "Deleting a null CUDA device.");
    luisa::delete_with_allocator(p);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    auto device_count = 0;
    luisa::compute::cuda::detail::initialize();
    LUISA_CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count > 0) {
        names.reserve(device_count);
        for (auto i = 0; i < device_count; i++) {
            CUdevice device{};
            LUISA_CHECK_CUDA(cuDeviceGet(&device, i));
            static thread_local char name[1024];
            LUISA_CHECK_CUDA(cuDeviceGetName(name, 1024, device));
            names.emplace_back(name);
        }
    }
}
