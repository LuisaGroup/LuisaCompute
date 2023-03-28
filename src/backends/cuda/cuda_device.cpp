//
// Created by Mike on 7/28/2021.
//

#include <cstring>
#include <fstream>
#include <future>
#include <thread>

#include <core/clock.h>
#include <core/binary_io.h>

#include <runtime/rhi/sampler.h>
#include <runtime/bindless_array.h>
#include <backends/common/string_scratch.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_codegen_ast.h>
#include <backends/cuda/cuda_compiler.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_shader_native.h>
#include <backends/cuda/cuda_shader_optix.h>
#include <backends/cuda/optix_api.h>
#include <backends/cuda/cuda_swapchain.h>

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
    : DeviceInterface{std::move(ctx)},
      _handle{device_id}, _io{io} {

    // provide a default binary IO
    if (_io == nullptr) {
        _default_io = luisa::make_unique<DefaultBinaryIO>(context(), "cuda");
        _io = _default_io.get();
    }
    _compiler = luisa::make_unique<CUDACompiler>(this);

    luisa::string builtin_kernel_src;
    {
        auto builtin_kernel_stream = _io->read_internal_shader("cuda_builtin_kernels.cu");
        builtin_kernel_src.resize(builtin_kernel_stream->length());
        builtin_kernel_stream->read(luisa::span{
            reinterpret_cast<std::byte *>(builtin_kernel_src.data()),
            builtin_kernel_src.size()});
    }

    auto sm_option = luisa::format("-arch=sm_{}", handle().compute_capability());
    std::array options{sm_option.c_str(),
                       "--std=c++17",
                       "--use_fast_math",
                       "-default-device",
                       "-restrict",
                       "-extra-device-vectorization",
                       "-dw",
                       "-w",
                       "-ewp"};
    auto builtin_kernel_ptx = _compiler->compile(builtin_kernel_src, options);

    // prepare default shaders
    with_handle([this, &builtin_kernel_ptx] {
        LUISA_CHECK_CUDA(cuCtxResetPersistingL2Cache());
        LUISA_CHECK_CUDA(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1));
        LUISA_CHECK_CUDA(cuModuleLoadData(
            &_builtin_kernel_module, builtin_kernel_ptx.data()));
        LUISA_CHECK_CUDA(cuModuleGetFunction(
            &_accel_update_function, _builtin_kernel_module,
            "update_accel"));
        LUISA_CHECK_CUDA(cuModuleGetFunction(
            &_bindless_array_update_function, _builtin_kernel_module,
            "update_bindless_array"));
    });
}

CUDADevice::~CUDADevice() noexcept {
    with_handle([this] {
        LUISA_CHECK_CUDA(cuCtxSynchronize());
        LUISA_CHECK_CUDA(cuModuleUnload(_builtin_kernel_module));
    });
}

BufferCreationInfo CUDADevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    BufferCreationInfo info{};
    info.element_stride = CUDACompiler::type_size(element);
    info.total_size_bytes = info.element_stride * elem_count;
    auto buffer = with_handle([size = info.total_size_bytes] {
        return new_with_allocator<CUDABuffer>(size);
    });
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer;
    return info;
}

BufferCreationInfo CUDADevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
    LUISA_ERROR_WITH_LOCATION("CUDA device does not support creating buffer from IR types.");
}

void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
    with_handle([buffer = reinterpret_cast<CUDABuffer *>(handle)] {
        delete_with_allocator(buffer);
    });
}

ResourceCreationInfo CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
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
        return new_with_allocator<CUDAMipmapArray>(array_handle, format, mipmap_levels);
    });
    return {.handle = reinterpret_cast<uint64_t>(p), .native_handle = p};
}

void CUDADevice::destroy_texture(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDAMipmapArray *>(handle)] {
        delete_with_allocator(array);
    });
}

ResourceCreationInfo CUDADevice::create_bindless_array(size_t size) noexcept {
    auto p = with_handle([size] { return new_with_allocator<CUDABindlessArray>(size); });
    return {.handle = reinterpret_cast<uint64_t>(p), .native_handle = p};
}

void CUDADevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(handle)] {
        delete_with_allocator(array);
    });
}

ResourceCreationInfo CUDADevice::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    LUISA_ERROR_WITH_LOCATION("Depth buffers are not supported on CUDA.");
}

void CUDADevice::destroy_depth_buffer(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Depth buffers are not supported on CUDA.");
}

ResourceCreationInfo CUDADevice::create_stream(StreamTag stream_tag) noexcept {
#ifndef LC_CUDA_ENABLE_VULKAN_SWAPCHAIN
    if (stream_tag == StreamTag::GRAPHICS) {
        LUISA_WARNING_WITH_LOCATION("Swapchains are not enabled on CUDA backend, "
                                    "Graphics streams might not work properly.");
    }
#endif
    auto p = with_handle([&] { return new_with_allocator<CUDAStream>(this); });
    return {.handle = reinterpret_cast<uint64_t>(p), .native_handle = p};
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

void CUDADevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    if (!list.empty()) {
        with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle),
                     commands = list.steal_commands(),
                     callbacks = list.steal_callbacks()]() mutable noexcept {
            CUDACommandEncoder encoder{stream};
            for (auto &cmd : commands) { cmd->accept(encoder); }
            encoder.commit(std::move(callbacks));
        });
    }
}

SwapChainCreationInfo CUDADevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle,
                                                    uint width, uint height,
                                                    bool allow_hdr, bool vsync,
                                                    uint back_buffer_size) noexcept {
#ifdef LC_CUDA_ENABLE_VULKAN_SWAPCHAIN
    auto chain = with_handle([&] {
        return new_with_allocator<CUDASwapchain>(
            this, window_handle, width, height,
            allow_hdr, vsync, back_buffer_size);
    });
    SwapChainCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(chain);
    info.native_handle = chain;
    info.storage = chain->pixel_storage();
    return info;
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

void CUDADevice::destroy_swap_chain(uint64_t handle) noexcept {
#ifdef LC_CUDA_ENABLE_VULKAN_SWAPCHAIN
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
#ifdef LC_CUDA_ENABLE_VULKAN_SWAPCHAIN
    with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle),
                 chain = reinterpret_cast<CUDASwapchain *>(swapchain_handle),
                 image = reinterpret_cast<CUDAMipmapArray *>(image_handle)] {
        chain->present(stream, image);
    });
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

ShaderCreationInfo CUDADevice::_create_shader(const string &source,
                                              ShaderOption option,
                                              uint3 block_size,
                                              bool is_raytracing,
                                              luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept {

    // NVRTC options
    auto sm_option = luisa::format("-arch=compute_{}", _handle.compute_capability());
    auto nvrtc_version_option = luisa::format("-DLC_NVRTC_VERSION={}", _compiler->nvrtc_version());
    auto optix_version_option = luisa::format("-DLC_OPTIX_VERSION={}", optix::VERSION);
    luisa::vector<const char *> options{sm_option.c_str(),
                                        nvrtc_version_option.c_str(),
                                        optix_version_option.c_str(),
                                        "--std=c++17",
                                        "-default-device",
                                        "-restrict",
                                        "-extra-device-vectorization",
                                        "-dw",
                                        "-w",
                                        "-ewp"};
    if (option.enable_debug_info) {
        options.emplace_back("-line-info");
        options.emplace_back("-DLC_DEBUG");
    }
    if (option.enable_fast_math) {
        options.emplace_back("-use_fast_math");
    }

    // compute hash
    auto src_hash = _compiler->compute_hash(source, options);

    // generate a default name if not specified
    auto uses_user_path = !option.name.empty();
    if (!uses_user_path) { option.name = luisa::format("kernel_{:016x}.ptx", src_hash); }
    if (!option.name.ends_with(".ptx") &&
        !option.name.ends_with(".PTX")) { option.name.append(".ptx"); }

    // try disk cache
    auto ptx = [&] {
        luisa::unique_ptr<BinaryStream> ptx_stream;
        if (uses_user_path) {
            ptx_stream = _io->read_shader_bytecode(option.name);
        } else if (option.enable_cache) {
            ptx_stream = _io->read_shader_cache(option.name);
        }
        luisa::string ptx_data;
        if (ptx_stream != nullptr) {
            ptx_data.resize(ptx_stream->length());
            ptx_stream->read(luisa::span{
                reinterpret_cast<std::byte *>(ptx_data.data()),
                ptx_data.size() * sizeof(char)});
            auto expected_header = _compiler->checksum_header(src_hash);
            if (!ptx_data.starts_with(expected_header)) {
                LUISA_WARNING_WITH_LOCATION(
                    "Shader '{}' cache checksum mismatches.",
                    option.name);
                ptx_data.clear();
            }
        }
        return ptx_data;
    }();

    // compile if not found in cache
    if (ptx.empty()) {
        ptx = _compiler->compile(source, options, src_hash);
        luisa::span ptx_data{reinterpret_cast<const std::byte *>(ptx.data()), ptx.size()};
        if (uses_user_path) {
            _io->write_shader_bytecode(option.name, ptx_data);
        } else if (option.enable_cache) {
            _io->write_shader_cache(option.name, ptx_data);
        }
    }

    if (option.compile_only) {// no shader object should be created
        return ShaderCreationInfo::make_invalid();
    }

    // create the shader object
    auto p = with_handle([&]() noexcept -> CUDAShader * {
        if (is_raytracing) {
            return new_with_allocator<CUDAShaderOptiX>(
                this, ptx.data(), ptx.size(), "__raygen__main",
                option.enable_debug_info, std::move(bound_arguments));
        }
        return new_with_allocator<CUDAShaderNative>(
            ptx.data(), ptx.size(), "kernel_main",
            block_size, std::move(bound_arguments));
    });

    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p;
    info.block_size = block_size;
    return info;
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    Clock clk;
    StringScratch scratch;
    CUDACodegenAST codegen{scratch};
    codegen.emit(kernel);
    LUISA_INFO("Generated CUDA source in {} ms.", clk.toc());
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
    return _create_shader(scratch.string(), option,
                          kernel.block_size(),
                          kernel.requires_raytracing(),
                          std::move(bound_arguments));
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    LUISA_ERROR_WITH_LOCATION("TODO");
}

ShaderCreationInfo CUDADevice::load_shader(luisa::string_view name,
                                           luisa::span<const Type *const> arg_types) noexcept {
    // TODO: verify arg_types
    auto ptx_stream = _io->read_shader_bytecode(name);
    if (ptx_stream == nullptr) { return ShaderCreationInfo::make_invalid(); }
    luisa::string ptx;
    ptx.resize(ptx_stream->length());
    ptx_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(ptx.data()),
        ptx.size() * sizeof(char)});
    auto is_raytracing = ptx.find("__raygen__main") != luisa::string::npos;
    // create the shader object
    //    auto p = with_handle([&]() noexcept -> CUDAShader * {
    //        if (is_raytracing) {
    //            return new_with_allocator<CUDAShaderOptiX>(
    //                this, ptx.data(), ptx.size(), "__raygen__main");
    //        }
    //        return new_with_allocator<CUDAShaderNative>(
    //            ptx.data(), ptx.size(), "kernel_main", block_size);
    //    });
    //
    //    ShaderCreationInfo info{};
    //    info.handle = reinterpret_cast<uint64_t>(p);
    //    info.native_handle = p;
    //    info.block_size = block_size;
    //    return info;
    LUISA_ERROR_WITH_LOCATION("TODO");
}

Usage CUDADevice::shader_arg_usage(uint64_t handle, size_t index) noexcept {
    // TODO
    LUISA_ASSERT(false, "Un-Implemented");
    return Usage::NONE;
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
    with_handle([shader = reinterpret_cast<CUDAShader *>(handle)] {
        delete_with_allocator(shader);
    });
}

ResourceCreationInfo CUDADevice::create_event() noexcept {
    auto event_handle = with_handle([] {
        CUevent event = nullptr;
        LUISA_CHECK_CUDA(cuEventCreate(
            &event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
        return event;
    });
    return {.handle = reinterpret_cast<uint64_t>(event_handle), .native_handle = event_handle};
}

void CUDADevice::destroy_event(uint64_t handle) noexcept {
    with_handle([event = reinterpret_cast<CUevent>(handle)] {
        LUISA_CHECK_CUDA(cuEventDestroy(event));
    });
}

void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUevent>(handle);
        auto stream = reinterpret_cast<CUDAStream *>(stream_handle);
        stream->signal(event);
    });
}

void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUevent>(handle);
        auto stream = reinterpret_cast<CUDAStream *>(stream_handle);
        stream->wait(event);
    });
}

void CUDADevice::synchronize_event(uint64_t handle) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUevent>(handle);
        LUISA_CHECK_CUDA(cuEventSynchronize(event));
    });
}

ResourceCreationInfo CUDADevice::create_mesh(const AccelOption &option) noexcept {
    auto mesh_handle = with_handle([&option] {
        return new_with_allocator<CUDAMesh>(option);
    });
    return {.handle = reinterpret_cast<uint64_t>(mesh_handle),
            .native_handle = mesh_handle};
}

void CUDADevice::destroy_mesh(uint64_t handle) noexcept {
    with_handle([=] {
        auto mesh = reinterpret_cast<CUDAMesh *>(handle);
        delete_with_allocator(mesh);
    });
}

ResourceCreationInfo CUDADevice::create_procedural_primitive(const AccelOption &option) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

void CUDADevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

ResourceCreationInfo CUDADevice::create_accel(const AccelOption &option) noexcept {
    auto accel_handle = with_handle([&option] {
        return new_with_allocator<CUDAAccel>(option);
    });
    return {.handle = reinterpret_cast<uint64_t>(accel_handle),
            .native_handle = accel_handle};
}

void CUDADevice::destroy_accel(uint64_t handle) noexcept {
    with_handle([accel = reinterpret_cast<CUDAAccel *>(handle)] {
        delete_with_allocator(accel);
    });
}

string CUDADevice::query(luisa::string_view property) noexcept {
    LUISA_WARNING_WITH_LOCATION("Unknown device property '{}'.", property);
    return {};
}

DeviceExtension *CUDADevice::extension(luisa::string_view name) noexcept {
    LUISA_WARNING_WITH_LOCATION("Device extension '{}' is not implemented.", name);
    return nullptr;
}

CUDADevice::Handle::Handle(size_t index) noexcept {
    // global init
    static std::once_flag flag;
    std::call_once(flag, [] {
        LUISA_CHECK_CUDA(cuInit(0));
        static_cast<void>(optix::api());
    });

    // cuda
    auto driver_version = 0;
    LUISA_CHECK_CUDA(cuDriverGetVersion(&driver_version));
    _driver_version = driver_version;

    auto device_count = 0;
    LUISA_CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        LUISA_ERROR_WITH_LOCATION("No available device found for CUDA backend.");
    }
    if (index >= device_count) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid device index {} (device count = {}). Limiting to {}.",
            index, device_count, device_count - 1);
        index = device_count - 1;
    }
    LUISA_CHECK_CUDA(cuDeviceGet(&_device, index));
    auto compute_cap_major = 0;
    auto compute_cap_minor = 0;
    LUISA_CHECK_CUDA(cuDeviceGetUuid(&_uuid, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));

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
    LUISA_INFO("Destroyed CUDA device: {}.", name());
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
#ifndef NDEBUG
        // Disable due to too much overhead
        // optix_options.validationMode = optix::DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
        optix_options.logCallbackFunction = [](uint level, const char *tag, const char *message, void *) noexcept {
            auto log = luisa::format("Logs from OptiX ({}): {}", tag, message);
            if (level >= 4) {
                LUISA_INFO("{}", log);
            } else [[unlikely]] {
                LUISA_WARNING("{}", log);
            }
        };
        LUISA_CHECK_OPTIX(optix::api().deviceContextCreate(
            _context, &optix_options, &_optix_context));
    }
    return _optix_context;
}

void CUDADevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
    // TODO: set resource name
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
    auto p = dynamic_cast<luisa::compute::cuda::CUDADevice *>(device);
    LUISA_ASSERT(p != nullptr, "Deleting a null CUDA device.");
    luisa::delete_with_allocator(p);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    auto device_count = 0;
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
