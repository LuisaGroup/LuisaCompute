//
// Created by Mike on 7/28/2021.
//

#include <cstring>
#include <fstream>
#include <future>
#include <thread>

#include <runtime/rhi/sampler.h>
#include <runtime/bindless_array.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_buffer.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_compiler.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_shader.h>
#include <backends/cuda/optix_api.h>

#include <backends/cuda/cuda_builtin_kernels_embedded.inl.h>

namespace luisa::compute::cuda {
//
//uint64_t CUDADevice::create_buffer(size_t size_bytes) noexcept {
//    return with_handle([size = size_bytes, this] {
//        auto buffer = 0ull;
//        LUISA_CHECK_CUDA(cuMemAlloc(&buffer, size));
//        return buffer;
//    });
//}
//
//void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
//    with_handle([buffer = handle, this] {
//        LUISA_CHECK_CUDA(cuMemFree(buffer));
//    });
//}

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
//
//uint64_t CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
//    return with_handle([=] {
//        auto [array_format, num_channels] = cuda_array_format_and_channels(format);
//        CUDA_ARRAY3D_DESCRIPTOR array_desc{};
//        array_desc.Width = width;
//        array_desc.Height = height;
//        array_desc.Depth = dimension == 2u ? 0u : depth;
//        array_desc.Format = array_format;
//        array_desc.NumChannels = num_channels;
//        array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
//        auto array_handle = [&] {
//            if (mipmap_levels == 1u) {
//                CUarray handle{nullptr};
//                LUISA_CHECK_CUDA(cuArray3DCreate(&handle, &array_desc));
//                return reinterpret_cast<uint64_t>(handle);
//            }
//            CUmipmappedArray handle{nullptr};
//            LUISA_CHECK_CUDA(cuMipmappedArrayCreate(&handle, &array_desc, mipmap_levels));
//            return reinterpret_cast<uint64_t>(handle);
//        }();
//        return reinterpret_cast<uint64_t>(
//            new_with_allocator<CUDAMipmapArray>(array_handle, format, mipmap_levels));
//    });
//}
//
//void CUDADevice::destroy_texture(uint64_t handle) noexcept {
//    with_handle([array = reinterpret_cast<CUDAMipmapArray *>(handle)] {
//        delete_with_allocator(array);
//    });
//}
//
//uint64_t CUDADevice::create_stream(bool for_present) noexcept {
//    return with_handle([&] {
//        return reinterpret_cast<uint64_t>(new_with_allocator<CUDAStream>(this));
//    });
//}
//
//void CUDADevice::destroy_stream(uint64_t handle) noexcept {
//    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
//        delete_with_allocator(stream);
//    });
//}
//
//void CUDADevice::synchronize_stream(uint64_t handle) noexcept {
//    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
//        stream->synchronize();
//    });
//}
//
//void CUDADevice::dispatch(uint64_t stream_handle, move_only_function<void()> &&func) noexcept {
//    with_handle([this, stream = reinterpret_cast<CUDAStream *>(stream_handle), &func] {
//        stream->dispatch(std::move(func));
//    });
//}
//
//void CUDADevice::dispatch(uint64_t stream_handle, const CommandList &list) noexcept {
//    with_handle([this, stream = reinterpret_cast<CUDAStream *>(stream_handle), &list] {
//        CUDACommandEncoder encoder{this, stream};
//        for (auto &&cmd : list) { cmd->accept(encoder); }
//        stream->barrier();
//        stream->dispatch_callbacks();
//    });
//}
//
//void CUDADevice::dispatch(uint64_t stream_handle, luisa::span<const CommandList> lists) noexcept {
//    with_handle([this, stream = reinterpret_cast<CUDAStream *>(stream_handle), lists] {
//        for (auto &&list : lists) {
//            CUDACommandEncoder encoder{this, stream};
//            for (auto &&cmd : list) { cmd->accept(encoder); }
//            stream->barrier();
//        }
//        stream->dispatch_callbacks();
//    });
//}
//
//uint64_t CUDADevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
//    Clock clock;
//    auto ptx = CUDACompiler::instance().compile(context(), kernel, _handle.compute_capability());
//    auto entry = kernel.raytracing() ?
//                     luisa::format("__raygen__rg_{:016X}", kernel.hash()) :
//                     luisa::format("kernel_{:016X}", kernel.hash());
//    LUISA_INFO("Generated PTX for {} in {} ms.", entry, clock.toc());
//    return with_handle([&] {
//        auto shader = CUDAShader::create(this, ptx.c_str(), ptx.size(), entry.c_str(), kernel.raytracing());
//        return reinterpret_cast<uint64_t>(shader);
//    });
//}
//
//void CUDADevice::destroy_shader(uint64_t handle) noexcept {
//    with_handle([shader = reinterpret_cast<CUDAShader *>(handle)] {
//        CUDAShader::destroy(shader);
//    });
//}
//
//uint64_t CUDADevice::create_event() noexcept {
//    return with_handle([] {
//        CUevent event = nullptr;
//        LUISA_CHECK_CUDA(cuEventCreate(
//            &event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
//        return reinterpret_cast<uint64_t>(event);
//    });
//}
//
//void CUDADevice::destroy_event(uint64_t handle) noexcept {
//    with_handle([event = reinterpret_cast<CUevent>(handle)] {
//        LUISA_CHECK_CUDA(cuEventDestroy(event));
//    });
//}
//
//void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
//    with_handle([event = reinterpret_cast<CUevent>(handle),
//                 stream = reinterpret_cast<CUDAStream *>(stream_handle)] {
//        LUISA_CHECK_CUDA(cuEventRecord(event, stream->handle(true)));
//    });
//}
//
//void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
//    with_handle([event = reinterpret_cast<CUevent>(handle),
//                 stream = reinterpret_cast<CUDAStream *>(stream_handle)] {
//        LUISA_CHECK_CUDA(cuStreamWaitEvent(stream->handle(true), event, CU_EVENT_WAIT_DEFAULT));
//    });
//}
//
//void CUDADevice::synchronize_event(uint64_t handle) noexcept {
//    with_handle([event = reinterpret_cast<CUevent>(handle)] {
//        LUISA_CHECK_CUDA(cuEventSynchronize(event));
//    });
//}
//
//uint64_t CUDADevice::create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelUsageHint hint) noexcept {
//    return with_handle([=] {
//        auto mesh = new_with_allocator<CUDAMesh>(
//            v_buffer, v_offset, v_stride, v_count,
//            t_buffer, t_offset, t_count, hint);
//        return reinterpret_cast<uint64_t>(mesh);
//    });
//}
//
//void CUDADevice::destroy_mesh(uint64_t handle) noexcept {
//    with_handle([mesh = reinterpret_cast<CUDAMesh *>(handle)] {
//        delete_with_allocator(mesh);
//    });
//}
//
//uint64_t CUDADevice::create_accel(AccelUsageHint hint) noexcept {
//    return with_handle([=, this] {
//        auto accel = new_with_allocator<CUDAAccel>(hint);
//        return reinterpret_cast<uint64_t>(accel);
//    });
//}
//
//void CUDADevice::destroy_accel(uint64_t handle) noexcept {
//    with_handle([accel = reinterpret_cast<CUDAAccel *>(handle)] {
//        delete_with_allocator(accel);
//    });
//}

CUDADevice::CUDADevice(Context &&ctx, size_t device_id) noexcept
    : DeviceInterface{std::move(ctx)}, _handle{device_id} {
    with_handle([this] {
        LUISA_CHECK_CUDA(cuCtxResetPersistingL2Cache());
        LUISA_CHECK_CUDA(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1));
        LUISA_CHECK_CUDA(cuModuleLoadData(
            &_builtin_kernel_module, cuda_builtin_kernels_source));
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
    LUISA_ASSERT(stream_tag != StreamTag::GRAPHICS,
                 "Graphics streams are not supported by CUDA.");
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
    LUISA_ERROR_WITH_LOCATION("Swap chains are not supported on CUDA.");
}

void CUDADevice::destroy_swap_chain(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Swap chains are not supported on CUDA.");
}

void CUDADevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Swap chains are not supported on CUDA.");
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    LUISA_ERROR_WITH_LOCATION("TODO");
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    LUISA_ERROR_WITH_LOCATION("TODO");
}

ShaderCreationInfo CUDADevice::load_shader(luisa::string_view name,
                                           luisa::span<const Type *const> arg_types) noexcept {
    LUISA_ERROR_WITH_LOCATION("TODO");
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
    with_handle([shader = reinterpret_cast<CUDAShader *>(handle)] {
        CUDAShader::destroy(shader);
    });
}

ResourceCreationInfo CUDADevice::create_raster_shader(const MeshFormat &mesh_format,
                                                      const RasterState &raster_state,
                                                      luisa::span<const PixelFormat> rtv_format,
                                                      DepthFormat dsv_format,
                                                      Function vert, Function pixel,
                                                      const ShaderOption &shader_option) noexcept {
    LUISA_ERROR_WITH_LOCATION("Rasterization is not supported on CUDA.");
}

void CUDADevice::save_raster_shader(const MeshFormat &mesh_format,
                                    Function vert, Function pixel,
                                    luisa::string_view name,
                                    bool enable_debug_info,
                                    bool enable_fast_math) noexcept {
    LUISA_ERROR_WITH_LOCATION("Rasterization is not supported on CUDA.");
}

ResourceCreationInfo CUDADevice::load_raster_shader(const MeshFormat &mesh_format,
                                                    const RasterState &raster_state,
                                                    luisa::span<const PixelFormat> rtv_format,
                                                    DepthFormat dsv_format,
                                                    luisa::span<const Type *const> types,
                                                    luisa::string_view ser_path) noexcept {
    LUISA_ERROR_WITH_LOCATION("Rasterization is not supported on CUDA.");
}

void CUDADevice::destroy_raster_shader(uint64_t handle) noexcept {
    LUISA_ERROR_WITH_LOCATION("Rasterization is not supported on CUDA.");
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
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));
    LUISA_INFO(
        "Created CUDA device at index {}: {} (capability = {}.{}).",
        index, name(), compute_cap_major, compute_cap_minor);
    _compute_capability = 10u * compute_cap_major + compute_cap_minor;
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_context, _device));

    // optix
    optix::DeviceContextOptions optix_options{};
    optix_options.logCallbackLevel = 4u;
#ifndef NDEBUG
// Disable due to too much overhead
//    optix_options.validationMode = optix::DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    optix_options.logCallbackFunction = [](uint level, const char *tag, const char *message, void *) noexcept {
        auto log = fmt::format("Logs from OptiX ({}): {}", tag, message);
        if (level >= 4) {
            LUISA_INFO("{}", log);
        } else [[unlikely]] {
            LUISA_WARNING("{}", log);
        }
    };
    LUISA_CHECK_OPTIX(optix::api().deviceContextCreate(
        _context, &optix_options, &_optix_context));
}

CUDADevice::Handle::~Handle() noexcept {
    LUISA_CHECK_OPTIX(optix::api().deviceContextDestroy(_optix_context));
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRelease(_device));
    LUISA_INFO("Destroyed CUDA device: {}.", name());
}

std::string_view CUDADevice::Handle::name() const noexcept {
    static constexpr auto device_name_length = 1024u;
    static thread_local char device_name[device_name_length];
    LUISA_CHECK_CUDA(cuDeviceGetName(device_name, device_name_length, _device));
    return device_name;
}

}// namespace luisa::compute::cuda

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx, const luisa::compute::DeviceConfig *config) noexcept {
    auto device_id = config == nullptr ? 0u : config->device_index;
    return luisa::new_with_allocator<luisa::compute::cuda::CUDADevice>(
        std::move(ctx), device_id);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    auto p = dynamic_cast<luisa::compute::cuda::CUDADevice *>(device);
    LUISA_ASSERT(p != nullptr, "Deleting a null CUDA device.");
    luisa::delete_with_allocator(p);
}
