//
// Created by Mike on 7/28/2021.
//

#include <cstring>
#include <fstream>

#include <nlohmann/json.hpp>

#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <runtime/sampler.h>
#include <runtime/bindless_array.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_mesh.h>
#include <backends/cuda/cuda_accel.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_compiler.h>
#include <backends/cuda/cuda_bindless_array.h>
#include <backends/cuda/cuda_command_encoder.h>
#include <backends/cuda/cuda_mipmap_array.h>
#include <backends/cuda/cuda_shader.h>

namespace luisa::compute::cuda {

uint64_t CUDADevice::create_buffer(size_t size_bytes) noexcept {
    return with_handle([size = size_bytes] {
        CUdeviceptr ptr = 0ul;
        LUISA_CHECK_CUDA(cuMemAlloc(&ptr, size));
        return ptr;
    });
}

void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
    with_handle([buffer = handle] {
        LUISA_CHECK_CUDA(cuMemFree(buffer));
    });
}

[[nodiscard]] static auto cuda_array_format_and_channels(PixelFormat format) noexcept {
    CUarray_format array_format = CU_AD_FORMAT_UNSIGNED_INT8;
    auto num_channels = 0;
    switch (format) {
        case PixelFormat::R8SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT8;
            num_channels = 1;
            break;
        case PixelFormat::R8UInt:
        case PixelFormat::R8UNorm:
            array_format = CU_AD_FORMAT_UNSIGNED_INT8;
            num_channels = 1;
            break;
        case PixelFormat::RG8SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT8;
            num_channels = 2;
            break;
        case PixelFormat::RG8UInt:
        case PixelFormat::RG8UNorm:
            array_format = CU_AD_FORMAT_UNSIGNED_INT8;
            num_channels = 2;
            break;
        case PixelFormat::RGBA8SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT8;
            num_channels = 4;
            break;
        case PixelFormat::RGBA8UInt:
        case PixelFormat::RGBA8UNorm:
            array_format = CU_AD_FORMAT_UNSIGNED_INT8;
            num_channels = 4;
            break;
        case PixelFormat::R16SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT16;
            num_channels = 1;
            break;
        case PixelFormat::R16UInt:
        case PixelFormat::R16UNorm:
            array_format = CU_AD_FORMAT_UNSIGNED_INT16;
            num_channels = 1;
            break;
        case PixelFormat::RG16SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT16;
            num_channels = 2;
            break;
        case PixelFormat::RG16UInt:
        case PixelFormat::RG16UNorm:
            array_format = CU_AD_FORMAT_UNSIGNED_INT16;
            num_channels = 2;
            break;
        case PixelFormat::RGBA16SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT16;
            num_channels = 4;
            break;
        case PixelFormat::RGBA16UInt:
        case PixelFormat::RGBA16UNorm:
            array_format = CU_AD_FORMAT_UNSIGNED_INT16;
            num_channels = 4;
            break;
        case PixelFormat::R32SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT32;
            num_channels = 1;
            break;
        case PixelFormat::R32UInt:
            array_format = CU_AD_FORMAT_UNSIGNED_INT32;
            num_channels = 1;
            break;
        case PixelFormat::RG32SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT32;
            num_channels = 2;
            break;
        case PixelFormat::RG32UInt:
            array_format = CU_AD_FORMAT_UNSIGNED_INT32;
            num_channels = 2;
            break;
        case PixelFormat::RGBA32SInt:
            array_format = CU_AD_FORMAT_SIGNED_INT32;
            num_channels = 4;
            break;
        case PixelFormat::RGBA32UInt:
            array_format = CU_AD_FORMAT_UNSIGNED_INT32;
            num_channels = 4;
            break;
        case PixelFormat::R16F:
            array_format = CU_AD_FORMAT_HALF;
            num_channels = 1;
            break;
        case PixelFormat::RG16F:
            array_format = CU_AD_FORMAT_HALF;
            num_channels = 2;
            break;
        case PixelFormat::RGBA16F:
            array_format = CU_AD_FORMAT_HALF;
            num_channels = 4;
            break;
        case PixelFormat::R32F:
            array_format = CU_AD_FORMAT_FLOAT;
            num_channels = 1;
            break;
        case PixelFormat::RG32F:
            array_format = CU_AD_FORMAT_FLOAT;
            num_channels = 2;
            break;
        case PixelFormat::RGBA32F:
            array_format = CU_AD_FORMAT_FLOAT;
            num_channels = 4;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("Invalid pixel format.");
    }
    return std::make_pair(array_format, num_channels);
}

uint64_t CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels) noexcept {
    return with_handle([=] {
        auto [array_format, num_channels] = cuda_array_format_and_channels(format);
        CUDA_ARRAY3D_DESCRIPTOR array_desc{};
        array_desc.Width = width;
        array_desc.Height = height;
        array_desc.Depth = dimension == 2u ? 0u : depth;
        array_desc.Format = array_format;
        array_desc.NumChannels = num_channels;
        array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
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
        return reinterpret_cast<uint64_t>(
            new_with_allocator<CUDAMipmapArray>(array_handle, format, mipmap_levels));
    });
}

void CUDADevice::destroy_texture(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDAMipmapArray *>(handle)] {
        delete_with_allocator(array);
    });
}

uint64_t CUDADevice::create_stream() noexcept {
    return with_handle([&] {
        return reinterpret_cast<uint64_t>(new_with_allocator<CUDAStream>());
    });
}

void CUDADevice::destroy_stream(uint64_t handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
        delete_with_allocator(stream);
    });
}

void CUDADevice::synchronize_stream(uint64_t handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
        LUISA_CHECK_CUDA(cuStreamSynchronize(stream->handle()));
    });
}

void CUDADevice::dispatch(uint64_t stream_handle, CommandList list) noexcept {
    with_handle([this, stream = reinterpret_cast<CUDAStream *>(stream_handle), cmd_list = std::move(list)] {
        CUDACommandEncoder encoder{this, stream};
        for (auto cmd : cmd_list) {
            cmd->accept(encoder);
        }
    });
}

uint64_t CUDADevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    Clock clock;
    auto ptx = CUDACompiler::instance().compile(context(), kernel, _handle.compute_capability());
    auto entry = kernel.raytracing() ?
        fmt::format("__raygen__rg_{:016X}", kernel.hash()) :
        fmt::format("kernel_{:016X}", kernel.hash());
    LUISA_INFO(
        "Generated PTX for {} in {} ms.",
        entry, clock.toc());
    return with_handle([&] {
        auto shader = CUDAShader::create(this, ptx.c_str(), ptx.size(), entry.c_str(), kernel.raytracing());
        return reinterpret_cast<uint64_t>(shader);
    });
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
    with_handle([shader = reinterpret_cast<CUDAShader *>(handle)] {
        CUDAShader::destroy(shader);
    });
}

uint64_t CUDADevice::create_event() noexcept {
    return with_handle([] {
        CUevent event = nullptr;
        LUISA_CHECK_CUDA(cuEventCreate(
            &event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
        return reinterpret_cast<uint64_t>(event);
    });
}

void CUDADevice::destroy_event(uint64_t handle) noexcept {
    with_handle([event = reinterpret_cast<CUevent>(handle)] {
        LUISA_CHECK_CUDA(cuEventDestroy(event));
    });
}

void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    with_handle([event = reinterpret_cast<CUevent>(handle),
                 stream = reinterpret_cast<CUDAStream *>(stream_handle)] {
        LUISA_CHECK_CUDA(cuEventRecord(event, stream->handle()));
    });
}

void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    with_handle([event = reinterpret_cast<CUevent>(handle),
                 stream = reinterpret_cast<CUDAStream *>(stream_handle)] {
        LUISA_CHECK_CUDA(cuStreamWaitEvent(stream->handle(), event, CU_EVENT_WAIT_DEFAULT));
    });
}

void CUDADevice::synchronize_event(uint64_t handle) noexcept {
    with_handle([event = reinterpret_cast<CUevent>(handle)] {
        LUISA_CHECK_CUDA(cuEventSynchronize(event));
    });
}

uint64_t CUDADevice::create_mesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count, uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint) noexcept {
    return with_handle([=] {
        auto mesh = new_with_allocator<CUDAMesh>(
            v_buffer, v_offset, v_stride, v_count,
            t_buffer, t_offset, t_count, hint);
        return reinterpret_cast<uint64_t>(mesh);
    });
}

void CUDADevice::destroy_mesh(uint64_t handle) noexcept {
    with_handle([mesh = reinterpret_cast<CUDAMesh *>(handle)] {
        delete_with_allocator(mesh);
    });
}

uint64_t CUDADevice::create_accel(AccelBuildHint hint) noexcept {
    return with_handle([=] {
        auto accel = new_with_allocator<CUDAAccel>(hint);
        return reinterpret_cast<uint64_t>(accel);
    });
}

void CUDADevice::destroy_accel(uint64_t handle) noexcept {
    with_handle([accel = reinterpret_cast<CUDAAccel *>(handle)] {
        delete_with_allocator(accel);
    });
}

CUDADevice::CUDADevice(const Context &ctx, uint device_id) noexcept
    : Device::Interface{ctx}, _handle{device_id} {}

uint64_t CUDADevice::create_bindless_array(size_t size) noexcept {
    return with_handle([size] {
        return reinterpret_cast<uint64_t>(new_with_allocator<CUDABindlessArray>(size));
    });
}

void CUDADevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(handle)] {
        delete_with_allocator(array);
    });
}

void CUDADevice::emplace_buffer_in_bindless_array(uint64_t array, size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), index, buffer = handle, offset_bytes] {
        array->emplace_buffer(index, buffer, offset_bytes);
    });
}

void CUDADevice::emplace_tex2d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), index, tex = reinterpret_cast<CUDAMipmapArray *>(handle), sampler] {
        array->emplace_tex2d(index, tex, sampler);
    });
}

void CUDADevice::emplace_tex3d_in_bindless_array(uint64_t array, size_t index, uint64_t handle, Sampler sampler) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), index, tex = reinterpret_cast<CUDAMipmapArray *>(handle), sampler] {
        array->emplace_tex3d(index, tex, sampler);
    });
}

bool CUDADevice::is_buffer_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), handle] {
        return array->has_buffer(handle);
    });
}

bool CUDADevice::is_texture_in_bindless_array(uint64_t array, uint64_t handle) const noexcept {
    return with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), tex = reinterpret_cast<CUDAMipmapArray *>(handle)] {
        return array->has_array(tex);
    });
}

void CUDADevice::remove_buffer_in_bindless_array(uint64_t array, size_t index) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), index] {
        array->remove_buffer(index);
    });
}

void CUDADevice::remove_tex2d_in_bindless_array(uint64_t array, size_t index) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), index] {
        array->remove_tex2d(index);
    });
}

void CUDADevice::remove_tex3d_in_bindless_array(uint64_t array, size_t index) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(array), index] {
        array->remove_tex3d(index);
    });
}

void CUDADevice::emplace_back_instance_in_accel(uint64_t accel_handle, uint64_t mesh_handle, float4x4 transform) noexcept {
    auto accel = reinterpret_cast<CUDAAccel *>(accel_handle);
    auto mesh = reinterpret_cast<CUDAMesh *>(mesh_handle);
    accel->add_instance(mesh, transform);
}

void CUDADevice::set_instance_transform_in_accel(uint64_t accel_handle, size_t index, float4x4 transform) noexcept {
    auto accel = reinterpret_cast<CUDAAccel *>(accel_handle);
    accel->set_transform(index, transform);
}

bool CUDADevice::is_buffer_in_accel(uint64_t accel, uint64_t buffer) const noexcept {
    return reinterpret_cast<CUDAAccel *>(accel)->uses_resource(buffer);
}

bool CUDADevice::is_mesh_in_accel(uint64_t accel, uint64_t mesh) const noexcept {
    return reinterpret_cast<CUDAAccel *>(accel)->uses_resource(mesh);
}

uint64_t CUDADevice::get_vertex_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return reinterpret_cast<CUDAMesh *>(mesh_handle)->vertex_buffer_handle();
}

uint64_t CUDADevice::get_triangle_buffer_from_mesh(uint64_t mesh_handle) const noexcept {
    return reinterpret_cast<CUDAMesh *>(mesh_handle)->triangle_buffer_handle();
}

CUDADevice::Handle::Handle(uint index) noexcept {

    // global init
    static std::once_flag flag;
    std::call_once(flag, [] {
        LUISA_CHECK_CUDA(cuInit(0));
        LUISA_CHECK_OPTIX(optixInit());
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
    OptixDeviceContextOptions optix_options{};
    optix_options.logCallbackLevel = 4;
    optix_options.logCallbackFunction = [](uint32_t level, const char *tag, const char *message, void *) noexcept {
        auto log = fmt::format("Logs from OptiX ({}): {}", tag, message);
        if (level >= 4) {
            LUISA_INFO("{}", log);
        } else [[unlikely]] {
            LUISA_WARNING("{}", log);
        }
    };
    LUISA_CHECK_OPTIX(optixDeviceContextCreate(_context, &optix_options, &_optix_context));
}

CUDADevice::Handle::~Handle() noexcept {
    LUISA_CHECK_OPTIX(optixDeviceContextDestroy(_optix_context));
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

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, std::string_view properties) noexcept {
    return new luisa::compute::cuda::CUDADevice{ctx, 0};// TODO: decode properties
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    delete device;
}
