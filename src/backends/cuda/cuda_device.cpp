//
// Created by Mike on 7/28/2021.
//

#include <runtime/sampler.h>
#include <runtime/heap.h>
#include <backends/cuda/cuda_heap.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_stream.h>
#include <backends/cuda/cuda_command_encoder.h>

namespace luisa::compute::cuda {

uint64_t CUDADevice::create_buffer(size_t size_bytes, uint64_t heap_handle, uint32_t index_in_heap) noexcept {
    if (heap_handle != Heap::invalid_handle) {// from heap
        return with_handle([heap = reinterpret_cast<CUDAHeap *>(heap_handle), index = index_in_heap, size = size_bytes] {
            return reinterpret_cast<uint64_t>(heap->allocate_buffer(index, size));
        });
    }
    return with_handle([size = size_bytes] {
        CUdeviceptr ptr = 0ul;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&ptr, size, nullptr));
        return reinterpret_cast<uint64_t>(new CUDABuffer{ptr});
    });
}

void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
    with_handle([buffer = reinterpret_cast<CUDABuffer *>(handle)] {
        if (auto heap = buffer->heap(); heap != nullptr) {
            heap->destroy_buffer(buffer);
        } else {
            LUISA_CHECK_CUDA(cuMemFreeAsync(buffer->handle(), nullptr));
            delete buffer;
        }
    });
}

uint64_t CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, Sampler sampler, uint64_t heap_handle, uint32_t index_in_heap) {

    if (heap_handle != Heap::invalid_handle) {// from heap
        return with_handle([heap = reinterpret_cast<CUDAHeap *>(heap_handle),
                            index = index_in_heap,
                            format, dimension,
                            size = make_uint3(width, height, depth),
                            mipmap_levels, sampler] {
            return reinterpret_cast<uint64_t>(heap->allocate_texture(index, format, dimension, size, mipmap_levels, sampler));
        });
    }

    return with_handle([format, dimension, size = make_uint3(width, height, depth)] {
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
        }

        auto array = [dimension, array_format, num_channels, size] {
            CUarray array_handle{nullptr};
            if (dimension == 2u) {
                CUDA_ARRAY_DESCRIPTOR array_desc{};
                array_desc.Width = size.x;
                array_desc.Height = size.y;
                array_desc.Format = array_format;
                array_desc.NumChannels = num_channels;
                LUISA_CHECK_CUDA(cuArrayCreate(&array_handle, &array_desc));
            } else {
                CUDA_ARRAY3D_DESCRIPTOR array_desc{};
                array_desc.Width = size.x;
                array_desc.Height = size.y;
                array_desc.Depth = size.z;
                array_desc.Format = array_format;
                array_desc.NumChannels = num_channels;
                LUISA_CHECK_CUDA(cuArray3DCreate(&array_handle, &array_desc));
            }
            return array_handle;
        }();
        CUDA_RESOURCE_DESC resource_desc{};
        resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        resource_desc.res.array.hArray = array;
        CUsurfObject surface = 0u;
        LUISA_CHECK_CUDA(cuSurfObjectCreate(&surface, &resource_desc));
        return reinterpret_cast<uint64_t>(new CUDATexture{surface, array, dimension});
    });
}

void CUDADevice::destroy_texture(uint64_t handle) noexcept {
    with_handle([texture = reinterpret_cast<CUDATexture *>(handle)] {
        if (auto heap = texture->heap(); heap != nullptr) {
            heap->destroy_texture(texture);
        } else {
            LUISA_CHECK_CUDA(cuArrayDestroy(texture->array()));
            LUISA_CHECK_CUDA(cuSurfObjectDestroy(texture->handle()));
            delete texture;
        }
    });
}

uint64_t CUDADevice::create_heap(size_t size) noexcept {
    return with_handle([this, size] {
        return reinterpret_cast<uint64_t>(new CUDAHeap{this, size});
    });
}

size_t CUDADevice::query_heap_memory_usage(uint64_t handle) noexcept {
    return with_handle([heap = reinterpret_cast<CUDAHeap *>(handle)] {
        return heap->memory_usage();
    });
}

void CUDADevice::destroy_heap(uint64_t handle) noexcept {
    with_handle([heap = reinterpret_cast<CUDAHeap *>(handle)] {
        delete heap;
    });
}

uint64_t CUDADevice::create_stream() noexcept {
    return with_handle([&] {
        return reinterpret_cast<uint64_t>(new CUDAStream);
    });
}

void CUDADevice::destroy_stream(uint64_t handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
        delete stream;
    });
}

void CUDADevice::synchronize_stream(uint64_t handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
        LUISA_CHECK_CUDA(cuStreamSynchronize(stream->handle()));
    });
}

void CUDADevice::dispatch(uint64_t stream_handle, CommandList list) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle), cmd_list = std::move(list)] {
        CUDACommandEncoder encoder{stream};
        for (auto cmd : cmd_list) {
            cmd->accept(encoder);
        }
    });
}

uint64_t CUDADevice::create_shader(Function kernel, std::string_view meta_options) noexcept {
    return 0;
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
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

uint64_t CUDADevice::create_mesh() noexcept {
    return 0;
}

void CUDADevice::destroy_mesh(uint64_t handle) noexcept {
}

uint64_t CUDADevice::create_accel() noexcept {
    return 0;
}

void CUDADevice::destroy_accel(uint64_t handle) noexcept {
}

CUDADevice::CUDADevice(const Context &ctx, uint device_id) noexcept
    : Device::Interface{ctx}, _handle{device_id} {}

CUDADevice::Handle::Handle(uint index) noexcept {
    static std::once_flag flag;
    std::call_once(flag, [] { LUISA_CHECK_CUDA(cuInit(0)); });
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
    auto supports_memory_pools = 0;
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&supports_memory_pools, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, _device));
    LUISA_INFO("CUDA device supports memory pools: {}.", supports_memory_pools != 0);
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_context, _device));
}

CUDADevice::Handle::~Handle() noexcept {
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

LUISA_EXPORT_API luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::cuda::CUDADevice{ctx, id};
}

LUISA_EXPORT_API void destroy(luisa::compute::Device::Interface *device) noexcept {
    delete device;
}
