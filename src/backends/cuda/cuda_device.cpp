//
// Created by Mike on 7/28/2021.
//

#include <runtime/texture.h>
#include <backends/cuda/cuda_device.h>

namespace luisa::compute::cuda {

uint64_t CUDADevice::create_buffer(size_t size_bytes, uint64_t heap_handle, uint32_t index_in_heap) noexcept {
    return 0;
}

void CUDADevice::destroy_buffer(uint64_t handle) noexcept {

}

uint64_t CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, TextureSampler sampler, uint64_t heap_handle, uint32_t index_in_heap) {
    return 0;
}

void CUDADevice::destroy_texture(uint64_t handle) noexcept {
}

uint64_t CUDADevice::create_heap(size_t size) noexcept {
    return 0;
}

size_t CUDADevice::query_heap_memory_usage(uint64_t handle) noexcept {
    return 0;
}

void CUDADevice::destroy_heap(uint64_t handle) noexcept {

}

uint64_t CUDADevice::create_stream() noexcept {
    return _handle.with([&] {
        CUstream stream = nullptr;
        LUISA_CHECK_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
        return reinterpret_cast<uint64_t>(stream);
    });
}

void CUDADevice::destroy_stream(uint64_t handle) noexcept {
    _handle.with([stream = reinterpret_cast<CUstream>(handle)] {
        LUISA_CHECK_CUDA(cuStreamDestroy(stream));
    });
}

void CUDADevice::synchronize_stream(uint64_t handle) noexcept {
    _handle.with([stream = reinterpret_cast<CUstream>(handle)] {
        LUISA_CHECK_CUDA(cuStreamSynchronize(stream));
    });
}

void CUDADevice::dispatch(uint64_t stream_handle, CommandList list) noexcept {
}

uint64_t CUDADevice::create_shader(Function kernel) noexcept {
    return 0;
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
}

uint64_t CUDADevice::create_event() noexcept {
    return _handle.with([] {
        CUevent event = nullptr;
        LUISA_CHECK_CUDA(cuEventCreate(
            &event, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
        return reinterpret_cast<uint64_t>(event);
    });
}

void CUDADevice::destroy_event(uint64_t handle) noexcept {
    _handle.with([event = reinterpret_cast<CUevent>(handle)] {
        LUISA_CHECK_CUDA(cuEventDestroy(event));
    });
}

void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    _handle.with([event = reinterpret_cast<CUevent>(handle),
                  stream = reinterpret_cast<CUstream>(stream_handle)] {
      LUISA_CHECK_CUDA(cuEventRecord(event, stream));
    });
}

void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    _handle.with([event = reinterpret_cast<CUevent>(handle),
                  stream = reinterpret_cast<CUstream>(stream_handle)] {
        LUISA_CHECK_CUDA(cuStreamWaitEvent(stream, event, CU_EVENT_WAIT_DEFAULT));
    });
}

void CUDADevice::synchronize_event(uint64_t handle) noexcept {
    _handle.with([event = reinterpret_cast<CUevent>(handle)] {
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
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_context, _device));
    auto compute_cap_major = 0;
    auto compute_cap_minor = 0;
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));
    LUISA_INFO(
        "Created CUDA device at index {}: {} (capability = {}.{}).",
        index, name(), compute_cap_major, compute_cap_minor);
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

LUISA_EXPORT luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::cuda::CUDADevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device::Interface *device) noexcept {
    delete device;
}
