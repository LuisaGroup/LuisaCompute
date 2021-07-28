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
    return 0;
}
void CUDADevice::destroy_stream(uint64_t handle) noexcept {
}
void CUDADevice::synchronize_stream(uint64_t stream_handle) noexcept {
}
void CUDADevice::dispatch(uint64_t stream_handle, CommandList list) noexcept {
}
uint64_t CUDADevice::create_shader(Function kernel) noexcept {
    return 0;
}
void CUDADevice::destroy_shader(uint64_t handle) noexcept {
}
uint64_t CUDADevice::create_event() noexcept {
    return 0;
}
void CUDADevice::destroy_event(uint64_t handle) noexcept {
}
void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
}
void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
}
void CUDADevice::synchronize_event(uint64_t handle) noexcept {
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
    : Device::Interface{ctx} {}

}// namespace luisa::compute::cuda

LUISA_EXPORT luisa::compute::Device::Interface *create(const luisa::compute::Context &ctx, uint32_t id) noexcept {
    return new luisa::compute::cuda::CUDADevice{ctx, id};
}

LUISA_EXPORT void destroy(luisa::compute::Device::Interface *device) noexcept {
    delete device;
}
