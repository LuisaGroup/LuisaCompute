#pragma once

#include <runtime/device.h>

using namespace luisa;
using namespace luisa::compute;

namespace lc::ispc {
class ISPCDevice final : public Device::Interface {
    void *native_handle() const noexcept override;

    // buffer
    uint64_t create_buffer(
        size_t size_bytes,
        uint64_t heap_handle,// == uint64(-1) when not from heap
        uint32_t index_in_heap) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    void *buffer_native_handle(uint64_t handle) const noexcept override;

    // texture
    uint64_t create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels,
        TextureSampler sampler,
        uint64_t heap_handle,// == uint64(-1) when not from heap
        uint32_t index_in_heap) override;
    void destroy_texture(uint64_t handle) noexcept override;
    void *texture_native_handle(uint64_t handle) const noexcept override;

    // texture heap
    uint64_t create_heap(size_t size) noexcept override;
    size_t query_heap_memory_usage(uint64_t handle) noexcept override;
    void destroy_heap(uint64_t handle) noexcept override;

    // stream
    uint64_t create_stream() noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(uint64_t stream_handle, CommandList) noexcept override;
    void *stream_native_handle(uint64_t handle) const noexcept override;

    // kernel
    uint64_t create_shader(Function kernel) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // event
    uint64_t create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override;
    void synchronize_event(uint64_t handle) noexcept override;

    // accel
    uint64_t create_mesh() noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;
    uint64_t create_accel() noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;
};

}// namespace lc::ispc
