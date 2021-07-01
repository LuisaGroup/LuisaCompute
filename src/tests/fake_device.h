//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#include <runtime/device.h>
#include <runtime/context.h>

namespace luisa::compute {

class FakeDevice : public Device::Interface {

private:
    uint64_t _handle{0u};

public:
    explicit FakeDevice(const Context &ctx) noexcept
        : Device::Interface{ctx} {}
    ~FakeDevice() noexcept override = default;
    uint64_t create_buffer(size_t) noexcept override { return _handle++; }
    void dispose_buffer(uint64_t) noexcept override {}
    uint64_t create_stream() noexcept override { return _handle++; }
    void dispose_stream(uint64_t) noexcept override {}
    void synchronize_stream(uint64_t stream_handle) noexcept override {}
    void dispatch(uint64_t stream_handle, CommandBuffer) noexcept override {}
    void compile(const detail::FunctionBuilder *) noexcept override {}
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, uint64_t heap_handle, uint32_t index_in_heap) override { return _handle++; }
    void dispose_texture(uint64_t handle) noexcept override {}
    uint64_t create_event() noexcept override { return _handle++; }
    void synchronize_event(uint64_t handle) noexcept override {}
    void dispose_event(uint64_t handle) noexcept override {}
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override {}
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override {}
    virtual uint64_t create_mesh(uint64_t stream_handle,
                                 uint64_t vertex_buffer_handle, size_t vertex_buffer_offset_bytes, size_t vertex_count,
                                 uint64_t index_buffer_handle, size_t index_buffer_offset_bytes, size_t triangle_count) noexcept override { return _handle++; }
    virtual void dispose_mesh(uint64_t handle) noexcept override {}
    virtual uint64_t create_accel(uint64_t stream_handle,
                                  uint64_t mesh_handle_buffer_handle, size_t mesh_handle_buffer_offset_bytes,
                                  uint64_t transform_buffer_handle, size_t transform_buffer_offset_bytes,
                                  size_t mesh_count) noexcept override { return _handle++; }
    virtual void dispose_accel(uint64_t handle) noexcept override {}

    [[nodiscard]] static auto create(const Context &ctx) noexcept {
        auto deleter = [](Device::Interface *d) { delete d; };
        return Device{Device::Handle{new FakeDevice{ctx}, deleter}};
    }
    virtual uint64_t create_texture_heap(size_t size) noexcept override { return _handle++; }
    virtual size_t query_texture_heap_memory_usage(uint64_t handle) noexcept override { return 0u; }
    virtual void dispose_texture_heap(uint64_t handle) noexcept override {}
};

}// namespace luisa::compute
