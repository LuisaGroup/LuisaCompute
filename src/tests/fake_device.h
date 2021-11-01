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
    uint64_t create_buffer(size_t, uint64_t heap_handle, uint32_t index_in_heap) noexcept override { return _handle++; }
    void destroy_buffer(uint64_t) noexcept override {}
    uint64_t create_stream() noexcept override { return _handle++; }
    void destroy_stream(uint64_t) noexcept override {}
    void synchronize_stream(uint64_t stream_handle) noexcept override {}
    void dispatch(uint64_t stream_handle, CommandList) noexcept override {}
    uint64_t create_shader(Function kernel, std::string_view meta_options) noexcept override { return _handle++; }
    void destroy_shader(uint64_t handle) noexcept override {}
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels,
                            Sampler sampler, uint64_t heap_handle, uint32_t index_in_heap) override { return _handle++; }
    void destroy_texture(uint64_t handle) noexcept override {}
    uint64_t create_event() noexcept override { return _handle++; }
    void synchronize_event(uint64_t handle) noexcept override {}
    void destroy_event(uint64_t handle) noexcept override {}
    void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override {}
    void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override {}
    virtual uint64_t create_mesh() noexcept override { return _handle++; }
    virtual void destroy_mesh(uint64_t handle) noexcept override {}
    virtual uint64_t create_accel() noexcept override { return _handle++; }
    virtual void destroy_accel(uint64_t handle) noexcept override {}

    [[nodiscard]] static auto create(const Context &ctx) noexcept {
        auto deleter = [](Device::Interface *d) { delete d; };
        return Device{Device::Handle{new FakeDevice{ctx}, deleter}};
    }
    virtual uint64_t create_heap(size_t size) noexcept override { return _handle++; }
    virtual size_t query_heap_memory_usage(uint64_t handle) noexcept override { return 0u; }
    virtual void destroy_heap(uint64_t handle) noexcept override {}
    virtual void *buffer_native_handle(uint64_t handle) const noexcept override { return nullptr; }
    virtual void *texture_native_handle(uint64_t handle) const noexcept override { return nullptr; }
    virtual void *native_handle() const noexcept override { return nullptr; }
    virtual void *stream_native_handle(uint64_t) const noexcept override { return nullptr; }
};

}// namespace luisa::compute
