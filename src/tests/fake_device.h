//
// Created by Mike Smith on 2021/3/17.
//

#pragma once

#include <runtime/device.h>
#include <runtime/context.h>

namespace luisa::compute {

class FakeDevice : public Device {

private:
    uint64_t _handle{0u};

public:
    explicit FakeDevice(const Context &ctx) noexcept : Device{ctx} {}
    ~FakeDevice() noexcept override = default;
    uint64_t create_buffer(size_t) noexcept override { return _handle++; }
    void dispose_buffer(uint64_t) noexcept override {}
    uint64_t create_stream() noexcept override { return _handle++; }
    void dispose_stream(uint64_t) noexcept override {}
    void synchronize_stream(uint64_t stream_handle) noexcept override {}
    void dispatch(uint64_t stream_handle, CommandBuffer) noexcept override {}
    void compile_kernel(uint32_t uid) noexcept override {}
    uint64_t create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool is_bindless) override { return _handle++; }
    void dispose_texture(uint64_t handle) noexcept override {}
    uint64_t create_event() noexcept override { return _handle++; }
    void synchronize_event(uint64_t handle) noexcept override {}
    void dispose_event(uint64_t handle) noexcept override {}
};

}// namespace luisa::compute
