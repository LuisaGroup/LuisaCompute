//
// Created by Mike Smith on 2020/12/2.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>

#include <core/memory.h>
#include <core/concepts.h>
#include <runtime/pixel_format.h>
#include <runtime/command_buffer.h>

namespace luisa::compute {

class Context;

class Device {

private:
    const Context &_ctx;

public:
    explicit Device(const Context &ctx) noexcept : _ctx{ctx} {}
    virtual ~Device() noexcept = default;
    
    [[nodiscard]] const Context &context() const noexcept { return _ctx; }

    // buffer
    [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
    virtual void dispose_buffer(uint64_t handle) noexcept = 0;

    // texture
    [[nodiscard]] virtual uint64_t create_texture(
        PixelFormat format, uint dimension, uint width, uint height, uint depth,
        uint mipmap_levels, bool is_bindless) = 0;
    virtual void dispose_texture(uint64_t handle) noexcept = 0;

    // stream
    [[nodiscard]] virtual uint64_t create_stream() noexcept = 0;
    virtual void dispose_stream(uint64_t handle) noexcept = 0;
    virtual void synchronize_stream(uint64_t stream_handle) noexcept = 0;
    virtual void dispatch(uint64_t stream_handle, CommandBuffer, std::function<void()>) noexcept = 0;

    // kernel
    virtual void prepare_kernel(uint32_t uid) noexcept = 0;
};

using DeviceDeleter = void(Device *);
using DeviceCreator = Device *(const Context &ctx, uint32_t index);
using DeviceHandle = std::unique_ptr<Device, DeviceDeleter *>;

}// namespace luisa::compute
