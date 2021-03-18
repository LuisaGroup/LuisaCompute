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
#include <runtime/command.h>

namespace luisa::compute {

template<typename T>
class Buffer;

class Stream;

class Device {

private:
    template<typename T>
    friend class Buffer;
    [[nodiscard]] virtual uint64_t _create_buffer(size_t size_bytes) noexcept = 0;
    virtual void _dispose_buffer(uint64_t handle) noexcept = 0;

    friend class Stream;
    [[nodiscard]] virtual uint64_t _create_stream() noexcept = 0;
    virtual void _dispose_stream(uint64_t handle) noexcept = 0;
    virtual void _synchronize_stream(uint64_t stream_handle) noexcept = 0;
    virtual void _dispatch(uint64_t stream_handle, BufferCopyCommand) noexcept = 0;
    virtual void _dispatch(uint64_t stream_handle, BufferUploadCommand) noexcept = 0;
    virtual void _dispatch(uint64_t stream_handle, BufferDownloadCommand) noexcept = 0;
    virtual void _dispatch(uint64_t stream_handle, KernelLaunchCommand) noexcept = 0;
    virtual void _dispatch(uint64_t stream_handle, std::function<void()>) noexcept = 0;

public:
    virtual ~Device() noexcept = default;

    template<typename T>
    [[nodiscard]] auto create_buffer(size_t size) noexcept {
        auto handle = _create_buffer(size * sizeof(T));
        return Buffer<T>{this, size, handle};
    }

    [[nodiscard]] Stream create_stream() noexcept;
};

using DeviceDeleter = void(Device *);
using DeviceCreator = Device *(uint32_t index);
using DeviceHandle = std::unique_ptr<Device, DeviceDeleter *>;

}// namespace luisa::compute
