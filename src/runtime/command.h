//
// Created by Mike Smith on 2021/2/15.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <core/memory.h>

namespace luisa::compute {

class Device;
class Stream;

class Command {

private:
    Device *_device;

protected:
    explicit Command(Device *device) noexcept : _device{device} {}
    ~Command() noexcept = default;

public:
    Command(Command &&) noexcept = delete;
    Command(const Command &) noexcept = delete;
    Command &operator=(Command &&) noexcept = delete;
    Command &operator=(const Command &) noexcept = delete;

    [[nodiscard]] auto device() const noexcept { return _device; }

    virtual void commit(Stream &) = 0;
    virtual void recycle() = 0;
    
    virtual void finalize(Stream &) { /* do nothing by default... */ }
};

#define LUISA_MAKE_COMMAND_OVERRIDE(Type)                                           \
    void commit(Stream &stream) override;                                           \
    void recycle() override;                                                        \
                                                                                    \
    friend class Pool<Type>;                                                        \
    template<                                                                       \
        typename DeviceType, typename... Args,                                      \
        std::enable_if_t<std::is_base_of_v<Device, DeviceType>, int> = 0>           \
    [[nodiscard]] static auto create(DeviceType *device, Args &&...args) noexcept { \
        return device->command_pool().obtain(device, std::forward<Args>(args)...);  \
    }

class BufferUploadCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    const void *_data;

private:
    BufferUploadCommand(Device *device, uint64_t handle, size_t offset_bytes, size_t size_bytes, const void *data) noexcept
        : Command{device}, _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}

public:
    void finalize(Stream &stream) override;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_OVERRIDE(BufferUploadCommand)
};

class BufferDownloadCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    void *_data;

private:
    BufferDownloadCommand(Device *device, uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : Command{device}, _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_OVERRIDE(BufferDownloadCommand)
};

class BufferCopyCommand : public Command {

private:
    uint64_t _src_handle;
    uint64_t _dst_handle;
    size_t _src_offset;
    size_t _dst_offset;
    size_t _size;

private:
    BufferCopyCommand(Device *device, uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : Command{device}, _src_handle{src}, _dst_handle{dst}, _src_offset{src_offset}, _dst_offset{dst_offset}, _size{size} {}

public:
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    LUISA_MAKE_COMMAND_OVERRIDE(BufferCopyCommand)
};

#undef LUISA_MAKE_COMMAND_OVERRIDE

class CommandPool {

#define LUISA_MAKE_COMMAND_POOL(Type)                                                          \
private:                                                                                       \
    Pool<Type> _pool_##Type;                                                                   \
                                                                                               \
public:                                                                                        \
    template<typename T, typename... Args, std::enable_if_t<std::is_same_v<T, Type>, int> = 0> \
    [[nodiscard]] auto obtain(Args &&...args) noexcept {                                       \
        return _pool_##Type.obtain(std::forward<Args>(args)...);                               \
    }                                                                                          \
    void recycle(Type *cmd) noexcept { _pool_##Type.recycle(cmd); }

    LUISA_MAKE_COMMAND_POOL(BufferCopyCommand)
    LUISA_MAKE_COMMAND_POOL(BufferUploadCommand)
    LUISA_MAKE_COMMAND_POOL(BufferDownloadCommand)

#undef LUISA_MAKE_COMMAND_POOL
};

}// namespace luisa::compute
