//
// Created by Mike Smith on 2021/3/3.
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <variant>
#include <memory>
#include <vector>
#include <array>
#include <span>

#include <core/logging.h>
#include <core/basic_types.h>

namespace luisa::compute {

class BufferUploadCommand {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    const void *_data;

public:
    BufferUploadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, const void *data) noexcept
        : _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
};

class BufferDownloadCommand {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    void *_data;

public:
    BufferDownloadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
};

class BufferCopyCommand {

private:
    uint64_t _src_handle;
    uint64_t _dst_handle;
    size_t _src_offset;
    size_t _dst_offset;
    size_t _size;

public:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : _src_handle{src}, _dst_handle{dst}, _src_offset{src_offset}, _dst_offset{dst_offset}, _size{size} {}
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
};

class KernelArgumentEncoder {

public:
    using Data = std::array<std::byte, 1024u>;
    struct Deleter {
        void operator()(Data *ptr) const noexcept;
    };
    using Storage = std::unique_ptr<Data, Deleter>;

    struct BufferArgument {
        uint64_t handle;
        size_t offset_bytes;
    };

    struct TextureArgument {
        uint64_t handle;
    };

    struct UniformArgument {
        std::span<const std::byte> data;
    };

    using Argument = std::variant<BufferArgument, TextureArgument, UniformArgument>;

private:
    std::vector<Argument> _arguments;
    Storage _storage;
    std::byte *_ptr;
    [[nodiscard]] static std::vector<std::unique_ptr<Data>> &_available_blocks() noexcept;
    [[nodiscard]] static Storage _allocate() noexcept;
    static void _recycle(Data *storage) noexcept;

public:
    KernelArgumentEncoder() noexcept : _storage{_allocate()}, _ptr{_storage->data()} {}
    void encode_buffer(uint64_t handle, size_t offset_bytes) noexcept;
    void encode_uniform(const void *data, size_t size, size_t alignment) noexcept;
    [[nodiscard]] std::span<const Argument> arguments() const noexcept;
    [[nodiscard]] std::span<const std::byte> uniform_data() const noexcept;
};

class KernelLaunchCommand {

private:
    KernelArgumentEncoder _encoder;
    uint3 _dispatch_size;
    uint3 _block_size;
    uint32_t _kernel_uid;

public:
    KernelLaunchCommand(uint32_t kernel_uid, KernelArgumentEncoder encoder, uint3 dispatch_size, uint3 block_size) noexcept
        : _encoder{std::move(encoder)}, _dispatch_size{dispatch_size}, _block_size{block_size}, _kernel_uid{kernel_uid} {}
    [[nodiscard]] auto kernel_uid() const noexcept { return _kernel_uid; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto dispatch_size() const noexcept { return _dispatch_size; }
    [[nodiscard]] auto arguments() const noexcept { return _encoder.arguments(); }
};

struct SynchronizeCommand {};

[[nodiscard]] constexpr auto synchronize() noexcept { return SynchronizeCommand{}; }

}// namespace luisa::compute
