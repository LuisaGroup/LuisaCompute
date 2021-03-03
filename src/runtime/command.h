//
// Created by Mike Smith on 2021/3/3.
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <variant>
#include <vector>
#include <span>

#include <core/data_types.h>

namespace luisa::compute {

class BufferUploadCommand {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    const void *_data;

private:
    BufferUploadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, const void *data) noexcept
        : _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}

public:
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

private:
    BufferDownloadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}

public:
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

private:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : _src_handle{src}, _dst_handle{dst}, _src_offset{src_offset}, _dst_offset{dst_offset}, _size{size} {}

public:
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
};

class KernelLaunchCommand {

public:
    struct BufferArgument {
        uint64_t handle;
        size_t offset;
    };
    
    struct TextureArgument {
        uint64_t handle;
    };
    
    struct UniformArgument {
        std::span<uint8_t> data;
    };
    
    using Argument = std::variant<BufferArgument, TextureArgument, UniformArgument>;

private:
    uint3 _block_size;
    uint3 _grid_size;
    uint32_t _kernel_uid;
    std::vector<Argument> _arguments;

public:
    KernelLaunchCommand(uint32_t kernel_uid, uint3 block_size, uint3 grid_size, std::vector<Argument> args) noexcept
        : _block_size{block_size}, _grid_size{grid_size}, _kernel_uid{kernel_uid}, _arguments{std::move(args)} {}
    [[nodiscard]] auto kernel_uid() const noexcept { return _kernel_uid; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto grid_size() const noexcept { return _grid_size; }
    [[nodiscard]] std::span<const Argument> arguments() const noexcept;
};

}
