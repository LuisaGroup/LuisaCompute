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

class BufferUploadCommand;
class BufferDownloadCommand;
class BufferCopyCommand;
class KernelLaunchCommand;

struct CommandVisitor {
    virtual void visit(const BufferCopyCommand *) noexcept = 0;
    virtual void visit(const BufferUploadCommand *) noexcept = 0;
    virtual void visit(const BufferDownloadCommand *) noexcept = 0;
    virtual void visit(const KernelLaunchCommand *) noexcept = 0;
};

#define LUISA_MAKE_COMMAND_ACCEPT_VISITOR() \
    void accept(CommandVisitor &visitor) const noexcept override { visitor.visit(this); }

#define LUISA_MAKE_COMMAND_CREATOR(Cmd)                                        \
    template<typename... Args>                                                 \
    [[nodiscard]] static auto create(Args &&...args) noexcept {                \
        auto t0 = std::chrono::high_resolution_clock::now();                   \
        auto cmd = std::unique_ptr<Cmd>{new Cmd{std::forward<Args>(args)...}}; \
        auto t1 = std::chrono::high_resolution_clock::now();                   \
        using namespace std::chrono_literals;                                  \
        LUISA_VERBOSE_WITH_LOCATION(                                           \
            "Created {} in {} ms.",                                            \
            #Cmd, (t1 - t0) / 1ns * 1e-6);                                     \
        return cmd;                                                            \
    }

class Command {

public:
    static constexpr auto max_resource_count = 64u;

    struct Resource {

        enum struct Tag : uint32_t {
            NONE,
            BUFFER,
            TEXTURE
        };

        enum struct Usage : uint32_t {
            NONE,
            READ = 1u,
            WRITE = 2u,
            READ_WRITE = READ | WRITE
        };

        uint64_t handle{0u};
        Tag tag{Tag::NONE};
        Usage usage{Usage::NONE};

        constexpr Resource() noexcept = default;
        constexpr Resource(uint64_t handle, Tag tag, Usage usage) noexcept
            : handle{handle}, tag{tag}, usage{usage} {}
    };

private:
    std::array<Resource, max_resource_count> _resource_slots{};
    size_t _resource_count{0u};

    void _use_resource(uint64_t handle, Resource::Tag tag, Resource::Usage usage) noexcept;

protected:
    void _buffer_read_only(uint64_t handle) noexcept;
    void _buffer_write_only(uint64_t handle) noexcept;
    void _buffer_read_write(uint64_t handle) noexcept;
    void _texture_read_only(uint64_t handle) noexcept;
    void _texture_write_only(uint64_t handle) noexcept;
    void _texture_read_write(uint64_t handle) noexcept;

public:
    [[nodiscard]] std::span<const Resource> resources() const noexcept;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    virtual ~Command() noexcept = default;
};

class BufferUploadCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    const void *_data;

private:
    BufferUploadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, const void *data) noexcept
        : _handle{handle},
          _offset{offset_bytes},
          _size{size_bytes},
          _data{data} { _buffer_write_only(_handle); }

public:
    LUISA_MAKE_COMMAND_CREATOR(BufferUploadCommand)
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_ACCEPT_VISITOR()
};

class BufferDownloadCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    void *_data;

private:
    BufferDownloadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : _handle{handle},
          _offset{offset_bytes},
          _size{size_bytes},
          _data{data} { _buffer_read_only(_handle); }

public:
    LUISA_MAKE_COMMAND_CREATOR(BufferDownloadCommand)
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_ACCEPT_VISITOR()
};

class BufferCopyCommand : public Command {

private:
    uint64_t _src_handle;
    uint64_t _dst_handle;
    size_t _src_offset;
    size_t _dst_offset;
    size_t _size;

private:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : _src_handle{src},
          _dst_handle{dst},
          _src_offset{src_offset},
          _dst_offset{dst_offset},
          _size{size} {
        _buffer_read_only(_src_handle);
        _buffer_write_only(_dst_handle);
    }

public:
    LUISA_MAKE_COMMAND_CREATOR(BufferCopyCommand)
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    LUISA_MAKE_COMMAND_ACCEPT_VISITOR()
};

// TODO...
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

class KernelLaunchCommand : public Command {

private:
    KernelArgumentEncoder _encoder;
    uint3 _dispatch_size;
    uint3 _block_size;
    uint32_t _kernel_uid;

private:
    KernelLaunchCommand(uint32_t kernel_uid, KernelArgumentEncoder encoder, uint3 dispatch_size, uint3 block_size) noexcept
        : _encoder{std::move(encoder)}, _dispatch_size{dispatch_size}, _block_size{block_size}, _kernel_uid{kernel_uid} {}

public:
    LUISA_MAKE_COMMAND_CREATOR(KernelLaunchCommand)
    [[nodiscard]] auto kernel_uid() const noexcept { return _kernel_uid; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }
    [[nodiscard]] auto dispatch_size() const noexcept { return _dispatch_size; }
    [[nodiscard]] auto arguments() const noexcept { return _encoder.arguments(); }
    LUISA_MAKE_COMMAND_ACCEPT_VISITOR()
};

#undef LUISA_MAKE_COMMAND_ACCEPT_VISITOR
#undef LUISA_MAKE_COMMAND_CREATOR

}// namespace luisa::compute
