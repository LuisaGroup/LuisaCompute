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
#include <core/memory.h>

namespace luisa::compute {

class Command;
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

namespace detail {

#define LUISA_MAKE_COMMAND_POOL_DECL(Cmd) \
    [[nodiscard]] Pool<Cmd> &pool_##Cmd() noexcept;
LUISA_MAKE_COMMAND_POOL_DECL(BufferCopyCommand)
LUISA_MAKE_COMMAND_POOL_DECL(BufferUploadCommand)
LUISA_MAKE_COMMAND_POOL_DECL(BufferDownloadCommand)
LUISA_MAKE_COMMAND_POOL_DECL(KernelLaunchCommand)
#undef LUISA_MAKE_COMMAND_POOL_DECL

class CommandRecycle : private CommandVisitor {

private:
    void visit(const BufferCopyCommand *command) noexcept override;
    void visit(const BufferUploadCommand *command) noexcept override;
    void visit(const BufferDownloadCommand *command) noexcept override;
    void visit(const KernelLaunchCommand *command) noexcept override;

public:
    void operator()(Command *command) noexcept;
};

}// namespace detail

using CommandHandle = std::unique_ptr<Command, detail::CommandRecycle>;

#define LUISA_MAKE_COMMAND_ACCEPT_VISITOR()                        \
    void accept(CommandVisitor &visitor) const noexcept override { \
        visitor.visit(this);                                       \
    }

#define LUISA_MAKE_COMMAND_CREATOR(Cmd)                                          \
    template<typename... Args>                                                   \
    [[nodiscard]] static auto create(Args &&...args) noexcept {                  \
        auto t0 = std::chrono::high_resolution_clock::now();                     \
        auto command = detail::pool_##Cmd().create(std::forward<Args>(args)...); \
        auto t1 = std::chrono::high_resolution_clock::now();                     \
        using namespace std::chrono_literals;                                    \
        LUISA_VERBOSE_WITH_LOCATION(                                             \
            "Created {} in {} ms.", #Cmd, (t1 - t0) / 1ns * 1e-6);               \
        auto command_ptr = static_cast<Command *>(command.release());            \
        return CommandHandle{command_ptr};                                       \
    }

#define LUISA_MAKE_COMMAND_COMMON(Cmd) \
    LUISA_MAKE_COMMAND_CREATOR(Cmd)    \
    LUISA_MAKE_COMMAND_ACCEPT_VISITOR()

class Command {

public:
    static constexpr auto max_resource_count = 48u;

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

protected:
    void _use_resource(uint64_t handle, Resource::Tag tag, Resource::Usage usage) noexcept;
    void _buffer_read_only(uint64_t handle) noexcept;
    void _buffer_write_only(uint64_t handle) noexcept;
    void _buffer_read_write(uint64_t handle) noexcept;
    void _texture_read_only(uint64_t handle) noexcept;
    void _texture_write_only(uint64_t handle) noexcept;
    void _texture_read_write(uint64_t handle) noexcept;

protected:
    ~Command() noexcept = default;

public:
    [[nodiscard]] std::span<const Resource> resources() const noexcept;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
};

class BufferUploadCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    const void *_data;

public:
    BufferUploadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, const void *data) noexcept
        : _handle{handle},
          _offset{offset_bytes},
          _size{size_bytes},
          _data{data} { _buffer_write_only(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(BufferUploadCommand)
};

class BufferDownloadCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    void *_data;

public:
    BufferDownloadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : _handle{handle},
          _offset{offset_bytes},
          _size{size_bytes},
          _data{data} { _buffer_read_only(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(BufferDownloadCommand)
};

class BufferCopyCommand : public Command {

private:
    uint64_t _src_handle;
    uint64_t _dst_handle;
    size_t _src_offset;
    size_t _dst_offset;
    size_t _size;

public:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : _src_handle{src},
          _dst_handle{dst},
          _src_offset{src_offset},
          _dst_offset{dst_offset},
          _size{size} {
        _buffer_read_only(_src_handle);
        _buffer_write_only(_dst_handle);
    }
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    LUISA_MAKE_COMMAND_COMMON(BufferCopyCommand)
};

class KernelLaunchCommand : public Command {

public:
    struct Argument {
        enum struct Tag : uint32_t {
            BUFFER,
            TEXTURE,
            UNIFORM
        };
        Tag tag;
    };

    struct BufferArgument : Argument {
        uint64_t handle;
        size_t offset;
        Resource::Usage usage;
    };

    struct TextureArgument : Argument {
        uint64_t handle;
        Resource::Usage usage;
    };

    struct UniformArgument : Argument {
        size_t size;
        size_t alignment;
    };

    struct ArgumentBuffer : std::array<std::byte, 2048u> {};

private:
    uint32_t _kernel_uid;
    uint32_t _argument_count{0u};
    size_t _argument_buffer_size{0u};
    uint3 _dispatch_size;
    uint3 _block_size;
    ArgumentBuffer _argument_buffer{};

public:
    explicit KernelLaunchCommand(uint32_t uid) noexcept : _kernel_uid{uid} {}
    void set_launch_size(uint3 dispatch_size, uint3 block_size) noexcept;
    [[nodiscard]] auto kernel_uid() const noexcept { return _kernel_uid; }
    [[nodiscard]] auto argument_count() const noexcept { return static_cast<size_t>(_argument_count); }
    [[nodiscard]] auto dispatch_size() const noexcept { return _dispatch_size; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }

    // Note: encode/decode order:
    //   1. captured buffers
    //   2. captured textures
    //   3. arguments
    void encode_buffer(uint64_t handle, size_t offset, Resource::Usage usage) noexcept;
    // TODO: encode texture
    void encode_uniform(const void *data, size_t size, size_t alignment) noexcept;
    template<typename T>
    void static_memcpy(T *dest, void const *source) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            *dest = reinterpret_cast<T const *>(source);
        } else {
            memcpy(dest, source, sizeof(T));
        }
    }
    template<typename Visit>
    void decode(Visit &&visit) const noexcept {
        auto p = _argument_buffer.data();
        while (p < _argument_buffer.data() + _argument_buffer_size) {
            Argument argument{};
            static_memcpy(&argument, p);
            switch (argument.tag) {
                case Argument::Tag::BUFFER: {
                    BufferArgument buffer_argument{};
                    static_memcpy(&buffer_argument, p);
                    decode(buffer_argument);
                    p += sizeof(BufferArgument);
                    break;
                }
                case Argument::Tag::TEXTURE: {
                    TextureArgument texture_argument{};
                    static_memcpy(&texture_argument, p);
                    decode(texture_argument);
                    p += sizeof(TextureArgument);
                    break;
                }
                case Argument::Tag::UNIFORM: {
                    UniformArgument uniform_argument{};
                    static_memcpy(&uniform_argument, p);
                    p += sizeof(UniformArgument);
                    std::span data{p, uniform_argument.size};
                    visit(data);
                    p += uniform_argument.size;
                    break;
                }
                default: {
                    LUISA_ERROR_WITH_LOCATION("Invalid argument.");
                    break;
                }
            }
        }
    }
    LUISA_MAKE_COMMAND_COMMON(KernelLaunchCommand)
};

#undef LUISA_MAKE_COMMAND_ACCEPT_VISITOR
#undef LUISA_MAKE_COMMAND_CREATOR
#undef LUISA_MAKE_COMMAND_COMMON

namespace detail {

}// namespace detail

}// namespace luisa::compute
