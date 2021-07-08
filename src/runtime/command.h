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

#include <core/clock.h>
#include <core/logging.h>
#include <core/basic_types.h>
#include <core/memory.h>
#include <ast/variable.h>
#include <ast/function.h>
#include <runtime/pixel.h>

namespace luisa::compute {

#define LUISA_ALL_COMMANDS          \
    BufferUploadCommand,            \
        BufferDownloadCommand,      \
        BufferCopyCommand,          \
        BufferToTextureCopyCommand, \
        ShaderDispatchCommand,      \
        TextureUploadCommand,       \
        TextureDownloadCommand,     \
        TextureCopyCommand,         \
        TextureToBufferCopyCommand, \
        AccelTraceClosestCommand,   \
        AccelTraceAnyCommand,       \
        AccelUpdateCommand

#define LUISA_MAKE_COMMAND_FWD_DECL(CMD) class CMD;
LUISA_MAP(LUISA_MAKE_COMMAND_FWD_DECL, LUISA_ALL_COMMANDS)
#undef LUISA_MAKE_COMMAND_FWD_DECL

struct CommandVisitor {
#define LUISA_MAKE_COMMAND_VISITOR_INTERFACE(CMD) \
    virtual void visit(const CMD *) noexcept = 0;
    LUISA_MAP(LUISA_MAKE_COMMAND_VISITOR_INTERFACE, LUISA_ALL_COMMANDS)
#undef LUISA_MAKE_COMMAND_VISITOR_INTERFACE
};

class Command;

namespace detail {

#define LUISA_MAKE_COMMAND_POOL_DECL(Cmd) \
    [[nodiscard]] Pool<Cmd> &pool_##Cmd() noexcept;
LUISA_MAP(LUISA_MAKE_COMMAND_POOL_DECL, LUISA_ALL_COMMANDS)
#undef LUISA_MAKE_COMMAND_POOL_DECL

class CommandRecycle : private CommandVisitor {

#define LUISA_MAKE_COMMAND_RECYCLE(CMD) \
    void visit(const CMD *command) noexcept override;
    LUISA_MAP(LUISA_MAKE_COMMAND_RECYCLE, LUISA_ALL_COMMANDS)
#undef LUISA_MAKE_COMMAND_RECYCLE

public:
    void operator()(class Command *command) noexcept;
};

}// namespace detail

using CommandHandle = std::unique_ptr<Command, detail::CommandRecycle>;

#define LUISA_MAKE_COMMAND_COMMON(Cmd)                                           \
    template<typename... Args>                                                   \
    [[nodiscard]] static auto create(Args &&...args) noexcept {                  \
        Clock clock;                                                             \
        auto command = detail::pool_##Cmd().create(std::forward<Args>(args)...); \
        LUISA_VERBOSE_WITH_LOCATION(                                             \
            "Created {} in {} ms.", #Cmd, clock.toc());                          \
        auto command_ptr = static_cast<Command *>(command.release());            \
        return CommandHandle{command_ptr};                                       \
    }                                                                            \
    void accept(CommandVisitor &visitor) const noexcept override {               \
        visitor.visit(this);                                                     \
    }

class Command {

public:
    static constexpr auto max_resource_count = 48u;

    struct Resource {

        enum struct Tag : uint32_t {
            NONE,
            BUFFER,
            TEXTURE
        };

        using Usage = Variable::Usage;

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

class BufferToTextureCopyCommand : public Command {

private:
    uint64_t _buffer_handle;
    size_t _buffer_offset;
    uint64_t _texture_handle;
    PixelStorage _pixel_storage;
    uint _texture_level;
    uint _texture_offset[3];
    uint _texture_size[3];
    uint64_t _heap;

public:
    BufferToTextureCopyCommand(uint64_t buffer, size_t buffer_offset,
                               uint64_t texture, PixelStorage storage,
                               uint level, uint3 offset, uint3 size,
                               uint64_t heap = std::numeric_limits<uint64_t>::max()) noexcept
        : _buffer_handle{buffer}, _buffer_offset{buffer_offset},
          _texture_handle{texture}, _pixel_storage{storage}, _texture_level{level},
          _texture_offset{offset.x, offset.y, offset.z},
          _texture_size{size.x, size.y, size.z},
          _heap{heap} {
        _buffer_read_only(_buffer_handle);
        _texture_write_only(_texture_handle);
    }
    [[nodiscard]] auto buffer() const noexcept { return _buffer_handle; }
    [[nodiscard]] auto buffer_offset() const noexcept { return _buffer_offset; }
    [[nodiscard]] auto texture() const noexcept { return _texture_handle; }
    [[nodiscard]] auto storage() const noexcept { return _pixel_storage; }
    [[nodiscard]] auto level() const noexcept { return _texture_level; }
    [[nodiscard]] auto offset() const noexcept { return uint3(_texture_offset[0], _texture_offset[1], _texture_offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_texture_size[0], _texture_size[1], _texture_size[2]); }
    [[nodiscard]] auto heap() const noexcept { return _heap; }
    [[nodiscard]] auto from_heap() const noexcept { return _heap != std::numeric_limits<uint64_t>::max(); }
    LUISA_MAKE_COMMAND_COMMON(BufferToTextureCopyCommand)
};

class TextureToBufferCopyCommand : public Command {

private:
    uint64_t _buffer_handle;
    size_t _buffer_offset;
    uint64_t _texture_handle;
    PixelStorage _pixel_storage;
    uint _texture_level;
    uint _texture_offset[3];
    uint _texture_size[3];

public:
    TextureToBufferCopyCommand(uint64_t buffer, size_t buffer_offset,
                               uint64_t texture, PixelStorage storage,
                               uint level, uint3 offset, uint3 size) noexcept
        : _buffer_handle{buffer}, _buffer_offset{buffer_offset},
          _texture_handle{texture}, _pixel_storage{storage}, _texture_level{level},
          _texture_offset{offset.x, offset.y, offset.z},
          _texture_size{size.x, size.y, size.z} {
        _texture_read_only(_texture_handle);
        _buffer_write_only(_buffer_handle);
    }
    [[nodiscard]] auto buffer() const noexcept { return _buffer_handle; }
    [[nodiscard]] auto buffer_offset() const noexcept { return _buffer_offset; }
    [[nodiscard]] auto texture() const noexcept { return _texture_handle; }
    [[nodiscard]] auto storage() const noexcept { return _pixel_storage; }
    [[nodiscard]] auto level() const noexcept { return _texture_level; }
    [[nodiscard]] auto offset() const noexcept { return uint3(_texture_offset[0], _texture_offset[1], _texture_offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_texture_size[0], _texture_size[1], _texture_size[2]); }
    LUISA_MAKE_COMMAND_COMMON(TextureToBufferCopyCommand)
};

class TextureCopyCommand : public Command {

private:
    uint64_t _src_handle;
    uint64_t _dst_handle;
    uint _src_offset[3];
    uint _dst_offset[3];
    uint _size[3];
    uint _src_level;
    uint _dst_level;
    size_t _heap;

public:
    TextureCopyCommand(uint64_t src_handle,
                       uint64_t dst_handle,
                       uint src_level,
                       uint dst_level,
                       uint3 src_offset,
                       uint3 dst_offset,
                       uint3 size,
                       uint64_t heap = std::numeric_limits<uint64_t>::max()) noexcept
        : _src_handle{src_handle}, _dst_handle{dst_handle},
          _src_offset{src_offset.x, src_offset.y, src_offset.z},
          _dst_offset{dst_offset.x, dst_offset.y, dst_offset.z},
          _size{size.x, size.y, size.z},
          _src_level{src_level}, _dst_level{dst_level},
          _heap{heap} {
        _texture_read_only(_src_handle);
        _texture_write_only(_dst_handle);
    }
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return uint3(_src_offset[0], _src_offset[1], _src_offset[2]); }
    [[nodiscard]] auto dst_offset() const noexcept { return uint3(_dst_offset[0], _dst_offset[1], _dst_offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto src_level() const noexcept { return _src_level; }
    [[nodiscard]] auto dst_level() const noexcept { return _dst_level; }
    [[nodiscard]] auto dst_heap() const noexcept { return _heap; }
    [[nodiscard]] auto dst_from_heap() const noexcept { return _heap != std::numeric_limits<uint64_t>::max(); }
    LUISA_MAKE_COMMAND_COMMON(TextureCopyCommand)
};

class TextureUploadCommand : public Command {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _level;
    uint _offset[3];
    uint _size[3];
    uint64_t _heap;
    const void *_data;

public:
    TextureUploadCommand(
        uint64_t handle, PixelStorage storage, uint level,
        uint3 offset, uint3 size, const void *data,
        uint64_t heap = std::numeric_limits<uint64_t>::max()) noexcept
        : _handle{handle},
          _storage{storage},
          _level{level},
          _offset{offset.x, offset.y, offset.z},
          _size{size.x, size.y, size.z},
          _heap{heap},
          _data{data} { _texture_write_only(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto offset() const noexcept { return uint3(_offset[0], _offset[1], _offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto data() const noexcept { return _data; }
    [[nodiscard]] auto heap() const noexcept { return _heap; }
    [[nodiscard]] auto from_heap() const noexcept { return _heap != std::numeric_limits<uint64_t>::max(); }
    LUISA_MAKE_COMMAND_COMMON(TextureUploadCommand)
};

class TextureDownloadCommand : public Command {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _level;
    uint _offset[3];
    uint _size[3];
    void *_data;

public:
    TextureDownloadCommand(
        uint64_t handle, PixelStorage storage, uint level,
        uint3 offset, uint3 size, void *data) noexcept
        : _handle{handle},
          _storage{storage},
          _level{level},
          _offset{offset.x, offset.y, offset.z},
          _size{size.x, size.y, size.z},
          _data{data} { _texture_read_only(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto offset() const noexcept { return uint3(_offset[0], _offset[1], _offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(TextureDownloadCommand)
};

namespace detail {
class FunctionBuilder;
}

class ShaderDispatchCommand : public Command {

public:
    struct alignas(16) Argument {

        enum struct Tag : uint32_t {
            BUFFER,
            TEXTURE,
            UNIFORM,
            TEXTURE_HEAP
        };

        Tag tag;
        uint32_t variable_uid;

        Argument() noexcept = default;
        Argument(Tag tag, uint32_t vid) noexcept
            : tag{tag}, variable_uid{vid} {}
    };

    struct BufferArgument : Argument {
        uint64_t handle{};
        size_t offset{};
        BufferArgument() noexcept : Argument{Tag::BUFFER, 0u} {}
        BufferArgument(uint32_t vid, uint64_t handle, size_t offset) noexcept
            : Argument{Tag::BUFFER, vid},
              handle{handle},
              offset{offset} {}
    };

    struct TextureArgument : Argument {
        uint64_t handle{};
        TextureArgument() noexcept : Argument{Tag::TEXTURE, 0u} {}
        TextureArgument(uint32_t vid, uint64_t handle) noexcept
            : Argument{Tag::TEXTURE, vid},
              handle{handle} {}
    };

    struct UniformArgument : Argument {
        size_t size{};
        size_t alignment{};
        UniformArgument() noexcept : Argument{Tag::UNIFORM, 0u} {}
        UniformArgument(uint32_t vid, size_t size, size_t alignment) noexcept
            : Argument{Tag::UNIFORM, vid},
              size{size},
              alignment{alignment} {}
    };

    struct TextureHeapArgument : Argument {
        uint64_t handle{};
        TextureHeapArgument() noexcept : Argument{Tag::TEXTURE_HEAP, 0u} {}
        TextureHeapArgument(uint32_t vid, uint64_t handle) noexcept
            : Argument{Tag::TEXTURE_HEAP, vid},
              handle{handle} {}
    };

    struct ArgumentBuffer : std::array<std::byte, 2048u> {};

private:
    uint64_t _handle;
    Function _kernel;
    size_t _argument_buffer_size{0u};
    uint _dispatch_size[3]{};
    uint _block_size[3]{};
    uint32_t _argument_count{0u};
    ArgumentBuffer _argument_buffer{};

public:
    explicit ShaderDispatchCommand(uint64_t handle, Function kernel) noexcept;
    void set_dispatch_size(uint3 launch_size) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto kernel() const noexcept { return _kernel; }
    [[nodiscard]] auto argument_count() const noexcept { return static_cast<size_t>(_argument_count); }
    [[nodiscard]] auto dispatch_size() const noexcept { return uint3(_dispatch_size[0], _dispatch_size[1], _dispatch_size[2]); }

    // Note: encode/decode order:
    //   1. captured buffers
    //   2. captured textures
    //   3. captured texture heaps
    //   3. arguments
    void encode_buffer(uint32_t variable_uid, uint64_t handle, size_t offset, Resource::Usage usage) noexcept;
    void encode_texture(uint32_t variable_uid, uint64_t handle, Resource::Usage usage) noexcept;
    void encode_uniform(uint32_t variable_uid, const void *data, size_t size, size_t alignment) noexcept;
    void encode_texture_heap(uint32_t variable_uid, uint64_t handle) noexcept;

    template<typename Visit>
    void decode(Visit &&visit) const noexcept {
        auto p = _argument_buffer.data();
        while (p < _argument_buffer.data() + _argument_buffer_size) {
            Argument argument{};
            std::memcpy(&argument, p, sizeof(Argument));
            switch (argument.tag) {
                case Argument::Tag::BUFFER: {
                    BufferArgument buffer_argument{};
                    std::memcpy(&buffer_argument, p, sizeof(BufferArgument));
                    visit(argument.variable_uid, buffer_argument);
                    p += sizeof(BufferArgument);
                    break;
                }
                case Argument::Tag::TEXTURE: {
                    TextureArgument texture_argument{};
                    std::memcpy(&texture_argument, p, sizeof(TextureArgument));
                    visit(argument.variable_uid, texture_argument);
                    p += sizeof(TextureArgument);
                    break;
                }
                case Argument::Tag::UNIFORM: {
                    UniformArgument uniform_argument{};
                    std::memcpy(&uniform_argument, p, sizeof(UniformArgument));
                    p += sizeof(UniformArgument);
                    std::span data{p, uniform_argument.size};
                    visit(argument.variable_uid, data);
                    p += uniform_argument.size;
                    break;
                }
                case Argument::Tag::TEXTURE_HEAP: {
                    TextureHeapArgument arg;
                    std::memcpy(&arg, p, sizeof(TextureHeapArgument));
                    visit(argument.variable_uid, arg);
                    p += sizeof(TextureHeapArgument);
                    break;
                }
                default: {
                    LUISA_ERROR_WITH_LOCATION("Invalid argument.");
                    break;
                }
            }
        }
    }
    LUISA_MAKE_COMMAND_COMMON(ShaderDispatchCommand)
};

class AccelUpdateCommand : public Command {

public:
    LUISA_MAKE_COMMAND_COMMON(AccelUpdateCommand)
};

class AccelTraceClosestCommand : public Command {

public:
    LUISA_MAKE_COMMAND_COMMON(AccelTraceClosestCommand)
};

class AccelTraceAnyCommand : public Command {

public:
    LUISA_MAKE_COMMAND_COMMON(AccelTraceAnyCommand)
};

#undef LUISA_MAKE_COMMAND_COMMON

}// namespace luisa::compute
