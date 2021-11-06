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
#include <core/pool.h>
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
        AccelUpdateCommand,         \
        AccelBuildCommand,          \
        MeshUpdateCommand,          \
        MeshBuildCommand,           \
        BindlessArrayUpdateCommand

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
class CommandList;

namespace detail {

#define LUISA_MAKE_COMMAND_POOL_DECL(Cmd) \
    [[nodiscard]] Pool<Cmd> &pool_##Cmd() noexcept;
LUISA_MAP(LUISA_MAKE_COMMAND_POOL_DECL, LUISA_ALL_COMMANDS)
#undef LUISA_MAKE_COMMAND_POOL_DECL

}// namespace detail

#define LUISA_MAKE_COMMAND_COMMON_CREATE(Cmd)                                    \
    template<typename... Args>                                                   \
    [[nodiscard]] static auto create(Args &&...args) noexcept {                  \
        Clock clock;                                                             \
        auto command = detail::pool_##Cmd().create(std::forward<Args>(args)...); \
        LUISA_VERBOSE_WITH_LOCATION(                                             \
            "Created {} in {} ms.", #Cmd, clock.toc());                          \
        return command;                                                          \
    }

#define LUISA_MAKE_COMMAND_COMMON_ACCEPT(Cmd) \
    void accept(CommandVisitor &visitor) const noexcept override { visitor.visit(this); }

#define LUISA_MAKE_COMMAND_COMMON_RECYCLE(Cmd) \
    void _recycle() noexcept override { detail::pool_##Cmd().recycle(this); }

#define LUISA_MAKE_COMMAND_COMMON(Cmd)    \
    LUISA_MAKE_COMMAND_COMMON_CREATE(Cmd) \
    LUISA_MAKE_COMMAND_COMMON_ACCEPT(Cmd) \
    LUISA_MAKE_COMMAND_COMMON_RECYCLE(Cmd)

class Command {

public:
    struct Binding {

        enum struct Tag : uint32_t {
            NONE,
            BUFFER,
            TEXTURE,
            BINDLESS_ARRAY,
            ACCEL
        };

        uint64_t handle{0u};
        Tag tag{Tag::NONE};
        Usage usage{Usage::NONE};

        constexpr Binding(uint64_t handle, Tag tag, Usage usage) noexcept
            : handle{handle}, tag{tag}, usage{usage} {}
    };

private:
    std::vector<Binding> _resource_slots{};
    Command *_next_command{nullptr};

protected:
    void _use_resource(uint64_t handle, Binding::Tag tag, Usage usage) noexcept;
    void _buffer_read_only(uint64_t handle) noexcept;
    void _buffer_write_only(uint64_t handle) noexcept;
    void _buffer_read_write(uint64_t handle) noexcept;
    void _texture_read_only(uint64_t handle) noexcept;
    void _texture_write_only(uint64_t handle) noexcept;
    void _texture_read_write(uint64_t handle) noexcept;

private:
    [[nodiscard]] auto _next() const noexcept { return _next_command; }
    Command *_set_next(Command *cmd) noexcept { return cmd == nullptr ? this : (_next_command = cmd); }
    virtual void _recycle() noexcept = 0;

protected:
    ~Command() noexcept = default;

public:
    [[nodiscard]] std::span<const Binding> resources() const noexcept;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    [[nodiscard]] auto next() const noexcept { return _next(); }
    auto set_next(Command *cmd) noexcept { return _set_next(cmd); }
    void recycle();
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

public:
    BufferToTextureCopyCommand(uint64_t buffer, size_t buffer_offset,
                               uint64_t texture, PixelStorage storage,
                               uint level, uint3 offset, uint3 size) noexcept
        : _buffer_handle{buffer}, _buffer_offset{buffer_offset},
          _texture_handle{texture}, _pixel_storage{storage}, _texture_level{level},
          _texture_offset{offset.x, offset.y, offset.z},
          _texture_size{size.x, size.y, size.z} {
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

public:
    TextureCopyCommand(uint64_t src_handle,
                       uint64_t dst_handle,
                       uint src_level,
                       uint dst_level,
                       uint3 src_offset,
                       uint3 dst_offset,
                       uint3 size) noexcept
        : _src_handle{src_handle}, _dst_handle{dst_handle},
          _src_offset{src_offset.x, src_offset.y, src_offset.z},
          _dst_offset{dst_offset.x, dst_offset.y, dst_offset.z},
          _size{size.x, size.y, size.z},
          _src_level{src_level}, _dst_level{dst_level} {
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
    LUISA_MAKE_COMMAND_COMMON(TextureCopyCommand)
};

class TextureUploadCommand : public Command {

private:
    uint64_t _handle;
    PixelStorage _storage;
    uint _level;
    uint _offset[3];
    uint _size[3];
    const void *_data;

public:
    TextureUploadCommand(
        uint64_t handle, PixelStorage storage, uint level,
        uint3 offset, uint3 size, const void *data) noexcept
        : _handle{handle},
          _storage{storage},
          _level{level},
          _offset{offset.x, offset.y, offset.z},
          _size{size.x, size.y, size.z},
          _data{data} { _texture_write_only(_handle); }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto offset() const noexcept { return uint3(_offset[0], _offset[1], _offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto data() const noexcept { return _data; }
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
            BINDLESS_ARRAY,
            ACCEL,
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

    struct BindlessArrayArgument : Argument {
        uint64_t handle{};
        BindlessArrayArgument() noexcept : Argument{Tag::BINDLESS_ARRAY, 0u} {}
        BindlessArrayArgument(uint32_t vid, uint64_t handle) noexcept
            : Argument{Tag::BINDLESS_ARRAY, vid},
              handle{handle} {}
    };

    struct AccelArgument : Argument {
        uint64_t handle{};
        AccelArgument() noexcept : Argument{Tag::ACCEL, 0u} {}
        AccelArgument(uint32_t vid, uint64_t handle) noexcept
            : Argument{Tag::ACCEL, vid},
              handle{handle} {}
    };

    struct ArgumentBuffer : std::array<std::byte, 2048u> {};

private:
    uint64_t _handle;
    Function _kernel;
    size_t _argument_buffer_size{0u};
    uint _dispatch_size[3]{};
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
    //   4. captured acceleration structures
    //   4. arguments
    void encode_buffer(uint32_t variable_uid, uint64_t handle, size_t offset, Usage usage) noexcept;
    void encode_texture(uint32_t variable_uid, uint64_t handle, Usage usage) noexcept;
    void encode_uniform(uint32_t variable_uid, const void *data, size_t size, size_t alignment) noexcept;
    void encode_bindless_array(uint32_t variable_uid, uint64_t handle) noexcept;
    void encode_accel(uint32_t variable_uid, uint64_t handle) noexcept;

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
                case Argument::Tag::BINDLESS_ARRAY: {
                    BindlessArrayArgument arg;
                    std::memcpy(&arg, p, sizeof(BindlessArrayArgument));
                    visit(argument.variable_uid, arg);
                    p += sizeof(BindlessArrayArgument);
                    break;
                }
                case Argument::Tag::ACCEL: {
                    AccelArgument arg;
                    std::memcpy(&arg, p, sizeof(AccelArgument));
                    visit(argument.variable_uid, arg);
                    p += sizeof(AccelArgument);
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

enum struct AccelBuildHint {
    FAST_TRACE, // build with best quality
    FAST_UPDATE,// optimize for frequent update, usually with compaction
    FAST_REBUILD// optimize for frequent rebuild, maybe without compaction
};

class MeshBuildCommand : public Command {

private:
    uint64_t _handle;
    AccelBuildHint _hint;
    uint64_t _vertex_buffer_handle;
    size_t _vertex_buffer_offset;
    size_t _vertex_stride;
    size_t _vertex_count;
    uint64_t _triangle_buffer_handle;
    size_t _triangle_buffer_offset;
    size_t _triangle_count;

public:
    MeshBuildCommand(uint64_t handle, AccelBuildHint hint,
                     uint64_t v_handle, size_t v_offset, size_t v_stride, size_t v_count,
                     uint64_t t_handle, size_t t_offset, size_t t_count) noexcept
        : _handle{handle}, _hint{hint},
          _vertex_buffer_handle{v_handle}, _vertex_buffer_offset{v_offset}, _vertex_stride{v_stride}, _vertex_count{v_count},
          _triangle_buffer_handle{t_handle}, _triangle_buffer_offset{t_offset}, _triangle_count{t_count} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto vertex_buffer_handle() const noexcept { return _vertex_buffer_handle; }
    [[nodiscard]] auto vertex_buffer_offset() const noexcept { return _vertex_buffer_offset; }
    [[nodiscard]] auto vertex_stride() const noexcept { return _vertex_stride; }
    [[nodiscard]] auto vertex_count() const noexcept { return _vertex_count; }
    [[nodiscard]] auto triangle_buffer_handle() const noexcept { return _triangle_buffer_handle; }
    [[nodiscard]] auto triangle_buffer_offset() const noexcept { return _triangle_buffer_offset; }
    [[nodiscard]] auto triangle_count() const noexcept { return _triangle_count; }
    [[nodiscard]] auto hint() const noexcept { return _hint; }
    LUISA_MAKE_COMMAND_COMMON(MeshBuildCommand)
};

class MeshUpdateCommand : public Command {

private:
    uint64_t _handle;

public:
    explicit MeshUpdateCommand(uint64_t handle) noexcept : _handle{handle} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    LUISA_MAKE_COMMAND_COMMON(MeshUpdateCommand)
};

class AccelBuildCommand : public Command {

private:
    uint64_t _handle;
    AccelBuildHint _hint;
    std::span<const uint64_t> _instance_mesh_handles;
    std::span<const float4x4> _instance_transforms;

public:
    AccelBuildCommand(uint64_t handle, AccelBuildHint hint,
                      std::span<const uint64_t> instance_mesh_handles,
                      std::span<const float4x4> instance_transforms) noexcept
        : _handle{handle}, _hint{hint},
          _instance_mesh_handles{instance_mesh_handles},
          _instance_transforms{instance_transforms} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto hint() const noexcept { return _hint; }
    [[nodiscard]] auto instance_mesh_handles() const noexcept { return _instance_mesh_handles; }
    [[nodiscard]] auto instance_transforms() const noexcept { return _instance_transforms; }
    LUISA_MAKE_COMMAND_COMMON(AccelBuildCommand)
};

class AccelUpdateCommand : public Command {

private:
    uint64_t _handle;
    size_t _first_instance{0u};
    std::span<const float4x4> _instance_transforms;

public:
    explicit AccelUpdateCommand(uint64_t handle) noexcept : _handle{handle} {}
    AccelUpdateCommand(uint64_t handle, std::span<const float4x4> instance_transforms, size_t first) noexcept
        : _handle{handle},
          _first_instance{first},
          _instance_transforms{instance_transforms} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto should_update_transforms() const noexcept { return !_instance_transforms.empty(); }
    [[nodiscard]] auto first_instance_to_update() const noexcept { return _first_instance; }
    [[nodiscard]] auto updated_transforms() const noexcept { return _instance_transforms; }
    LUISA_MAKE_COMMAND_COMMON(AccelUpdateCommand)
};

class BindlessArrayUpdateCommand : public Command {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _count;

public:
    BindlessArrayUpdateCommand(uint64_t handle, size_t offset, size_t count) noexcept
        : _handle{handle}, _offset{offset}, _count{count} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto count() const noexcept { return _count; }
    LUISA_MAKE_COMMAND_COMMON(BindlessArrayUpdateCommand)
};

#undef LUISA_MAKE_COMMAND_COMMON_CREATE
#undef LUISA_MAKE_COMMAND_COMMON_ACCEPT
#undef LUISA_MAKE_COMMAND_COMMON_RECYCLE
#undef LUISA_MAKE_COMMAND_COMMON

}// namespace luisa::compute
