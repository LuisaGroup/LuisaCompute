#pragma once

#include <luisa/core/macro.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/variant.h>
#include <luisa/ast/usage.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/stream_tag.h>
#include <luisa/runtime/rhi/sampler.h>
#include <luisa/runtime/rhi/argument.h>
#include <luisa/runtime/rhi/curve_basis.h>
#include <luisa/runtime/rtx/motion_transform.h>

// for validation
namespace lc::validation {
class Stream;
}// namespace lc::validation

namespace luisa::compute {

struct IndirectDispatchArg {
    uint64_t handle;
    uint32_t offset;
    uint32_t max_dispatch_size;
};

#define LUISA_COMPUTE_RUNTIME_COMMANDS   \
    BufferUploadCommand,                 \
        BufferDownloadCommand,           \
        BufferCopyCommand,               \
        BufferToTextureCopyCommand,      \
        ShaderDispatchCommand,           \
        TextureUploadCommand,            \
        TextureDownloadCommand,          \
        TextureCopyCommand,              \
        TextureToBufferCopyCommand,      \
        AccelBuildCommand,               \
        MeshBuildCommand,                \
        CurveBuildCommand,               \
        ProceduralPrimitiveBuildCommand, \
        MotionInstanceBuildCommand,      \
        BindlessArrayUpdateCommand,      \
        CustomCommand

#define LUISA_MAKE_COMMAND_FWD_DECL(CMD) class CMD;
LUISA_MAP(LUISA_MAKE_COMMAND_FWD_DECL, LUISA_COMPUTE_RUNTIME_COMMANDS)
#undef LUISA_MAKE_COMMAND_FWD_DECL

struct CommandVisitor {
#define LUISA_MAKE_COMMAND_VISITOR_INTERFACE(CMD) \
    virtual void visit(const CMD *) noexcept = 0;
    LUISA_MAP(LUISA_MAKE_COMMAND_VISITOR_INTERFACE, LUISA_COMPUTE_RUNTIME_COMMANDS)
#undef LUISA_MAKE_COMMAND_VISITOR_INTERFACE
    virtual ~CommandVisitor() noexcept = default;
};

struct MutableCommandVisitor {
#define LUISA_MAKE_COMMAND_VISITOR_INTERFACE(CMD) \
    virtual void visit(CMD *) noexcept = 0;
    LUISA_MAP(LUISA_MAKE_COMMAND_VISITOR_INTERFACE, LUISA_COMPUTE_RUNTIME_COMMANDS)
#undef LUISA_MAKE_COMMAND_VISITOR_INTERFACE
    virtual ~MutableCommandVisitor() noexcept = default;
};

class Command;
class CommandList;

#define LUISA_MAKE_COMMAND_COMMON_ACCEPT()                                                \
    void accept(CommandVisitor &visitor) const noexcept override { visitor.visit(this); } \
    void accept(MutableCommandVisitor &visitor) noexcept override { visitor.visit(this); }

#define LUISA_MAKE_COMMAND_COMMON(Type) \
    LUISA_MAKE_COMMAND_COMMON_ACCEPT()  \
    StreamTag stream_tag() const noexcept override { return Type; }

class Command {

public:
    enum struct Tag {
#define LUISA_MAKE_COMMAND_TAG(Cmd) E##Cmd,
        LUISA_MAP(LUISA_MAKE_COMMAND_TAG, LUISA_COMPUTE_RUNTIME_COMMANDS)
#undef LUISA_MAKE_COMMAND_TAG
    };

private:
    Tag _tag;

public:
    explicit Command(Tag tag) noexcept : _tag(tag) {}
    virtual ~Command() noexcept = default;
    virtual void accept(CommandVisitor &visitor) const noexcept = 0;
    virtual void accept(MutableCommandVisitor &visitor) noexcept = 0;
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] virtual StreamTag stream_tag() const noexcept = 0;
};

class ShaderDispatchCommandBase {

public:
    using Argument = luisa::compute::Argument;

private:
    uint64_t _handle;
    luisa::vector<std::byte> _argument_buffer;
    size_t _argument_count;

protected:
    ShaderDispatchCommandBase(uint64_t shader_handle,
                              luisa::vector<std::byte> &&argument_buffer,
                              size_t argument_count) noexcept
        : _handle{shader_handle},
          _argument_buffer{std::move(argument_buffer)},
          _argument_count{argument_count} {}
    ~ShaderDispatchCommandBase() noexcept = default;

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto arguments() const noexcept {
        return luisa::span{reinterpret_cast<const Argument *>(_argument_buffer.data()), _argument_count};
    }
    [[nodiscard]] auto uniform(const Argument::Uniform &u) const noexcept {
        return luisa::span{_argument_buffer}.subspan(u.offset, u.size);
    }
};

class ShaderDispatchCommand final : public Command, public ShaderDispatchCommandBase {

public:
    using DispatchSize = luisa::variant<
        uint3,              // single dispatch
        IndirectDispatchArg,// indirect dispatch
        luisa::vector<uint3>// batched dispatch
        >;

private:
    DispatchSize _dispatch_size;

public:
    ShaderDispatchCommand(uint64_t shader_handle,
                          luisa::vector<std::byte> &&argument_buffer,
                          size_t argument_count,
                          DispatchSize dispatch_size) noexcept
        : Command{Tag::EShaderDispatchCommand},
          ShaderDispatchCommandBase{shader_handle,
                                    std::move(argument_buffer),
                                    argument_count},
          _dispatch_size{std::move(dispatch_size)} {}
    ShaderDispatchCommand(ShaderDispatchCommand const &) = delete;
    ShaderDispatchCommand(ShaderDispatchCommand &&) noexcept = default;
    [[nodiscard]] auto is_multiple_dispatch() const noexcept { return luisa::holds_alternative<luisa::vector<uint3>>(_dispatch_size); }
    [[nodiscard]] auto is_indirect() const noexcept { return luisa::holds_alternative<IndirectDispatchArg>(_dispatch_size); }
    [[nodiscard]] auto dispatch_size() const noexcept { return luisa::get<uint3>(_dispatch_size); }
    [[nodiscard]] auto indirect_dispatch() const noexcept { return luisa::get<IndirectDispatchArg>(_dispatch_size); }
    [[nodiscard]] luisa::span<const uint3> dispatch_sizes() const noexcept { return luisa::get<luisa::vector<uint3>>(_dispatch_size); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class BufferUploadCommand final : public Command {

private:
    uint64_t _handle{};
    size_t _offset{};
    size_t _size{};
    const void *_data{};

private:
    BufferUploadCommand() noexcept
        : Command{Command::Tag::EBufferUploadCommand} {}

public:
    BufferUploadCommand(uint64_t handle,
                        size_t offset_bytes,
                        size_t size_bytes,
                        const void *data) noexcept
        : Command{Command::Tag::EBufferUploadCommand},
          _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class BufferDownloadCommand final : public Command {

private:
    uint64_t _handle{};
    size_t _offset{};
    size_t _size{};
    void *_data{};

private:
    BufferDownloadCommand() noexcept
        : Command{Command::Tag::EBufferDownloadCommand} {}

public:
    BufferDownloadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : Command{Command::Tag::EBufferDownloadCommand},
          _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class BufferCopyCommand final : public Command {

private:
    uint64_t _src_handle{};
    uint64_t _dst_handle{};
    size_t _src_offset{};
    size_t _dst_offset{};
    size_t _size{};

private:
    BufferCopyCommand() noexcept
        : Command{Command::Tag::EBufferCopyCommand} {}

public:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : Command{Command::Tag::EBufferCopyCommand},
          _src_handle{src}, _dst_handle{dst},
          _src_offset{src_offset}, _dst_offset{dst_offset}, _size{size} {}
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class BufferToTextureCopyCommand final : public Command {

private:
    uint64_t _buffer_handle{};
    size_t _buffer_offset{};
    uint64_t _texture_handle{};
    PixelStorage _pixel_storage{};
    uint _texture_level{};
    uint _texture_offset[3]{};
    uint _texture_size[3]{};

private:
    BufferToTextureCopyCommand() noexcept
        : Command{Command::Tag::EBufferToTextureCopyCommand} {}

public:
    BufferToTextureCopyCommand(uint64_t buffer, size_t buffer_offset,
                               uint64_t texture, PixelStorage storage,
                               uint level, uint3 size, uint3 texture_offset = uint3::zero()) noexcept
        : Command{Command::Tag::EBufferToTextureCopyCommand},
          _buffer_handle{buffer}, _buffer_offset{buffer_offset},
          _texture_handle{texture}, _pixel_storage{storage}, _texture_level{level},
          _texture_offset{texture_offset.x, texture_offset.y, texture_offset.z},
          _texture_size{size.x, size.y, size.z} {}
    [[nodiscard]] auto buffer() const noexcept { return _buffer_handle; }
    [[nodiscard]] auto buffer_offset() const noexcept { return _buffer_offset; }
    [[nodiscard]] auto texture() const noexcept { return _texture_handle; }
    [[nodiscard]] auto storage() const noexcept { return _pixel_storage; }
    [[nodiscard]] auto level() const noexcept { return _texture_level; }
    [[nodiscard]] auto texture_offset() const noexcept { return uint3(_texture_offset[0], _texture_offset[1], _texture_offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_texture_size[0], _texture_size[1], _texture_size[2]); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class TextureToBufferCopyCommand final : public Command {

private:
    uint64_t _buffer_handle{};
    size_t _buffer_offset{};
    uint64_t _texture_handle{};
    PixelStorage _pixel_storage{};
    uint _texture_level{};
    uint _texture_offset[3]{};
    uint _texture_size[3]{};

private:
    TextureToBufferCopyCommand() noexcept
        : Command{Command::Tag::ETextureToBufferCopyCommand} {}

public:
    TextureToBufferCopyCommand(uint64_t buffer, size_t buffer_offset,
                               uint64_t texture, PixelStorage storage,
                               uint level, uint3 size, uint3 texture_offset = uint3::zero()) noexcept
        : Command{Command::Tag::ETextureToBufferCopyCommand},
          _buffer_handle{buffer}, _buffer_offset{buffer_offset},
          _texture_handle{texture}, _pixel_storage{storage}, _texture_level{level},
          _texture_offset{texture_offset.x, texture_offset.y, texture_offset.z},
          _texture_size{size.x, size.y, size.z} {}
    [[nodiscard]] auto buffer() const noexcept { return _buffer_handle; }
    [[nodiscard]] auto buffer_offset() const noexcept { return _buffer_offset; }
    [[nodiscard]] auto texture() const noexcept { return _texture_handle; }
    [[nodiscard]] auto storage() const noexcept { return _pixel_storage; }
    [[nodiscard]] auto level() const noexcept { return _texture_level; }
    [[nodiscard]] auto texture_offset() const noexcept { return uint3(_texture_offset[0], _texture_offset[1], _texture_offset[2]); }
    [[nodiscard]] auto size() const noexcept { return uint3(_texture_size[0], _texture_size[1], _texture_size[2]); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class TextureCopyCommand final : public Command {

private:
    PixelStorage _storage{};
    uint64_t _src_handle{};
    uint64_t _dst_handle{};
    uint _src_offset[3]{};
    uint _dst_offset[3]{};
    uint _size[3]{};
    uint _src_level{};
    uint _dst_level{};

private:
    TextureCopyCommand() noexcept
        : Command{Command::Tag::ETextureCopyCommand} {}

public:
    TextureCopyCommand(PixelStorage storage, uint64_t src_handle, uint64_t dst_handle,
                       uint src_level, uint dst_level, uint3 size, uint3 src_offset = uint3::zero(), uint3 dst_offset = uint3::zero()) noexcept
        : Command{Command::Tag::ETextureCopyCommand},
          _storage{storage}, _src_handle{src_handle}, _dst_handle{dst_handle},
          _src_offset{src_offset.x, src_offset.y, src_offset.z},
          _dst_offset{dst_offset.x, dst_offset.y, dst_offset.z},
          _size{size.x, size.y, size.z}, _src_level{src_level}, _dst_level{dst_level} {}
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto src_level() const noexcept { return _src_level; }
    [[nodiscard]] auto src_offset() const noexcept { return uint3(_src_offset[0], _src_offset[1], _src_offset[2]); }
    [[nodiscard]] auto dst_offset() const noexcept { return uint3(_dst_offset[0], _dst_offset[1], _dst_offset[2]); }
    [[nodiscard]] auto dst_level() const noexcept { return _dst_level; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class TextureUploadCommand final : public Command {

private:
    uint64_t _handle{};
    PixelStorage _storage{};
    uint _level{};
    // only for sparse texture
    uint _offset[3]{};
    uint _size[3]{};
    const void *_data{};

private:
    TextureUploadCommand() noexcept
        : Command{Command::Tag::ETextureUploadCommand} {}

public:
    TextureUploadCommand(uint64_t handle, PixelStorage storage,
                         uint level, uint3 size, const void *data, uint3 offset = uint3::zero()) noexcept
        : Command{Command::Tag::ETextureUploadCommand},
          _handle{handle}, _storage{storage}, _level{level},
          _offset{offset.x, offset.y, offset.z},
          _size{size.x, size.y, size.z},
          _data{data} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto offset() const noexcept { return uint3(_offset[0], _offset[1], _offset[2]); }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

class TextureDownloadCommand final : public Command {

private:
    uint64_t _handle{};
    PixelStorage _storage{};
    uint _level{};
    // only for sparse texture
    uint _offset[3]{};
    uint _size[3]{};
    void *_data{};

private:
    TextureDownloadCommand() noexcept
        : Command{Command::Tag::ETextureDownloadCommand} {}

public:
    TextureDownloadCommand(uint64_t handle, PixelStorage storage,
                           uint level, uint3 size, void *data, uint3 offset = uint3::zero()) noexcept
        : Command{Command::Tag::ETextureDownloadCommand},
          _handle{handle}, _storage{storage}, _level{level},
          _offset{offset.x, offset.y, offset.z},
          _size{size.x, size.y, size.z},
          _data{data} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto storage() const noexcept { return _storage; }
    [[nodiscard]] auto level() const noexcept { return _level; }
    [[nodiscard]] auto size() const noexcept { return uint3(_size[0], _size[1], _size[2]); }
    [[nodiscard]] auto offset() const noexcept { return uint3(_offset[0], _offset[1], _offset[2]); }
    [[nodiscard]] auto data() const noexcept { return _data; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COPY)
};

enum struct AccelBuildRequest : uint32_t {
    PREFER_UPDATE,
    FORCE_BUILD,
};

class MeshBuildCommand final : public Command {

private:
    uint64_t _handle{};
    AccelBuildRequest _request{};
    uint64_t _vertex_buffer{};
    size_t _vertex_buffer_offset{};
    size_t _vertex_buffer_size{};
    size_t _vertex_stride{};
    uint64_t _triangle_buffer{};
    size_t _triangle_buffer_offset{};
    size_t _triangle_buffer_size{};

private:
    MeshBuildCommand() noexcept
        : Command{Command::Tag::EMeshBuildCommand} {}

public:
    MeshBuildCommand(uint64_t handle, AccelBuildRequest request, uint64_t vertex_buffer,
                     size_t vertex_buffer_offset, size_t vertex_buffer_size, size_t vertex_stride,
                     uint64_t triangle_buffer, size_t triangle_buffer_offset, size_t triangle_buffer_size) noexcept
        : Command{Command::Tag::EMeshBuildCommand}, _handle{handle}, _request{request},
          _vertex_buffer{vertex_buffer}, _vertex_buffer_offset{vertex_buffer_offset},
          _vertex_buffer_size{vertex_buffer_size}, _vertex_stride{vertex_stride},
          _triangle_buffer{triangle_buffer}, _triangle_buffer_offset{triangle_buffer_offset},
          _triangle_buffer_size{triangle_buffer_size} {
    }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto request() const noexcept { return _request; }
    [[nodiscard]] auto vertex_buffer() const noexcept { return _vertex_buffer; }
    [[nodiscard]] auto vertex_stride() const noexcept { return _vertex_stride; }
    [[nodiscard]] auto vertex_buffer_offset() const noexcept { return _vertex_buffer_offset; }
    [[nodiscard]] auto vertex_buffer_size() const noexcept { return _vertex_buffer_size; }
    [[nodiscard]] auto triangle_buffer() const noexcept { return _triangle_buffer; }
    [[nodiscard]] auto triangle_buffer_offset() const noexcept { return _triangle_buffer_offset; }
    [[nodiscard]] auto triangle_buffer_size() const noexcept { return _triangle_buffer_size; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class CurveBuildCommand final : public Command {

private:
    uint64_t _handle{};
    AccelBuildRequest _request{};
    CurveBasis _basis{};
    size_t _cp_count{};
    size_t _seg_count{};
    uint64_t _cp_buffer{};
    size_t _cp_buffer_offset{};
    size_t _cp_stride{};
    uint64_t _seg_buffer{};
    size_t _seg_buffer_offset{};

private:
    CurveBuildCommand() noexcept
        : Command{Command::Tag::ECurveBuildCommand} {}

public:
    CurveBuildCommand(uint64_t handle, AccelBuildRequest request, CurveBasis basis,
                      size_t cp_count, size_t seg_count,
                      uint64_t cp_buffer, size_t cp_buffer_offset, size_t cp_stride,
                      uint64_t seg_buffer, size_t seg_buffer_offset) noexcept
        : Command{Command::Tag::ECurveBuildCommand},
          _handle{handle}, _request{request}, _basis{basis},
          _cp_count{cp_count}, _seg_count{seg_count},
          _cp_buffer{cp_buffer}, _cp_buffer_offset{cp_buffer_offset}, _cp_stride{cp_stride},
          _seg_buffer{seg_buffer}, _seg_buffer_offset{seg_buffer_offset} {}

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto request() const { return _request; }
    [[nodiscard]] auto basis() const { return _basis; }
    [[nodiscard]] auto cp_count() const { return _cp_count; }
    [[nodiscard]] auto seg_count() const { return _seg_count; }
    [[nodiscard]] auto cp_buffer() const { return _cp_buffer; }
    [[nodiscard]] auto cp_buffer_offset() const { return _cp_buffer_offset; }
    [[nodiscard]] auto cp_stride() const { return _cp_stride; }
    [[nodiscard]] auto seg_buffer() const { return _seg_buffer; }
    [[nodiscard]] auto seg_buffer_offset() const { return _seg_buffer_offset; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class ProceduralPrimitiveBuildCommand final : public Command {

private:
    uint64_t _handle{};
    AccelBuildRequest _request{};
    uint64_t _aabb_buffer{};
    size_t _aabb_buffer_offset{};
    size_t _aabb_buffer_size{};

public:
    ProceduralPrimitiveBuildCommand(uint64_t handle, AccelBuildRequest request, uint64_t aabb_buffer,
                                    size_t aabb_buffer_offset, size_t aabb_buffer_size)
        : Command{Command::Tag::EProceduralPrimitiveBuildCommand},
          _handle{handle}, _request{request}, _aabb_buffer{aabb_buffer},
          _aabb_buffer_offset{aabb_buffer_offset}, _aabb_buffer_size{aabb_buffer_size} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto request() const noexcept { return _request; }
    [[nodiscard]] auto aabb_buffer() const noexcept { return _aabb_buffer; }
    [[nodiscard]] auto aabb_buffer_offset() const noexcept { return _aabb_buffer_offset; }
    [[nodiscard]] auto aabb_buffer_size() const noexcept { return _aabb_buffer_size; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class MotionInstanceBuildCommand final : public Command {

private:
    uint64_t _handle{};
    uint64_t _child{};
    luisa::vector<MotionInstanceTransform> _keyframes;

public:
    MotionInstanceBuildCommand(uint64_t handle, uint64_t child,
                               luisa::vector<MotionInstanceTransform> keyframes) noexcept
        : Command{Command::Tag::EMotionInstanceBuildCommand},
          _handle{handle}, _child{child}, _keyframes{std::move(keyframes)} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto child() const noexcept { return _child; }
    [[nodiscard]] auto keyframes() const noexcept { return luisa::span{_keyframes}; }
    [[nodiscard]] auto steal_keyframes() noexcept { return std::move(_keyframes); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class AccelBuildCommand final : public Command {

public:
    struct alignas(16) Modification {

        // flags
        static constexpr auto flag_primitive = 1u << 0u;
        static constexpr auto flag_transform = 1u << 1u;
        static constexpr auto flag_opaque_on = 1u << 2u;
        static constexpr auto flag_opaque_off = 1u << 3u;
        static constexpr auto flag_opaque = flag_opaque_on | flag_opaque_off;
        static constexpr auto flag_visibility = 1u << 4u;
        static constexpr auto flag_user_id = 1u << 5u;

        // members
        uint index{};
        uint user_id{};
        uint flags{};
        uint vis_mask{};
        float affine[12]{};
        uint64_t primitive{};

        // ctor
        Modification() noexcept = default;
        explicit Modification(uint index) noexcept : index{index} {}
        // encode interfaces
        void set_transform(float4x4 m) noexcept {
            affine[0] = m[0][0];
            affine[1] = m[1][0];
            affine[2] = m[2][0];
            affine[3] = m[3][0];
            affine[4] = m[0][1];
            affine[5] = m[1][1];
            affine[6] = m[2][1];
            affine[7] = m[3][1];
            affine[8] = m[0][2];
            affine[9] = m[1][2];
            affine[10] = m[2][2];
            affine[11] = m[3][2];
            flags |= flag_transform;
        }
        void set_transform_data(const float affine_data[12]) noexcept {
            for (auto i = 0u; i < 12u; i++) { affine[i] = affine_data[i]; }
            flags |= flag_transform;
        }
        void set_visibility(uint8_t mask) noexcept {
            vis_mask = mask;
            flags |= flag_visibility;
        }
        void set_opaque(bool opaque) noexcept {
            flags &= ~flag_opaque;// clear old visibility flags
            flags |= opaque ? flag_opaque_on : flag_opaque_off;
        }
        void set_primitive(uint64_t handle) noexcept {
            primitive = handle;
            flags |= flag_primitive;
        }
        void set_user_id(uint id) noexcept {
            user_id = id;
            flags |= flag_user_id;
        }
    };

private:
    uint64_t _handle;
    uint32_t _instance_count;
    AccelBuildRequest _request;
    bool _update_instance_buffer_only;
    luisa::vector<Modification> _modifications;

public:
    AccelBuildCommand(uint64_t handle, uint32_t instance_count,
                      AccelBuildRequest request,
                      luisa::vector<Modification> modifications,
                      bool update_instance_buffer_only) noexcept
        : Command{Command::Tag::EAccelBuildCommand},
          _handle{handle}, _instance_count{instance_count},
          _request{request}, _update_instance_buffer_only{update_instance_buffer_only},
          _modifications{std::move(modifications)} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto request() const noexcept { return _request; }
    [[nodiscard]] auto instance_count() const noexcept { return _instance_count; }
    [[nodiscard]] auto modifications() const noexcept { return luisa::span{_modifications}; }
    [[nodiscard]] auto update_instance_buffer_only() const noexcept { return _update_instance_buffer_only; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class BindlessArrayUpdateCommand final : public Command {

public:
    struct Modification {

        enum struct Operation : uint {
            NONE,
            EMPLACE,
            REMOVE,
        };

        struct Buffer {
            uint64_t handle;
            size_t offset_bytes;
            Operation op;
            Buffer() noexcept
                : handle{0}, offset_bytes{0u}, op{Operation::NONE} {}
            Buffer(uint64_t handle, size_t offset_bytes, Operation op) noexcept
                : handle{handle}, offset_bytes{offset_bytes}, op{op} {}
            [[nodiscard]] static auto emplace(uint64_t handle, size_t offset_bytes) noexcept {
                return Buffer{handle, offset_bytes, Operation::EMPLACE};
            }
            [[nodiscard]] static auto remove() noexcept {
                return Buffer{0u, 0u, Operation::REMOVE};
            }
        };

        struct Texture {
            uint64_t handle;
            Sampler sampler;
            Operation op;
            Texture() noexcept
                : handle{0u}, sampler{}, op{Operation::NONE} {}
            Texture(uint64_t handle, Sampler sampler, Operation op) noexcept
                : handle{handle}, sampler{sampler}, op{op} {}
            [[nodiscard]] static auto emplace(uint64_t handle, Sampler sampler) noexcept {
                return Texture{handle, sampler, Operation::EMPLACE};
            }
            [[nodiscard]] static auto remove() noexcept {
                return Texture{0u, Sampler{}, Operation::REMOVE};
            }
        };

        size_t slot;
        Buffer buffer;
        Texture tex2d;
        Texture tex3d;

        explicit Modification(size_t slot) noexcept
            : slot{slot}, buffer{}, tex2d{}, tex3d{} {}

        explicit Modification(size_t slot, Buffer buffer, Texture tex2d, Texture tex3d) noexcept
            : slot{slot}, buffer{buffer}, tex2d{tex2d}, tex3d{tex3d} {}
    };

    static_assert(sizeof(Modification) == 64u);

private:
    uint64_t _handle;
    luisa::vector<Modification> _modifications;

public:
    BindlessArrayUpdateCommand(uint64_t handle,
                               luisa::vector<Modification> &&mods) noexcept
        : Command{Command::Tag::EBindlessArrayUpdateCommand},
          _handle{handle}, _modifications{std::move(mods)} {}
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto steal_modifications() noexcept { return std::move(_modifications); }
    [[nodiscard]] auto set_modifications(luisa::vector<Modification> &&mods) noexcept { return _modifications = std::move(mods); }
    [[nodiscard]] luisa::span<const Modification> modifications() const noexcept { return _modifications; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::COMPUTE)
};

class CustomCommand : public Command {

public:
    explicit CustomCommand() noexcept
        : Command{Command::Tag::ECustomCommand} {}
    [[nodiscard]] virtual uint64_t uuid() const noexcept = 0;
    virtual ~CustomCommand() noexcept override = default;
    LUISA_MAKE_COMMAND_COMMON_ACCEPT()
};

// For custom shader-dispatch or pass
class CustomDispatchCommand : public CustomCommand {

public:

    class ArgumentVisitor {
    public:
        virtual ~ArgumentVisitor() noexcept = default;
        virtual void visit(const Argument::Buffer &, Usage usage) noexcept = 0;
        virtual void visit(const Argument::Texture &, Usage usage) noexcept = 0;
        virtual void visit(const Argument::BindlessArray &, Usage usage) noexcept = 0;
        virtual void visit(const Argument::Accel &, Usage usage) noexcept = 0;
    };

public:
    explicit CustomDispatchCommand() noexcept = default;
    ~CustomDispatchCommand() noexcept override = default;

    virtual void traverse_arguments(ArgumentVisitor &visitor) const noexcept = 0;

    template<typename F>
        requires(!std::derived_from<std::remove_cvref_t<F>, ArgumentVisitor>)
    void traverse_arguments(F &&f) const noexcept {
        class Adapter final : public ArgumentVisitor {
        private:
            F &_f;

        public:
            explicit Adapter(F &f) noexcept : _f{f} {}
            void visit(const Argument::Buffer &resource, Usage usage) noexcept override {
                _f(resource, usage);
            }
            void visit(const Argument::Texture &resource, Usage usage) noexcept override {
                _f(resource, usage);
            }
            void visit(const Argument::BindlessArray &resource, Usage usage) noexcept override {
                _f(resource, usage);
            }
            void visit(const Argument::Accel &resource, Usage usage) noexcept override {
                _f(resource, usage);
            }
        };
        Adapter adapter{f};
        this->traverse_arguments(adapter);
    }
};

}// namespace luisa::compute
