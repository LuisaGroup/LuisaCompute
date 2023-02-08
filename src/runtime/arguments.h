#pragma once
#include <core/stl/memory.h>
namespace luisa::compute {

struct alignas(8) Argument {

    enum struct Tag : uint8_t {
        BUFFER,
        TEXTURE,
        UNIFORM,
        BINDLESS_ARRAY,
        ACCEL,
    };

    Tag tag{};

    Argument() noexcept = default;
    explicit Argument(Tag tag) noexcept : tag{tag} {}
};
struct IndirectDispatchArg {
    uint64_t handle;
};
struct BufferArgument : Argument {
    uint64_t handle{};
    size_t offset{};
    size_t size{};
    BufferArgument() noexcept : Argument{Tag::BUFFER} {}
    BufferArgument(uint64_t handle, size_t offset, size_t size) noexcept
        : Argument{Tag::BUFFER}, handle{handle}, offset{offset}, size{size} {}
};

struct TextureArgument : Argument {
    uint64_t handle{};
    uint32_t level{};
    TextureArgument() noexcept : Argument{Tag::TEXTURE} {}
    TextureArgument(uint64_t handle, uint32_t level) noexcept
        : Argument{Tag::TEXTURE}, handle{handle}, level{level} {}
};

struct UniformArgumentHead : Argument {
    size_t size{};
    UniformArgumentHead() noexcept : Argument{Tag::UNIFORM} {}
    explicit UniformArgumentHead(size_t size) noexcept
        : Argument{Tag::UNIFORM}, size{size} {}
};

struct UniformArgument : UniformArgumentHead {
    const std::byte *data{};
    UniformArgument(UniformArgumentHead head, const std::byte *data) noexcept
        : UniformArgumentHead{head}, data{data} {}
    [[nodiscard]] auto span() const noexcept { return luisa::span{data, size}; }
};

struct BindlessArrayArgument : Argument {
    uint64_t handle{};
    BindlessArrayArgument() noexcept : Argument{Tag::BINDLESS_ARRAY} {}
    explicit BindlessArrayArgument(uint64_t handle) noexcept
        : Argument{Tag::BINDLESS_ARRAY}, handle{handle} {}
};

struct AccelArgument : Argument {
    uint64_t handle{};
    AccelArgument() noexcept : Argument{Tag::ACCEL} {}
    explicit AccelArgument(uint64_t handle) noexcept
        : Argument{Tag::ACCEL}, handle{handle} {}
};
}// namespace luisa::compute