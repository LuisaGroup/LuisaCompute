#pragma once

#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/raster/viewport.h>

namespace luisa::compute {

class LC_RUNTIME_API ShaderDispatchCmdEncoder {

public:
    using Argument = ShaderDispatchCommandBase::Argument;

protected:
    uint64_t _handle;
    size_t _argument_count;
    size_t _argument_idx{0};
    luisa::vector<std::byte> _argument_buffer;

    [[nodiscard]] std::byte *_make_space(size_t size) noexcept;
    [[nodiscard]] Argument &_create_argument() noexcept;

public:
    ShaderDispatchCmdEncoder() noexcept = default;
    ShaderDispatchCmdEncoder(uint64_t handle,
                             size_t arg_count,
                             size_t uniform_size) noexcept;
    [[nodiscard]] static size_t compute_uniform_size(luisa::span<const Variable> arguments) noexcept;
    [[nodiscard]] static size_t compute_uniform_size(luisa::span<const Type *const> arg_types) noexcept;
    void encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept;
    void encode_texture(uint64_t handle, uint32_t level) noexcept;
    void encode_uniform(const void *data, size_t size) noexcept;
    void encode_bindless_array(uint64_t handle) noexcept;
    void encode_accel(uint64_t handle) noexcept;
};

class LC_RUNTIME_API ComputeDispatchCmdEncoder final : public ShaderDispatchCmdEncoder {

private:
    luisa::variant<uint3, IndirectDispatchArg, luisa::vector<uint3>> _dispatch_size;

public:
    explicit ComputeDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size) noexcept;
    ComputeDispatchCmdEncoder(ComputeDispatchCmdEncoder &&) noexcept = default;
    ComputeDispatchCmdEncoder &operator=(ComputeDispatchCmdEncoder &&) noexcept = default;
    ~ComputeDispatchCmdEncoder() noexcept = default;
    void set_dispatch_size(uint3 launch_size) noexcept;
    void set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept;
    void set_dispatch_sizes(luisa::span<const uint3> sizes) noexcept;

    luisa::unique_ptr<ShaderDispatchCommand> build() && noexcept;
};

}// namespace luisa::compute
