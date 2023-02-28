#pragma once

#include <ast/function.h>
#include <runtime/command.h>
#include <runtime/raster/viewport.h>

namespace luisa::compute {

class LC_RUNTIME_API ShaderDispatchCmdEncoder {

public:
    using Argument = ShaderDispatchCommandBase::Argument;

protected:
    uint64_t _handle;
    size_t _argument_count;
    size_t _argument_idx{0};
    luisa::vector<std::byte> _argument_buffer;
    ShaderDispatchCmdEncoder(uint64_t handle,
                             size_t arg_count,
                             size_t uniform_size) noexcept;
    void _encode_pending_bindings(
        luisa::span<const Function::Binding> bindings) noexcept;
    void _encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept;
    void _encode_texture(uint64_t handle, uint32_t level) noexcept;
    void _encode_uniform(const void *data, size_t size) noexcept;
    void _encode_bindless_array(uint64_t handle) noexcept;
    void _encode_accel(uint64_t handle) noexcept;
    [[nodiscard]] std::byte *_make_space(size_t size) noexcept;
    [[nodiscard]] Argument &_create_argument() noexcept;

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto argument_count() const noexcept { return static_cast<size_t>(_argument_count); }

public:
    [[nodiscard]] static size_t compute_uniform_size(luisa::span<const Variable> arguments) noexcept;
    [[nodiscard]] static size_t compute_uniform_size(luisa::span<const Type *const> arg_types) noexcept;
};

class LC_RUNTIME_API ComputeDispatchCmdEncoder final : public ShaderDispatchCmdEncoder {

private:
    luisa::variant<uint3, IndirectDispatchArg> _dispatch_size;
    luisa::span<const Function::Binding> _bindings;

public:
    explicit ComputeDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size,
                                       luisa::span<const Function::Binding> bindings) noexcept;
    ComputeDispatchCmdEncoder(ComputeDispatchCmdEncoder &&) noexcept = default;
    ComputeDispatchCmdEncoder &operator=(ComputeDispatchCmdEncoder &&) noexcept = default;
    ~ComputeDispatchCmdEncoder() noexcept = default;
    void set_dispatch_size(uint3 launch_size) noexcept;
    void set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept;
    [[nodiscard]] auto const &dispatch_size() const noexcept { return _dispatch_size; }

    void encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept;
    void encode_texture(uint64_t handle, uint32_t level) noexcept;
    void encode_uniform(const void *data, size_t size) noexcept;
    void encode_bindless_array(uint64_t handle) noexcept;
    void encode_accel(uint64_t handle) noexcept;
    luisa::unique_ptr<ShaderDispatchCommand> build() &&noexcept;
};

class RasterMesh;

class LC_RUNTIME_API RasterDispatchCmdEncoder final : public ShaderDispatchCmdEncoder {

private:
    luisa::span<const Function::Binding> _vertex_bindings;
    luisa::span<const Function::Binding> _pixel_bindings;
    luisa::span<const Function::Binding> _current_bindings;
    std::array<ShaderDispatchCommandBase::Argument::Texture, 8u> _rtv_texs;
    size_t _rtv_count{};
    ShaderDispatchCommandBase::Argument::Texture _dsv_tex{};
    luisa::vector<RasterMesh> _scene;
    Viewport _viewport{};

    void update_arg();

public:
    explicit RasterDispatchCmdEncoder(uint64_t handle, size_t arg_count, size_t uniform_size,
                                      luisa::span<const Function::Binding> vertex_bindings,
                                      luisa::span<const Function::Binding> pixel_bindings) noexcept;
    RasterDispatchCmdEncoder(RasterDispatchCmdEncoder const &) noexcept = delete;
    ~RasterDispatchCmdEncoder() noexcept;
    RasterDispatchCmdEncoder(RasterDispatchCmdEncoder &&) noexcept;
    RasterDispatchCmdEncoder &operator=(RasterDispatchCmdEncoder &&) noexcept;
    [[nodiscard]] auto rtv_texs() const noexcept { return luisa::span{_rtv_texs.data(), _rtv_count}; }
    [[nodiscard]] auto const &dsv_tex() const noexcept { return _dsv_tex; }
    void set_rtv_texs(luisa::span<const ShaderDispatchCommandBase::Argument::Texture> tex) noexcept;
    void set_dsv_tex(ShaderDispatchCommandBase::Argument::Texture tex) noexcept;
    void set_scene(luisa::vector<RasterMesh> &&scene) noexcept;
    void set_viewport(Viewport viewport) noexcept;
    // TODO
    void encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept;
    void encode_texture(uint64_t handle, uint32_t level) noexcept;
    void encode_uniform(const void *data, size_t size) noexcept;
    void encode_bindless_array(uint64_t handle) noexcept;
    void encode_accel(uint64_t handle) noexcept;
    luisa::unique_ptr<DrawRasterSceneCommand> build() &&noexcept;
};

}// namespace luisa::compute
