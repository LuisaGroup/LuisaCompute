#pragma once

#include <ast/function.h>
#include <runtime/command.h>
#include <runtime/raster/viewport.h>

namespace luisa::compute {

class LC_RUNTIME_API ShaderDispatchCmdEncoder {
public:
    using Argument = ShaderDispatchCommandBase::Argument;

protected:
    size_t _argument_count;
    size_t _argument_idx{0};
    luisa::vector<std::byte> _argument_buffer;
    ShaderDispatchCmdEncoder(size_t arg_count);
    void _encode_pending_bindings(Function kernel) noexcept;
    void _encode_buffer(Function kernel, uint64_t handle, size_t offset, size_t size) noexcept;
    void _encode_texture(Function kernel, uint64_t handle, uint32_t level) noexcept;
    void _encode_uniform(Function kernel, const void *data, size_t size) noexcept;
    void _encode_bindless_array(Function kernel, uint64_t handle) noexcept;
    void _encode_accel(Function kernel, uint64_t handle) noexcept;
    [[nodiscard]] std::byte *_make_space(size_t size) noexcept;
    Argument &create_arg();

public:
    [[nodiscard]] auto argument_count() const noexcept { return static_cast<size_t>(_argument_count); }
};

class LC_RUNTIME_API ComputeDispatchCmdEncoder final : public ShaderDispatchCmdEncoder {
private:
    uint64_t _handle{};
    luisa::variant<uint3, IndirectDispatchArg> _dispatch_size;
    Function _kernel;
public:
    explicit ComputeDispatchCmdEncoder(size_t arg_size, uint64_t handle, Function kernel) noexcept;
    ComputeDispatchCmdEncoder(ComputeDispatchCmdEncoder &&) noexcept = default;
    ComputeDispatchCmdEncoder &operator=(ComputeDispatchCmdEncoder &&) noexcept = default;
    ~ComputeDispatchCmdEncoder() noexcept = default;
    void set_dispatch_size(uint3 launch_size) noexcept;
    void set_dispatch_size(IndirectDispatchArg indirect_arg) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
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
    uint64_t _handle{};
    Function _vertex_func{};
    Function _pixel_func{};
    Function _default_func{};
    ShaderDispatchCommandBase::Argument::Texture _rtv_texs[8];
    size_t _rtv_count{};
    ShaderDispatchCommandBase::Argument::Texture _dsv_tex{};

    Function arg_kernel();

public:
    luisa::vector<RasterMesh> scene;
    Viewport viewport{};

    explicit RasterDispatchCmdEncoder(
        size_t arg_size,
        uint64_t handle,
        Function vertex_func,
        Function pixel_func) noexcept;
    RasterDispatchCmdEncoder(RasterDispatchCmdEncoder const &) noexcept = delete;
    ~RasterDispatchCmdEncoder() noexcept;
    RasterDispatchCmdEncoder(RasterDispatchCmdEncoder &&) noexcept;
    RasterDispatchCmdEncoder &operator=(RasterDispatchCmdEncoder &&) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto vertex_func() const noexcept { return _vertex_func; }
    [[nodiscard]] auto pixel_func() const noexcept { return _pixel_func; }
    [[nodiscard]] auto rtv_texs() const noexcept { return luisa::span<const ShaderDispatchCommandBase::Argument::Texture>{_rtv_texs, _rtv_count}; }
    [[nodiscard]] auto const &dsv_tex() const noexcept { return _dsv_tex; }
    void set_rtv_texs(luisa::span<const ShaderDispatchCommandBase::Argument::Texture> tex) {
        assert(tex.size() <= 8);
        _rtv_count = tex.size();
        memcpy(_rtv_texs, tex.data(), tex.size_bytes());
    }
    void set_dsv_tex(ShaderDispatchCommandBase::Argument::Texture tex) {
        _dsv_tex = tex;
    }
    // TODO
    void encode_buffer(uint64_t handle, size_t offset, size_t size) noexcept;
    void encode_texture(uint64_t handle, uint32_t level) noexcept;
    void encode_uniform(const void *data, size_t size) noexcept;
    void encode_bindless_array(uint64_t handle) noexcept;
    void encode_accel(uint64_t handle) noexcept;
    luisa::unique_ptr<DrawRasterSceneCommand> build() &&noexcept;
};

}// namespace luisa::compute