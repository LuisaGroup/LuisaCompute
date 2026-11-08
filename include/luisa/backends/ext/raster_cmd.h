#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

class LUISA_RUNTIME_API DrawRasterSceneCommand final : public CustomCommand, public ShaderDispatchCommandBase {
    friend lc::validation::Stream;

private:
    std::array<Argument::Texture, 8u> _rtv_texs;
    size_t _rtv_count;
    Argument::Texture _dsv_tex;
    luisa::vector<RasterMesh> _scene;
    Viewport _viewport;
    MeshFormat const *_mesh_format;
    RasterState _raster_state;
    // A raster draw built from a native (hand-written) shader has no
    // saved-argument ABI: there is no `Function` behind it, so the backend
    // cannot read the per-resource usages from the shader. Its launcher
    // therefore packs them into the command instead, one entry per resource
    // argument in `arguments()` order (uniforms excluded), mirroring
    // `NativeShaderDispatchCommand`. `_native_shader` is the discriminator: it
    // is false for every DSL raster draw, which keeps that path byte-identical.
    bool _native_shader{false};
    luisa::vector<Usage> _argument_usages;

public:
    DrawRasterSceneCommand(uint64_t shader_handle,
                           luisa::vector<std::byte> &&argument_buffer,
                           size_t argument_count,
                           std::array<Argument::Texture, 8u> rtv_textures,
                           size_t rtv_count,
                           Argument::Texture dsv_texture,
                           luisa::vector<RasterMesh> &&scene,
                           Viewport viewport,
                           const RasterState &raster_state,
                           MeshFormat const *mesh_format) noexcept
        : ShaderDispatchCommandBase{
              shader_handle, std::move(argument_buffer), argument_count},
          _rtv_texs{rtv_textures}, _rtv_count{rtv_count}, _dsv_tex{dsv_texture}, _scene{std::move(scene)}, _viewport{viewport}, _mesh_format{mesh_format}, _raster_state{raster_state} {
    }

public:
    DrawRasterSceneCommand(DrawRasterSceneCommand const &) noexcept = delete;
    DrawRasterSceneCommand(DrawRasterSceneCommand &&) noexcept = default;
    ~DrawRasterSceneCommand() noexcept override = default;
    uint64_t custom_cmd_uuid() const noexcept override { return to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE); }
    void set_dsv_texs(Argument::Texture const &dsv_tex) noexcept {
        _dsv_tex = dsv_tex;
    }
    void set_scene(luisa::vector<RasterMesh> &&scene) noexcept {
        _scene = std::move(scene);
    }
    // Marks this draw as native-shader backed and adopts its declared usages
    // (one per non-uniform resource argument). Called by
    // `NativeShaderLauncher::draw()`; a DSL raster draw never calls it.
    void set_native_shader(luisa::vector<Usage> &&argument_usages) noexcept {
        _native_shader = true;
        _argument_usages = std::move(argument_usages);
    }
    [[nodiscard]] bool is_native_shader() const noexcept { return _native_shader; }
    [[nodiscard]] luisa::span<const Usage> argument_usages() const noexcept {
        return _argument_usages;
    }

    [[nodiscard]] auto rtv_texs() noexcept { return luisa::span{_rtv_texs.data(), _rtv_count}; }
    [[nodiscard]] auto rtv_texs() const noexcept { return luisa::span{_rtv_texs.data(), _rtv_count}; }
    [[nodiscard]] auto const &dsv_tex() const noexcept { return _dsv_tex; }
    [[nodiscard]] auto const &raster_state() const noexcept { return _raster_state; }
    [[nodiscard]] auto scene() const noexcept {
        return luisa::span{_scene};
    }
    [[nodiscard]] auto steal_scene() noexcept {
        return std::move(_scene);
    }
    [[nodiscard]] auto const &mesh_format() const noexcept {
        return *_mesh_format;
    }
    [[nodiscard]] auto viewport() const noexcept { return _viewport; }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::GRAPHICS)
};

class ClearDepthCommand final : public CustomCommand {
    friend lc::validation::Stream;
    uint64_t _handle;
    float _value;

public:
    explicit ClearDepthCommand(uint64_t handle, float value) noexcept
        : _handle{handle}, _value(value) {
    }
    uint64_t custom_cmd_uuid() const noexcept override { return to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH); }
    void set_handle(uint64_t handle) noexcept { _handle = handle; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto value() const noexcept { return _value; }

    LUISA_MAKE_COMMAND_COMMON(StreamTag::GRAPHICS)
};

class ClearRenderTargetCommand final : public CustomCommand {
    friend lc::validation::Stream;
    uint64_t _handle;
    float4 _value;
    uint _level;

public:
    explicit ClearRenderTargetCommand(uint64_t handle, float4 value, uint level) noexcept
        : _handle{handle}, _value(value), _level(level) {
    }
    uint64_t custom_cmd_uuid() const noexcept override { return to_underlying(CustomCommandUUID::RASTER_CLEAR_RENDER_TARGET); }
    void set_handle(uint64_t handle) noexcept { _handle = handle; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto value() const noexcept { return _value; }
    [[nodiscard]] auto level() const noexcept { return _level; }

    LUISA_MAKE_COMMAND_COMMON(StreamTag::GRAPHICS)
};

}// namespace luisa::compute
