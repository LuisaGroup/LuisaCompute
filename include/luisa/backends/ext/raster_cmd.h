#pragma once

#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/command_encoder.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/backends/ext/registry.h>
#include <luisa/backends/ext/raster_ext_interface.h>

namespace luisa::compute {

class LC_RUNTIME_API DrawRasterSceneCommand final : public CustomDispatchCommand {
    friend lc::validation::Stream;

private:
    uint64_t _raster_scene;
    std::array<Argument::Texture, 8u> _rtv_texs;
    size_t _rtv_count;
    Argument::Texture _dsv_tex;
    Viewport _viewport;

public:
    DrawRasterSceneCommand(uint64_t raster_scene,
                           std::array<Argument::Texture, 8u> rtv_textures,
                           size_t rtv_count,
                           Argument::Texture dsv_texture,
                           Viewport viewport) noexcept
        : _raster_scene(raster_scene), _rtv_texs{rtv_textures}, _rtv_count{rtv_count}, _dsv_tex{dsv_texture}, _viewport{viewport} {
    }

public:
    DrawRasterSceneCommand(DrawRasterSceneCommand const &) noexcept = delete;
    DrawRasterSceneCommand(DrawRasterSceneCommand &&) noexcept = default;
    ~DrawRasterSceneCommand() noexcept override = default;
    [[nodiscard]] auto rtv_texs() const noexcept { return luisa::span{_rtv_texs.data(), _rtv_count}; }
    [[nodiscard]] auto const &dsv_tex() const noexcept { return _dsv_tex; }
    [[nodiscard]] auto raster_scene() const noexcept { return _raster_scene; }
    [[nodiscard]] auto viewport() const noexcept { return _viewport; }
    [[nodiscard]] uint64_t uuid() const noexcept override { return to_underlying(CustomCommandUUID::RASTER_DRAW_SCENE); }
    LUISA_MAKE_COMMAND_COMMON(StreamTag::GRAPHICS)
};

class LC_RUNTIME_API BuildRasterSceneCommand final : public CustomCommand {
public:
    struct Modification {
        luisa::fixed_vector<VertexBufferView, 2> vertex_buffers;
        luisa::variant<IndexBufferView, uint> index_buffer;
        ShaderDispatchCmdEncoder encoder;
        uint instance;
        RasterState state;
        uint flag;
        static constexpr uint flag_vertex_buffer = 1u << 0u;
        static constexpr uint flag_index_buffer = 1u << 1u;
        static constexpr uint flag_instance = 1u << 2u;
        static constexpr uint flag_shader = 1u << 3u;
        static constexpr uint flag_all = ~0u;
    };
    using Modifications = luisa::vector<std::pair<size_t, Modification>>;
    
private:
    Modifications _modifications;

public:
    explicit BuildRasterSceneCommand(Modifications &&modifications)
        : _modifications{std::move(modifications)} {}
    [[nodiscard]] uint64_t uuid() const noexcept override { return to_underlying(CustomCommandUUID::RASTER_BUILD_SCENE); }
    [[nodiscard]] auto modifications() const noexcept { return luisa::span{_modifications}; }
};

class ClearDepthCommand final : public CustomCommand {
    friend lc::validation::Stream;
    uint64_t _handle;
    float _value;

public:
    explicit ClearDepthCommand(uint64_t handle, float value) noexcept
        : _handle{handle}, _value(value) {
    }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto value() const noexcept { return _value; }
    [[nodiscard]] uint64_t uuid() const noexcept override { return to_underlying(CustomCommandUUID::RASTER_CLEAR_DEPTH); }

    LUISA_MAKE_COMMAND_COMMON(StreamTag::GRAPHICS)
};

}// namespace luisa::compute
