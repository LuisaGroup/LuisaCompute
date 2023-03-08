//
// Created by Mike Smith on 2023/2/21.
//

#include <runtime/raster/raster_scene.h>
#include <runtime/rhi/command.h>

namespace luisa::compute {

DrawRasterSceneCommand::~DrawRasterSceneCommand() noexcept = default;
luisa::span<const RasterMesh> DrawRasterSceneCommand::scene() const noexcept { return luisa::span{_scene}; }

DrawRasterSceneCommand::DrawRasterSceneCommand(uint64_t shader_handle,
                                               vector<std::byte> &&argument_buffer,
                                               size_t argument_count,
                                               std::array<Argument::Texture, 8u> rtv_textures,
                                               size_t rtv_count,
                                               ShaderDispatchCommandBase::Argument::Texture dsv_texture,
                                               vector<RasterMesh> &&scene,
                                               Viewport viewport) noexcept
    : ShaderDispatchCommandBase{Tag::EDrawRasterSceneCommand,
                                shader_handle,
                                std::move(argument_buffer),
                                argument_count},
      _rtv_texs{rtv_textures}, _rtv_count{rtv_count},
      _dsv_tex{dsv_texture}, _scene{std::move(scene)},
      _viewport{viewport} {}

DrawRasterSceneCommand::DrawRasterSceneCommand(DrawRasterSceneCommand &&) noexcept = default;

}// namespace luisa::compute
