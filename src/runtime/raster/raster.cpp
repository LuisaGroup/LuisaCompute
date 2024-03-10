#include <luisa/runtime/raster/raster_shader.h>
#include <luisa/runtime/depth_format.h>
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/raster_cmd.h>

namespace luisa::compute {

#ifndef NDEBUG
namespace detail {
void rastershader_check_rtv_format(luisa::span<const PixelFormat> rtv_format) noexcept {
    if (rtv_format.size() > 8) {
        LUISA_ERROR("Render target count must be less or equal than 8!");
    }
    for (size_t i = 0; i < rtv_format.size(); ++i) {
        if (rtv_format[i] > PixelFormat::RGBA32F)
            LUISA_ERROR("Illegal render target format at {}", (char)(i + 48));
    }
}
}// namespace detail
#endif
RasterScene::~RasterScene() noexcept {
    if (!_modifications.empty()) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Raster-Scene #{} destroyed with {} uncommitted modifications. "
            "Did you forget to call build()?",
            this->handle(), _modifications.size());
    }
}

RasterScene::RasterScene(RasterScene &&rhs) noexcept
    : _render_formats{std::move(rhs._render_formats)},
      _depth_format{rhs._depth_format},
      _modifications{std::move(rhs._modifications)},
      _instance_count{rhs._instance_count} {
}
}// namespace luisa::compute
