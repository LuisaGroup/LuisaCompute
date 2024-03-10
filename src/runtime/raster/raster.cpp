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
}// namespace luisa::compute
