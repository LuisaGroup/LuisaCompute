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
void RasterShaderInvoke::check_scene(luisa::vector<RasterMesh> const &scene) noexcept {
    for (auto &&mesh : scene) {
        auto vb = mesh.vertex_buffers();
        if (vb.size() != _mesh_format->vertex_stream_count()) {
            LUISA_ERROR("Vertex buffer count mismatch!");
        }
        for (size_t i = 0; i < vb.size(); ++i) {
            auto stream = _mesh_format->attributes(i);
            size_t target_stride = 0;
            for (auto &s : stream) {
                target_stride += VertexElementFormatStride(s.format);
            }
            if (target_stride != vb[i].stride()) {
                LUISA_ERROR("Vertex buffer {} stride mismatch!", std::to_string(i));
            }
        }
    }
}
namespace detail {
bool rastershader_rettype_is_float4(Type const *t) noexcept {
    return (t->is_vector() && t->dimension() == 4 && t->element()->tag() == Type::Tag::FLOAT32);
};
void rastershader_check_vertex_func(Function func) noexcept {
    // rastershader_check_func(func);
    auto ret_type = func.return_type();

    if (rastershader_rettype_is_float4(ret_type))
        return;
    if (ret_type->is_structure() && ret_type->members().size() >= 1 && rastershader_rettype_is_float4(ret_type->members()[0])) {
        if (ret_type->members().size() > 16) {
            LUISA_ERROR("Vertex shader return type's structure element count need less than 16!");
        }
        for (auto &&i : ret_type->members()) {
            if (!(i->is_vector() || i->is_scalar()))
                LUISA_ERROR("Vertex shader return type can only contain scalar and vector type!");
        }
        return;
    }
    LUISA_ERROR("First element of vertex shader's return type must be float4!");
}
void rastershader_check_pixel_func(Function func) noexcept {
    // rastershader_check_func(func);
    auto ret_type = func.return_type();
    if (rastershader_rettype_is_float4(ret_type)) {
        return;
    }
    if (ret_type->is_structure() && ret_type->members().size() >= 1) {
        if (ret_type->members().size() > 8) {
            LUISA_ERROR("Pixel shader return type's structure element count need less than 8!");
        }
        for (auto &&i : ret_type->members()) {
            if (!(i->is_vector() || i->is_scalar()))
                LUISA_ERROR("Pixel shader return type can only contain scalar and vector type!");
        }
        return;
    }

    LUISA_ERROR("Illegal pixel shader return type!");
}
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
