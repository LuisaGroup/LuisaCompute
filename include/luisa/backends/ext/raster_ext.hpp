#pragma once
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/runtime/device.h>
namespace luisa::compute{

template<typename V, typename P>
[[nodiscard]] typename RasterKernel<V, P>::RasterShaderType Device::compile(
    const RasterKernel<V, P> &kernel,
    const MeshFormat &mesh_format,
    const ShaderOption &option) noexcept {
    return _create<typename RasterKernel<V, P>::RasterShaderType>(extension<RasterExt>(), mesh_format, kernel.vert(), kernel.pixel(), option);
}

template<typename V, typename P>
void Device::compile_to(
    const RasterKernel<V, P> &kernel,
    const MeshFormat &mesh_format,
    luisa::string_view serialization_path,
    const ShaderOption &option) noexcept {
    _check_no_implicit_binding(kernel.vert(), serialization_path);
    _check_no_implicit_binding(kernel.pixel(), serialization_path);
    extension<RasterExt>()->create_raster_shader(mesh_format, kernel.vert(), kernel.pixel(), serialization_path, option);
}

template<typename... Args>
RasterShader<Args...> Device::load_raster_shader(
    luisa::string_view shader_name) noexcept {
    return _create<RasterShader<Args...>>(extension<RasterExt>(), shader_name);
}
}
