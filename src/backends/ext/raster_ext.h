#pragma once

#include <runtime/device.h>

namespace luisa::compute {
class MeshFormat;
class RasterExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "RasterExt";
    // shader
    [[nodiscard]] virtual ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept = 0;

    virtual void destroy_raster_shader(uint64_t handle) noexcept = 0;

    // depth buffer
    [[nodiscard]] virtual ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept = 0;
    virtual void destroy_depth_buffer(uint64_t handle) noexcept = 0;
};

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
}// namespace luisa::compute