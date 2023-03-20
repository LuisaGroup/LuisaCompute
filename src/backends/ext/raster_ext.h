#pragma once

#include <runtime/device.h>

namespace luisa::compute {

class MeshFormat;
class RasterState;

class RasterExt : public DeviceExtension {

public:
    static constexpr luisa::string_view name = "RasterExt";

    [[nodiscard]] virtual ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept = 0;

    virtual void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view name,
        bool enable_debug_info,
        bool enable_fast_math) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept = 0;

    virtual void destroy_raster_shader(uint64_t handle) noexcept {}
};

template<typename V, typename P>
[[nodiscard]] typename RasterKernel<V, P>::RasterShaderType Device::compile(
    const RasterKernel<V, P> &kernel,
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    luisa::span<PixelFormat const> rtv_format,
    DepthFormat dsv_format,
    const ShaderOption &option) noexcept {
    return _create<typename RasterKernel<V, P>::RasterShaderType>(static_cast<RasterExt *>(_impl->extension(RasterExt::name)), mesh_format, raster_state, rtv_format, dsv_format, kernel.vert(), kernel.pixel(), option);
}

template<typename V, typename P>
void Device::compile_to(
    const RasterKernel<V, P> &kernel,
    const MeshFormat &format,
    luisa::string_view serialization_path,
    bool enable_debug_info,
    bool enable_fast_math) noexcept {
    _check_no_implicit_binding(kernel.vert(), serialization_path);
    _check_no_implicit_binding(kernel.pixel(), serialization_path);
    static_cast<RasterExt *>(_impl->extension(RasterExt::name))->save_raster_shader(format, kernel.vert(), kernel.pixel(), serialization_path, enable_debug_info, enable_fast_math);
}

template<typename... Args>
RasterShader<Args...> Device::load_raster_shader(
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    luisa::span<PixelFormat const> rtv_format,
    DepthFormat dsv_format,
    luisa::string_view shader_name) noexcept {
    return _create<RasterShader<Args...>>(static_cast<RasterExt *>(_impl->extension(RasterExt::name)), mesh_format, raster_state, rtv_format, dsv_format, shader_name);
}
}// namespace luisa::compute