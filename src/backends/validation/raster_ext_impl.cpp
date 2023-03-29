#include "raster_ext_impl.h"
#include "rw_resource.h"
namespace lc::validation {
ResourceCreationInfo RasterExtImpl::create_raster_shader(
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    luisa::span<const PixelFormat> rtv_format,
    DepthFormat dsv_format,
    Function vert,
    Function pixel,
    const ShaderOption &shader_option) noexcept {
    auto res = _impl->create_raster_shader(mesh_format, raster_state, rtv_format, dsv_format, vert, pixel, shader_option);
    res.handle = reinterpret_cast<uint64_t>(new RWResource(res.handle, RWResource::Tag::RASTER_SHADER, false));
    return res;
}

void RasterExtImpl::save_raster_shader(
    const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    luisa::string_view name,
    bool enable_debug_info,
    bool enable_fast_math) noexcept {
    _impl->save_raster_shader(mesh_format, vert, pixel, name, enable_debug_info, enable_fast_math);
}

ResourceCreationInfo RasterExtImpl::load_raster_shader(
    const MeshFormat &mesh_format,
    const RasterState &raster_state,
    luisa::span<const PixelFormat> rtv_format,
    DepthFormat dsv_format,
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    auto res = _impl->load_raster_shader(mesh_format, raster_state, rtv_format, dsv_format, types, ser_path);
    res.handle = reinterpret_cast<uint64_t>(new RWResource(res.handle, RWResource::Tag::RASTER_SHADER, false));
    return res;
}
void RasterExtImpl::destroy_raster_shader(uint64_t handle) noexcept {
    auto res = reinterpret_cast<RWResource *>(handle);
    _impl->destroy_raster_shader(res->handle());
    delete res;
}
}// namespace lc::validation