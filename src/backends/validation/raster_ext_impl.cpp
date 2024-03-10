#include "raster_ext_impl.h"
#include "rw_resource.h"
#include "depth_buffer.h"
#include <luisa/core/logging.h>
namespace lc::validation {
ResourceCreationInfo RasterExtImpl::create_raster_shader(
    Function vert,
    Function pixel,
    const ShaderOption &shader_option) noexcept {
    auto res = _impl->create_raster_shader(vert, pixel, shader_option);
    if (res.valid())
        new RWResource(res.handle, RWResource::Tag::RASTER_SHADER, false);
    return res;
}

ResourceCreationInfo RasterExtImpl::load_raster_shader(
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    auto res = _impl->load_raster_shader(types, ser_path);
    new RWResource(res.handle, RWResource::Tag::RASTER_SHADER, false);
    return res;
}

void RasterExtImpl::destroy_raster_shader(uint64_t handle) noexcept {
    _impl->destroy_raster_shader(handle);
    RWResource::dispose(handle);
}

// depth buffer
ResourceCreationInfo RasterExtImpl::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    auto buffer = _impl->create_depth_buffer(format, width, height);
    new DepthBuffer(buffer.handle);
    return buffer;
}
void RasterExtImpl::destroy_depth_buffer(uint64_t handle) noexcept {
    _impl->destroy_depth_buffer(handle);
    RWResource::dispose(handle);

}
}// namespace lc::validation
