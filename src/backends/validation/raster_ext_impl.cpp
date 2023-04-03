#include "raster_ext_impl.h"
#include "rw_resource.h"
#include "device.h"
#include "depth_buffer.h"
namespace lc::validation {
ResourceCreationInfo RasterExtImpl::create_raster_shader(
    const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    const ShaderOption &shader_option) noexcept {
    auto res = _impl->create_raster_shader(mesh_format, vert, pixel, shader_option);
    if (res.handle != invalid_resource_handle)
        res.handle = reinterpret_cast<uint64_t>(new RWResource(res.handle, RWResource::Tag::RASTER_SHADER, false));
    return res;
}

ResourceCreationInfo RasterExtImpl::load_raster_shader(
    const MeshFormat &mesh_format,
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    auto res = _impl->load_raster_shader(mesh_format, types, ser_path);
    res.handle = reinterpret_cast<uint64_t>(new RWResource(res.handle, RWResource::Tag::RASTER_SHADER, false));
    return res;
}
void RasterExtImpl::destroy_raster_shader(uint64_t handle) noexcept {
    auto res = reinterpret_cast<RWResource *>(handle);
    _impl->destroy_raster_shader(res->handle());
    delete res;
}
// depth buffer
ResourceCreationInfo RasterExtImpl::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    std::lock_guard lck{Device::global_mtx()};
    auto buffer = _impl->create_depth_buffer(format, width, height);
    buffer.handle = reinterpret_cast<uint64_t>(new DepthBuffer(buffer.handle));
    return buffer;
}
void RasterExtImpl::destroy_depth_buffer(uint64_t handle) noexcept {
    std::lock_guard lck{Device::global_mtx()};
    auto buffer = reinterpret_cast<DepthBuffer *>(handle);
    _impl->destroy_depth_buffer(buffer->handle());
    delete buffer;
}
}// namespace lc::validation