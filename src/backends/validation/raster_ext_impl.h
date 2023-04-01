#pragma once
#include <backends/ext/raster_ext.h>
#include <vstl/common.h>
namespace lc::validation {
class Device;
using namespace luisa::compute;
class RasterExtImpl : public RasterExt, public vstd::IOperatorNewBase {
    RasterExt *_impl;

public:
    RasterExtImpl(RasterExt *impl) : _impl{impl} {}
    ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept override;

    void save_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        luisa::string_view name,
        bool enable_debug_info,
        bool enable_fast_math) noexcept override;

    ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        const RasterState &raster_state,
        luisa::span<const PixelFormat> rtv_format,
        DepthFormat dsv_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept override;

    void destroy_raster_shader(uint64_t handle) noexcept override;
    // depth buffer
    ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
};
}// namespace lc::validation