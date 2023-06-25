#pragma once
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/vstl/common.h>
namespace lc::validation {
class Device;
using namespace luisa::compute;
class RasterExtImpl final : public RasterExt, public vstd::IOperatorNewBase {
    RasterExt *_impl;

public:
    RasterExtImpl(RasterExt *impl) : _impl{impl} {}
    ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept override;

    ResourceCreationInfo load_raster_shader(
        const MeshFormat &mesh_format,
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept override;

    void warm_up_pipeline_cache(
        uint64_t shader_handle,
        luisa::span<PixelFormat const> render_target_formats,
        DepthFormat depth_format,
        const RasterState &state) noexcept override;

    void destroy_raster_shader(uint64_t handle) noexcept override;
    // depth buffer
    ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
};
}// namespace lc::validation
