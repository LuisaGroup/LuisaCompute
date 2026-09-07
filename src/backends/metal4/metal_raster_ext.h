#pragma once

#include <luisa/backends/ext/raster_ext_interface.h>

namespace luisa::compute::metal {

class MetalDevice;

class MetalRasterExt final : public RasterExt {

private:
    MetalDevice *_device;

public:
    explicit MetalRasterExt(MetalDevice *device) noexcept : _device{device} {}
    ~MetalRasterExt() noexcept = default;
    [[nodiscard]] ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept override;
    [[nodiscard]] ResourceCreationInfo load_raster_shader(
        luisa::span<Type const *const> types,
        luisa::string_view name) noexcept override;
    void destroy_raster_shader(uint64_t handle) noexcept override;
    [[nodiscard]] ResourceCreationInfo create_depth_buffer(
        DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::metal
