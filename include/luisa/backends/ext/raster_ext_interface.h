#pragma once

#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/raster/raster_state.h>
namespace luisa::compute {
template <typename T>
class ImageView;
enum class IndexFormat {
    UInt16,
    UInt32
};
struct IndexBufferView {
    uint64_t handle;
    uint64_t size;
    IndexFormat format;
};
class MeshFormat;
class RasterExt : public DeviceExtension {
protected:
    ~RasterExt() noexcept = default;
public:
    static constexpr luisa::string_view name = "RasterExt";
    // shader
    [[nodiscard]] virtual ResourceCreationInfo create_raster_shader(
        const MeshFormat &mesh_format,
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept = 0;

    [[nodiscard]] virtual ResourceCreationInfo load_raster_shader(
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept = 0;

    virtual void destroy_raster_shader(uint64_t handle) noexcept = 0;

    // depth buffer
    [[nodiscard]] virtual ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept = 0;
    virtual void destroy_depth_buffer(uint64_t handle) noexcept = 0;

    inline luisa::unique_ptr<Command> clear_render_target(ImageView<float> render_target, float4 value) noexcept;
};
}// namespace luisa::compute
