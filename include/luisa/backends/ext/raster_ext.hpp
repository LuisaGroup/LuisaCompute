#pragma once
#include <luisa/core/logging.h>
#include <luisa/backends/ext/raster_ext_interface.h>
#include <luisa/runtime/device.h>
#include <luisa/backends/ext/raster_cmd.h>
namespace luisa::compute {

namespace detail {
/// Fetches the raster extension of a device, failing closed when the backend
/// provides none. Every raster entry point must go through this helper:
/// `Device::extension<RasterExt>()` is null on a backend without one, so
/// dereferencing it directly would crash instead of reporting that the feature is
/// unavailable.
///
/// Backend support (verified against the device registration of each backend):
/// | backend | extension class | entry point |
/// | --- | --- | --- |
/// | dx | `DxRasterExt` | `src/backends/dx/DXApi/LCDevice.cpp` |
/// | vk | `VkRasterExt` | `src/backends/vk/device.cpp` |
/// | metal4 | `MetalRasterExt` | `src/backends/metal4/metal_device.cpp` |
/// Not supported: cuda, hip and the fallback backend, whose
/// `Device::extension<RasterExt>()` returns null. The validation layer wraps
/// whichever backend extension exists, so enabling it changes nothing here.
///
/// Note the two different things called "raster" in this tree: *this* extension
/// is the DSL rasterisation path (`RasterKernel` -> `create_raster_shader`), while
/// `NativeShaderExt::compile_raster` / `NativeShaderLauncher::draw` inject a
/// hand-written vertex + pixel pair and are implemented on dx and vk today (see
/// `luisa/backends/ext/native_shader_ext.h`).
[[nodiscard]] inline RasterExt *require_raster_ext(Device &device, luisa::string_view what) noexcept {
    if (auto ext = device.extension<RasterExt>(); ext != nullptr) { return ext; }
    LUISA_ERROR("'{}' requires a backend that provides the raster extension ('{}'), "
                "but the active backend '{}' does not.",
                what, RasterExt::name, device.backend_name());
}
}// namespace detail

// JIT route: compile the kernel's two stages and create the pipeline in memory.
// Provided by dx and metal4. **vk requires the AOT route below** - its
// `create_raster_shader` asserts `ShaderOption::compile_only`, which this overload
// leaves at its default `false`, so calling it on vk aborts.
template<typename V, typename P>
[[nodiscard]] typename RasterKernel<V, P>::RasterShaderType Device::compile(
    const RasterKernel<V, P> &kernel,
    const MeshFormat &mesh_format,
    const ShaderOption &option) noexcept {
    return _create<typename RasterKernel<V, P>::RasterShaderType>(
        detail::require_raster_ext(*this, "Compiling a raster kernel"), mesh_format,
        kernel.vert(), kernel.pixel(), option);
}

// AOT route: serialize the pipeline under `serialization_path` without creating it;
// `load_raster_shader<Args...>()` creates the object later. This is the only route on
// vk, and it works on dx and metal4 too.
template<typename V, typename P>
void Device::compile_to(
    const RasterKernel<V, P> &kernel,
    const MeshFormat &mesh_format,
    luisa::string_view serialization_path,
    const ShaderOption &option) noexcept {
    _check_no_implicit_binding(kernel.vert(), serialization_path);
    _check_no_implicit_binding(kernel.pixel(), serialization_path);
    auto raster_ext = detail::require_raster_ext(*this, "Compiling a raster kernel AOT");
    auto compile_option = option;
    compile_option.enable_cache = false;
    compile_option.compile_only = true;
    compile_option.name = luisa::string{serialization_path};
    static_cast<void>(raster_ext->create_raster_shader(
        mesh_format, kernel.vert(), kernel.pixel(), compile_option));
}

template<typename... Args>
RasterShader<Args...> Device::load_raster_shader(
    luisa::string_view shader_name) noexcept {
    return _create<RasterShader<Args...>>(
        detail::require_raster_ext(*this, "Loading a raster shader"), shader_name);
}
inline luisa::unique_ptr<Command> RasterExt::clear_render_target(ImageView<float> render_target, float4 value) noexcept {
    return luisa::make_unique<ClearRenderTargetCommand>(render_target.handle(), value, render_target.level());
}
}// namespace luisa::compute
