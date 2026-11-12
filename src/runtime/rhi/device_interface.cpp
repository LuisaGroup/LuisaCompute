#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

DeviceInterface::DeviceInterface(Context &&ctx) noexcept
    : _ctx_impl{std::move(ctx).impl()} {}

DeviceInterface::~DeviceInterface() noexcept = default;

ShaderCreationInfo DeviceInterface::create_tile_kernel(
    const ShaderOption &, const tile::Function &,
    const tile::CompileOptions &, tile::KernelMetadata &) noexcept {
    return ShaderCreationInfo::make_invalid();
}

Context DeviceInterface::context() const noexcept {
    return Context{_ctx_impl};
}

void DeviceInterface::set_stream_log_callback(uint64_t stream_handle,
                                              const StreamLogCallback &callback) noexcept {
    LUISA_WARNING_WITH_LOCATION("DeviceInterface::set_stream_log_callback() is not "
                                "implemented. Calls to this method are ignored.");
}

ResourceCreationInfo DeviceInterface::create_curve(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
    return {};
}

void DeviceInterface::destroy_curve(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo DeviceInterface::create_motion_instance(const AccelMotionOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
    return {};
}

void DeviceInterface::destroy_motion_instance(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo DeviceInterface::create_mega_geometry(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
    return {};
}

void DeviceInterface::destroy_mega_geometry(uint64_t handle) noexcept {
    // Backends that do not distinguish cluster meshes created them through
    // create_mesh, so the ordinary destructor is the correct fallback.
    this->destroy_mesh(handle);
}

uint64_t DeviceInterface::mega_geometry_device_address(uint64_t handle) noexcept {
    return 0u;
}

uint64_t DeviceInterface::buffer_device_address(uint64_t buffer_handle) noexcept {
    return 0u;
}

ResourceCreationInfo DeviceInterface::create_opacity_micromap(
    const OpacityMicromapOption &option) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Opacity micromaps are not supported by the '{}' backend, so this device "
        "cannot build one (requested {} triangles at subdivision level {}{}).",
        backend_name(), option.triangle_count, option.subdivision_level,
        option.format == 1u ? ", four-state" : ", two-state");
    return ResourceCreationInfo::make_invalid();
}

void DeviceInterface::destroy_opacity_micromap(uint64_t handle) noexcept {
    // Unreachable for a backend that never created one, but a handle this base
    // class did not create must not be silently ignored either.
    LUISA_ERROR_WITH_LOCATION(
        "destroy_opacity_micromap(0x{:x}) was called on the '{}' backend, which "
        "cannot create opacity micromaps.",
        handle, backend_name());
}

uint32_t DeviceInterface::opacity_micromap_max_subdivision_level(bool four_state) noexcept {
    return 0u;
}

}// namespace luisa::compute
