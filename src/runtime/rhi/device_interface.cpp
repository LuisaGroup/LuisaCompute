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

uint64_t DeviceInterface::buffer_device_address(uint64_t handle) noexcept {
    return 0u;
}

}// namespace luisa::compute
