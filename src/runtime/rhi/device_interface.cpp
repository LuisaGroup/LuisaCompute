#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/context.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

DeviceInterface::DeviceInterface(Context &&ctx) noexcept
    : _ctx_impl{std::move(ctx).impl()} {}

DeviceInterface::~DeviceInterface() noexcept = default;

Context DeviceInterface::context() const noexcept {
    return Context{_ctx_impl};
}

void DeviceInterface::set_stream_log_callback(uint64_t stream_handle,
                                              const StreamLogCallback &callback) noexcept {
    LUISA_ERROR("DeviceInterface::set_stream_log_callback() is not "
                                "implemented. Calls to this method are ignored.");
}

ResourceCreationInfo DeviceInterface::create_curve(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void DeviceInterface::destroy_curve(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo DeviceInterface::create_motion_instance(const AccelMotionOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void DeviceInterface::destroy_motion_instance(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute
