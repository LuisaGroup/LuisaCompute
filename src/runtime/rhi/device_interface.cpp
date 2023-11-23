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

ResourceCreationInfo DeviceInterface::create_curve(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void DeviceInterface::destroy_curve(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute

