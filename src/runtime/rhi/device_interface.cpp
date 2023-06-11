#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/context.h>

namespace luisa::compute {

DeviceInterface::DeviceInterface(Context &&ctx) noexcept
    : _ctx_impl{std::move(ctx).impl()} {}

DeviceInterface::~DeviceInterface() noexcept = default;

Context DeviceInterface::context() const noexcept {
    return Context{_ctx_impl};
}

}// namespace luisa::compute

