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

BufferCreationInfo DeviceInterface::create_buffer(const ir::CArc<ir::Type> *element, void *external_memory, size_t size_bytes) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

BufferCreationInfo DeviceInterface::create_buffer(const Type *element, void *external_memory, size_t size_bytes) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute

