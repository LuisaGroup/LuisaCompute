#include <backends/common/rust_device_common.h>
#include <backends/common/rust_device_common.h>
#include "remote_device.h"

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::compute::rust::create(std::move(ctx), config, "remote");
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    return luisa::compute::rust::destroy(device);
}
