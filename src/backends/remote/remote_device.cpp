#include <backends/common/rust_device_common.h>
#include <backends/common/rust_device_common.h>
#include "cpu_device.h"

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return new luisa::compute::rust::RustDevice{std::move(ctx), "remote"};
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    delete device;
}
