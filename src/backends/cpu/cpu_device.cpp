#include <luisa/core/platform.h>

#include "../common/rust_device_common.h"
#include "cpu_device.h"

#ifdef LUISA_ARCH_ARM64
#include <arm_neon.h>
#else

#endif

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::compute::rust::create(std::move(ctx), config, "cpu");
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    return luisa::compute::rust::destroy(device);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    names.emplace_back(luisa::cpu_name());
}

