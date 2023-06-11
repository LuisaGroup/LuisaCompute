#pragma once

#include <luisa/runtime/rhi/device_interface.h>

namespace luisa::compute::rust {

luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx, const luisa::compute::DeviceConfig *config, luisa::string_view name) noexcept;
void destroy(luisa::compute::DeviceInterface *device) noexcept;
}// namespace luisa::compute::rust

