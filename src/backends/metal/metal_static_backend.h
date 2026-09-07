#pragma once

#include <luisa/core/dll_export.h>

#if defined(LUISA_PLATFORM_IOS)
/// Registers the statically linked old Metal MSL backend with Context so an
/// iOS comparison application preserves the ordinary DeviceInterface path.
LUISA_EXPORT_API void
luisa_compute_metal_register_static_backend() LUISA_NOEXCEPT;
#endif
