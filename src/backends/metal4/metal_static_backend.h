#pragma once

#include <luisa/core/dll_export.h>

#if defined(LUISA_PLATFORM_IOS)
/// Registers the statically linked Metal4 backend with Context so ordinary
/// create_device("metal4") calls work inside a signed iOS application.
LUISA_EXPORT_API void
luisa_compute_metal4_register_static_backend() LUISA_NOEXCEPT;
#endif
