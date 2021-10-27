#pragma once

#include <core/platform.h>
#include <core/logging.h>

#define VSTL_ABORT() LUISA_ERROR_WITH_LOCATION("vstl::abort()")

#if defined(UNICODE) && !defined(VSTL_UNICODE)
#define VSTL_UNICODE
#endif

#if (defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)) && !defined(VSTL_DEBUG)
#define VSTL_DEBUG
#endif

#define VENGINE_DLL_COMMON
#define VENGINE_C_FUNC_COMMON
#define VENGINE_EXIT std::abort()

#define NOMINMAX