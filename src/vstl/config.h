#pragma once
#include <core/platform.h>
#include <core/logging.h>
#define VSTL_ABORT() LUISA_ERROR_WITH_LOCATION("vstl::abort()")

#if defined(UNICODE) && !defined(VSTL_UNICODE)
#define VSTL_UNICODE
#endif

#ifdef _MSC_VER
#define VSTL_EXPORT_C extern "C" _declspec(dllexport)
#else
#define VSTL_EXPORT_C extern "C"
#endif

#if (defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)) && !defined(VSTL_DEBUG)
#define VSTL_DEBUG
#endif

#define VENGINE_C_FUNC_COMMON
#define VENGINE_EXIT std::abort()

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <stdint.h>
using uint = uint32_t;
using uint64 = uint64_t;
using int64 = int64_t;
using int32 = int32_t;
using vbyte = uint8_t;
