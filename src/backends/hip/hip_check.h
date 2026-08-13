//
// Created by mike on 12/25/25.
//

#pragma once
#pragma once

#include <string_view>
#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

// ROCm exposes CUDA keyword compatibility macros to ordinary host compilers.
// Its empty GCC definition of __noinline__ corrupts libstdc++ 16 declarations
// such as [[__gnu__::__noinline__]] when standard headers are included later.
// The HIP/HIPRT headers have already consumed the keyword at this point, and
// Luisa's host implementation does not use it.
#if defined(__GNUC__) && !defined(__clang__) && defined(__noinline__)
#undef __noinline__
#endif

#include <luisa/core/magic_enum.h>
#include <luisa/core/logging.h>

#define LUISA_CHECK_HIP(...)                             \
    do {                                                 \
        if (auto ec = __VA_ARGS__; ec != hipSuccess) {   \
            auto err_name = hipGetErrorName(ec);         \
            auto err_string = hipGetErrorString(ec);     \
            if (!err_string) { err_string = "unknown"; } \
            LUISA_ERROR_WITH_LOCATION(                   \
                "{}: {}", err_name, err_string);         \
        }                                                \
    } while (false)

#define LUISA_CHECK_HIPRT(...)                           \
    do {                                                 \
        if (auto ec = __VA_ARGS__; ec != hiprtSuccess) { \
            LUISA_ERROR_WITH_LOCATION(                   \
                "HIPRT error: {} (code = {})",           \
                luisa::to_string(ec),                    \
                luisa::to_underlying(ec));               \
        }                                                \
    } while (false)
