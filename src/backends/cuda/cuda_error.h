#pragma once

#include <string_view>

#include <cuda.h>
#include <nvrtc.h>
#include "optix_api.h"

#include <luisa/core/logging.h>

#define LUISA_CHECK_CUDA(...)                            \
    do {                                                 \
        if (auto ec = __VA_ARGS__; ec != CUDA_SUCCESS) { \
            const char *err_name = nullptr;              \
            const char *err_string = nullptr;            \
            cuGetErrorName(ec, &err_name);               \
            cuGetErrorString(ec, &err_string);           \
            LUISA_ERROR_WITH_LOCATION(                   \
                "{}: {}", err_name, err_string);         \
        }                                                \
    } while (false)

#define LUISA_CHECK_NVRTC(...)                            \
    do {                                                  \
        if (auto ec = __VA_ARGS__; ec != NVRTC_SUCCESS) { \
            LUISA_ERROR_WITH_LOCATION(                    \
                "NVRTC error: {}",                        \
                nvrtcGetErrorString(ec));                 \
        }                                                 \
    } while (false)

#define LUISA_CHECK_OPTIX(...)                       \
    do {                                             \
        if (auto error = __VA_ARGS__; error != 0u) { \
            LUISA_ERROR_WITH_LOCATION(               \
                "{}: {}",                            \
                optix::api().getErrorName(error),    \
                optix::api().getErrorString(error)); \
        }                                            \
    } while (false)

#define LUISA_CHECK_OPTIX_WITH_LOG(log, log_size, ...)     \
    do {                                                   \
        log_size = sizeof(log) - 1u;                       \
        if (auto error = __VA_ARGS__; error != 0u) {       \
            using namespace std::string_view_literals;     \
            LUISA_ERROR_WITH_LOCATION(                     \
                "{}: {}\n{}{}",                            \
                optix::api().getErrorName(error),          \
                optix::api().getErrorString(error),        \
                log,                                       \
                log_size > sizeof(log) ? " ..."sv : ""sv); \
        }                                                  \
    } while (false)
