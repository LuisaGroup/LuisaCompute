#pragma once

#include <string_view>
#include <luisa/core/logging.h>
#include "optix_api.h"

#define LUISA_CHECK_OPTIX(...)                                          \
    do {                                                                \
        if (auto error = __VA_ARGS__; error != OPTIX_SUCCESS) {         \
            LUISA_ERROR_WITH_LOCATION(                                  \
                "{}: {}",                                               \
                optix::api().optixGetErrorName(error),                  \
                optix::api().optixGetErrorString(error));               \
        }                                                               \
    } while (false)

#define LUISA_CHECK_OPTIX_WITH_LOG(log, log_size, ...)                  \
    do {                                                                \
        log_size = sizeof(log) - 1u;                                    \
        if (auto error = __VA_ARGS__; error != OPTIX_SUCCESS) {         \
            using namespace std::string_view_literals;                  \
            LUISA_ERROR_WITH_LOCATION(                                  \
                "{}: {}\n{}{}",                                         \
                optix::api().optixGetErrorName(error),                  \
                optix::api().optixGetErrorString(error),                \
                static_cast<const char *>(log),                         \
                log_size > sizeof(log) ? " ..."sv : ""sv);              \
        }                                                               \
    } while (false)
