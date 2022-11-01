//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <string_view>

#include <cuda.h>
#include <nvrtc.h>
#include <backends/cuda/optix_api.h>

#include <core/logging.h>

#define LUISA_CHECK_CUDA(...)                                  \
    [&] {                                                      \
        if (auto ec = __VA_ARGS__; ec != CUDA_SUCCESS) {       \
            const char *err = nullptr;                         \
            cuGetErrorString(ec, &err);                        \
            LUISA_ERROR_WITH_LOCATION("CUDA error: {}.", err); \
        }                                                      \
    }()

#define LUISA_CHECK_NVRTC(...)                            \
    [&] {                                                 \
        if (auto ec = __VA_ARGS__; ec != NVRTC_SUCCESS) { \
            LUISA_ERROR_WITH_LOCATION(                    \
                "NVRTC error: {}.",                       \
                nvrtcGetErrorString(ec));                 \
        }                                                 \
    }()

#define LUISA_CHECK_OPTIX(...)                       \
    [&] {                                            \
        if (auto error = __VA_ARGS__; error != 0u) { \
            LUISA_ERROR_WITH_LOCATION(               \
                "{}: {}.",                           \
                optix::api().getErrorName(error),    \
                optix::api().getErrorString(error)); \
        }                                            \
    }()

#define LUISA_CHECK_OPTIX_WITH_LOG(log, log_size, ...)     \
    [&] {                                                  \
        log_size = sizeof(log);                            \
        if (auto error = __VA_ARGS__; error != 0u) {       \
            using namespace std::string_view_literals;     \
            LUISA_ERROR_WITH_LOCATION(                     \
                "{}: {}\n{}{}",                            \
                optix::api().getErrorName(error),          \
                optix::api().getErrorString(error),        \
                log,                                       \
                log_size > sizeof(log) ? " ..."sv : ""sv); \
        }                                                  \
    }()
