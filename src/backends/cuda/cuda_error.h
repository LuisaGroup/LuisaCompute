//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>
#include <nvrtc.h>
#include <optix.h>

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

#define LUISA_CHECK_OPTIX(...)                                  \
    [&] {                                                       \
        if (auto error = __VA_ARGS__; error != OPTIX_SUCCESS) { \
            LUISA_ERROR_WITH_LOCATION(                          \
                "{}: {}.",                                      \
                optixGetErrorName(error),                       \
                optixGetErrorString(error));                    \
        }                                                       \
    }()
