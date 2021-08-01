//
// Created by Mike on 7/30/2021.
//

#pragma once

#include <cuda.h>
#include <core/logging.h>

#define LUISA_CHECK_CUDA(...)                                  \
    [&] {                                                      \
        if (auto ec = __VA_ARGS__; ec != CUDA_SUCCESS) {       \
            const char *err = nullptr;                         \
            cuGetErrorString(ec, &err);                        \
            LUISA_ERROR_WITH_LOCATION("CUDA error: {}.", err); \
        }                                                      \
    }()
