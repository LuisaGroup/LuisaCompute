#pragma once
#ifdef LUISA_BIN_2_OBJ
#define LUISA_CUDA_DECL_VARNAME(VAR_NAME)              \
    extern const uint8_t _binary_##VAR_NAME##_start[]; \
    extern const uint8_t _binary_##VAR_NAME##_end[];

LUISA_CUDA_DECL_VARNAME(metal_builtin_kernels_metal)
#define luisa_compute_metal_builtin_kernels ((const char *)_binary_metal_builtin_kernels_metal_start)
#define luisa_compute_metal_builtin_kernels_size ((unsigned long long)(_binary_metal_builtin_kernels_metal_end - _binary_metal_builtin_kernels_metal_start))

LUISA_CUDA_DECL_VARNAME(metal_device_lib_metal)
#define luisa_compute_metal_device_lib ((const char *)_binary_metal_device_lib_metal_start)
#define luisa_compute_metal_device_lib_size ((unsigned long long)(_binary_metal_device_lib_metal_end - _binary_metal_device_lib_metal_start))

#else
#include "metal_builtin_embedded.h"
#endif