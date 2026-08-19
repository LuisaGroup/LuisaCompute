#pragma once
#ifdef LUISA_BIN_2_OBJ
extern "C" {
#include <cstdint>
#define LUISA_CUDA_DECL_VARNAME(VAR_NAME)              \
    extern const uint8_t _binary_##VAR_NAME##_start[]; \
    extern const uint8_t _binary_##VAR_NAME##_end[];

LUISA_CUDA_DECL_VARNAME(cuda_builtin_kernels_cu)
#define luisa_compute_cuda_builtin_kernels ((unsigned char *)_binary_cuda_builtin_kernels_cu_start)
#define luisa_compute_cuda_builtin_kernels_size ((unsigned long long)(_binary_cuda_builtin_kernels_cu_end - _binary_cuda_builtin_kernels_cu_start))
LUISA_CUDA_DECL_VARNAME(cuda_device_coop_h)
#define luisa_compute_cuda_device_coop ((unsigned char *)_binary_cuda_device_coop_h_start)
#define luisa_compute_cuda_device_coop_size ((unsigned long long)(_binary_cuda_device_coop_h_end - _binary_cuda_device_coop_h_start))
LUISA_CUDA_DECL_VARNAME(cuda_device_half_h)
#define luisa_compute_cuda_device_half ((unsigned char *)_binary_cuda_device_half_h_start)
#define luisa_compute_cuda_device_half_size ((unsigned long long)(_binary_cuda_device_half_h_end - _binary_cuda_device_half_h_start))
LUISA_CUDA_DECL_VARNAME(cuda_device_math_h)
#define luisa_compute_cuda_device_math ((unsigned char *)_binary_cuda_device_math_h_start)
#define luisa_compute_cuda_device_math_size ((unsigned long long)(_binary_cuda_device_math_h_end - _binary_cuda_device_math_h_start))
LUISA_CUDA_DECL_VARNAME(cuda_device_resource_h)
#define luisa_compute_cuda_device_resource ((unsigned char *)_binary_cuda_device_resource_h_start)
#define luisa_compute_cuda_device_resource_size ((unsigned long long)(_binary_cuda_device_resource_h_end - _binary_cuda_device_resource_h_start))
LUISA_CUDA_DECL_VARNAME(cuda_device_tensor_h)
#define luisa_compute_cuda_device_tensor ((unsigned char *)_binary_cuda_device_tensor_h_start)
#define luisa_compute_cuda_device_tensor_size ((unsigned long long)(_binary_cuda_device_tensor_h_end - _binary_cuda_device_tensor_h_start))
}
#else
#include "cuda_builtin_embedded.h"
#endif