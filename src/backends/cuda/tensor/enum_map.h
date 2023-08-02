#pragma once

#include <cublas_v2.h>
#include <cusparse.h>

namespace luisa::compute::cuda::tensor {
cudaDataType_t cuda_enum_map(luisa::compute::tensor::TensorBasicDataType type) noexcept {
    cudaDataType_t ret;
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::INT32:
            ret = CUDA_R_32I;
            break;
        case luisa::compute::tensor::TensorBasicDataType::INT64:
            ret = CUDA_R_64I;
            break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32:
            ret = CUDA_R_32F;
            break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64:
            ret = CUDA_R_64F;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("error tensor basic data type mapping.");
            break;
    }
    return ret;
}
}