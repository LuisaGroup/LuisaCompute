#pragma once
#include <luisa/tensor/las_interface.h>
#include "../cuda_stream.h"
#include <cublas_v2.h>
#include <cusparse.h>
namespace luisa::compute::cuda::tensor {
class CudaLAS : public luisa::compute::tensor::LASInterface {
    using DTensor = luisa::compute::tensor::DTensor;
    using MatrixMulOptions = luisa::compute::tensor::MatrixMulOptions;
private:
    CUDAStream *_stream{nullptr};
    cublasHandle_t _cublas_handle{nullptr};
    cusparseHandle_t _cusparse_handle{nullptr};

    static cublasOperation_t enum_map(luisa::compute::tensor::MatrixOperation op) noexcept;
    static cublasFillMode_t enum_map(luisa::compute::tensor::DenseMatrixFillMode op) noexcept;
    static cublasDiagType_t enum_map(luisa::compute::tensor::DenseMatrixDiagType op) noexcept;

public:
    CudaLAS(CUDAStream *stream) noexcept;
    virtual ~CudaLAS() noexcept;

    // BLAS
    // level-1
    
    // Note the result of Iamax and Iamin is an index starting from 1 (not 0, because of the fortran style)
    virtual void Iamax(DTensor &result, const DTensor &vec_x) noexcept override;
    virtual void Iamin(DTensor &result, const DTensor &vec_x) noexcept override;

    virtual void dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept override;
    virtual void nrm2(DTensor &result, const DTensor &vec_x) noexcept override;

    // level-2
    virtual void mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept override;
    virtual void sv(DTensor &x, const DTensor &A) noexcept override;

    // level-3
    virtual void mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept override;
    virtual void sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept override;

    // SPARSE
};
}// namespace luisa::compute::cuda::tensor