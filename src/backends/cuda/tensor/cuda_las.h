#pragma once
#include <luisa/tensor/las_interface.h>
#include "../cuda_stream.h"
#include <cublas_v2.h>
#include <cusparse.h>

namespace luisa::compute::cuda::tensor {
class CudaLAS : public luisa::compute::tensor::LASInterface {
    template<typename T>
    using S = luisa::shared_ptr<T>;

    using BackendTensorRes = luisa::compute::tensor::BackendTensorRes;
    using DTensor = luisa::compute::tensor::DTensor;
    using MatrixMulOptions = luisa::compute::tensor::MatrixMulOptions;
    using DenseStorageView = luisa::compute::tensor::DenseStorageView;

    CUDAStream *_stream{nullptr};
    cublasHandle_t _cublas_handle{nullptr};
    cusparseHandle_t _cusparse_handle{nullptr};

public:
    CudaLAS(CUDAStream *stream) noexcept;
    virtual ~CudaLAS() noexcept;
    virtual S<BackendTensorRes> alloc_backend_tensor_res(const DTensor &) noexcept override;

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
    virtual void sparse_axpby(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_vec_x, const DTensor &beta) noexcept override;
    virtual void gather(DTensor &sp_vec_x, const DTensor &dn_vec_y) noexcept override;
    virtual void scatter(DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept override;

    virtual size_t spvv_buffer_size(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept override;
    virtual void spvv(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x, DenseStorageView ext_buffer) noexcept override;

    // level-2
    virtual size_t spmv_buffer_size(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta) noexcept override;
    virtual void spmv(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta, DenseStorageView ext_buffer) noexcept override;
};
}// namespace luisa::compute::cuda::tensor