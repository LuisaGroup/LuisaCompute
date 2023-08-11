#pragma once
#include "view.h"
#include <luisa/core/stl/memory.h>

namespace luisa::compute::tensor {

// linear algebric subroutine
class DTensor;

class BackendTensorRes;

class LASInterface {
    template<typename T>
    using S = luisa::shared_ptr<T>;
public:
    // The backend may need to create some data structure to describe the tensor.
    // Tensor<T> should keep the handle and release the handle at the end of its life cycle.
    virtual S<BackendTensorRes> alloc_backend_tensor_res(const DTensor &) noexcept { return nullptr; }

    virtual uint64_t create_backend_tensor_res(const DTensor &) noexcept { return 0; }
    virtual void destroy_backend_tensor_res(const DTensor &) noexcept {}

    // BLAS
    // level-1
    virtual void iamax(DTensor &result, const DTensor &vec_x) noexcept = 0;
    virtual void iamin(DTensor &result, const DTensor &vec_x) noexcept = 0;
    virtual void dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept = 0;
    virtual void nrm2(DTensor &result, const DTensor &vec_x) noexcept = 0;

    // level-2
    virtual void mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept = 0;
    virtual void sv(DTensor &x, const DTensor &A) noexcept = 0;
    virtual void mv_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept = 0;
    virtual void mv_strided_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept = 0;

    // level-3
    virtual void mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept = 0;
    virtual void sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept = 0;
    virtual void mm_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept = 0;
    virtual void mm_stride_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept = 0;
    virtual void sm_batched(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept = 0;
    // SPARSE

    // level-1
    virtual void sparse_axpby(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_vec_x, const DTensor &beta) noexcept = 0;
    virtual void gather(DTensor &sp_vec_x, const DTensor &dn_vec_y) noexcept = 0;
    virtual void scatter(DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept = 0;

    virtual size_t spvv_buffer_size(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept = 0;
    virtual void spvv(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x, DenseStorageView ext_buffer) noexcept = 0;

    // level-2
    virtual size_t spmv_buffer_size(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta) noexcept = 0;
    virtual void spmv(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta, DenseStorageView ext_buffer) noexcept = 0;
};
}// namespace luisa::compute::tensor