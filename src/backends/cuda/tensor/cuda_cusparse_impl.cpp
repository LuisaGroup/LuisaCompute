#include "cuda_las.h"
#include <luisa/tensor/tensor.h>

#include "../utils/cusparse_check.h"
#include "../cuda_device.h"
#include "../cuda_buffer.h"

#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "cuda_tensor_res.h"

namespace luisa::compute::cuda::tensor {

using namespace luisa::compute::tensor;
static cusparseOperation_t cusparse_enum_map(luisa::compute::tensor::MatrixOperation op) noexcept;

static cusparseDnVecDescr_t dn_vec_desc(const DTensor &tensor) { return dynamic_cast<CusparseDnVecDescRes *>(tensor.backend_tensor_res())->desc_handle(); }
static cusparseSpVecDescr_t sp_vec_desc(const DTensor &tensor) { return dynamic_cast<CusparseSpVecDescRes *>(tensor.backend_tensor_res())->desc_handle(); }
static cusparseDnMatDescr_t dn_mat_desc(const DTensor &tensor) { return dynamic_cast<CusparseDnMatDescRes *>(tensor.backend_tensor_res())->desc_handle(); }
static cusparseSpMatDescr_t sp_mat_desc(const DTensor &tensor) { return dynamic_cast<CusparseSpMatDescRes *>(tensor.backend_tensor_res())->desc_handle(); }

// Cusparse Impl
void CudaLAS::sparse_axpby(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_vec_x, const DTensor &beta) noexcept {
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    auto alpha_ = alpha.scalar_view();
    auto sp_vec_x_ = sp_vec_desc(sp_vec_x);
    auto beta_ = beta.scalar_view();
    LUISA_CHECK_CUSPARSE(
        cusparseAxpby(_cusparse_handle, raw_ptr(alpha_), sp_vec_x_, raw_ptr(beta_), dn_vec_y_));
}

void CudaLAS::gather(DTensor &sp_vec_x, const DTensor &dn_vec_y) noexcept {
    auto sp_vec_x_ = sp_vec_desc(sp_vec_x);
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    LUISA_CHECK_CUSPARSE(
        cusparseGather(_cusparse_handle, dn_vec_y_, sp_vec_x_));
}

void CudaLAS::scatter(DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept {
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    auto sp_vec_x_ = sp_vec_desc(sp_vec_x);
    LUISA_CHECK_CUSPARSE(
        cusparseScatter(_cusparse_handle, sp_vec_x_, dn_vec_y_));
}

size_t CudaLAS::spvv_buffer_size(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x) noexcept {
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    auto sp_vec_x_ = sp_vec_desc(sp_vec_x);
    auto r = result.scalar_view();
    size_t ext_buffer_size = 0;
    LUISA_CHECK_CUSPARSE(
        cusparseSpVV_bufferSize(
            _cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, sp_vec_x_, dn_vec_y_, raw_ptr(r),
            cuda_enum_map(result.basic_data_type()), &ext_buffer_size));
}

void CudaLAS::spvv(DTensor &result, const DTensor &dn_vec_y, const DTensor &sp_vec_x, DenseStorageView ext_buffer) noexcept {
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    auto sp_vec_x_ = sp_vec_desc(sp_vec_x);
    auto r = result.scalar_view();
    LUISA_CHECK_CUSPARSE(
        cusparseSpVV(
            _cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, sp_vec_x_, dn_vec_y_, raw_ptr(r),
            cuda_enum_map(result.basic_data_type()), raw_ptr(ext_buffer)));
}

size_t CudaLAS::spmv_buffer_size(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta) noexcept {
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    auto alpha_ = alpha.scalar_view();
    auto sp_mat_A_ = sp_mat_desc(sp_mat_A);
    auto A_view = sp_mat_A.sparse_matrix_view();
    auto dn_vec_x_ = dn_vec_desc(dn_vec_x);
    auto beta_ = beta.scalar_view();
    size_t ext_buffer_size = 0;
    LUISA_CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            _cusparse_handle,
            cusparse_enum_map(A_view.operation), raw_ptr(alpha_), sp_mat_A_, dn_vec_x_, raw_ptr(beta_), dn_vec_y_,
            cuda_enum_map(dn_vec_y.basic_data_type()), CUSPARSE_MV_ALG_DEFAULT, &ext_buffer_size));
}

void CudaLAS::spmv(DTensor &dn_vec_y, const DTensor &alpha, const DTensor &sp_mat_A, const DTensor &dn_vec_x, const DTensor &beta, DenseStorageView ext_buffer) noexcept {
    auto dn_vec_y_ = dn_vec_desc(dn_vec_y);
    auto alpha_ = alpha.scalar_view();
    auto sp_mat_A_ = sp_mat_desc(sp_mat_A);
    auto A_view = sp_mat_A.sparse_matrix_view();
    auto dn_vec_x_ = dn_vec_desc(dn_vec_x);
    auto beta_ = beta.scalar_view();

    LUISA_CHECK_CUSPARSE(
        cusparseSpMV(
            _cusparse_handle,
            cusparse_enum_map(A_view.operation), raw_ptr(alpha_), sp_mat_A_, dn_vec_x_, raw_ptr(beta_), dn_vec_y_,
            cuda_enum_map(dn_vec_y.basic_data_type()), CUSPARSE_MV_ALG_DEFAULT, raw_ptr(ext_buffer)));
}

cusparseOperation_t cusparse_enum_map(luisa::compute::tensor::MatrixOperation op) noexcept {
    cusparseOperation_t ret;
    switch (op) {
        case luisa::compute::tensor::MatrixOperation::NONE:
            ret = CUSPARSE_OPERATION_NON_TRANSPOSE;
            break;
        case luisa::compute::tensor::MatrixOperation::TRANS:
            ret = CUSPARSE_OPERATION_TRANSPOSE;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix operation mapping.");
            break;
    }
    return ret;
}
}// namespace luisa::compute::cuda::tensor