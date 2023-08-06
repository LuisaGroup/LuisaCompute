#include "cuda_las.h"
#include <cublas_api.h>
#include <luisa/tensor/tensor.h>
#include "../utils/cublas_check.h"

#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "cuda_tensor_res.h"
#include "week_type_ex.h"

namespace luisa::compute::cuda::tensor {
static cublasOperation_t cublas_enum_map(luisa::compute::tensor::MatrixOperation op) noexcept;
static cublasFillMode_t cublas_enum_map(luisa::compute::tensor::DenseMatrixFillMode op) noexcept;
static cublasDiagType_t cublas_enum_map(luisa::compute::tensor::DenseMatrixDiagType op) noexcept;
static cublasSideMode_t cublas_enum_map(luisa::compute::tensor::MatrixMulOptions op) noexcept;

// Cublas Impl
void CudaLAS::Iamax(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(
        cublasIamaxEx(_cublas_handle,
                      x.n, raw_ptr(x),
                      cuda_enum_map(vec_x.basic_data_type()),
                      x.desc.inc, raw<int>(r)));
}

void CudaLAS::Iamin(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(
        cublasIaminEx(_cublas_handle,
                      x.n, raw_ptr(x),
                      cuda_enum_map(vec_x.basic_data_type()),
                      x.desc.inc, raw<int>(r)));
}

void CudaLAS::dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept {
    auto x = vec_x.dense_vector_view();
    auto y = vec_y.dense_vector_view();
    auto r = result.scalar_view();

    LUISA_CHECK_CUBLAS(
        cublasDotEx(_cublas_handle,
                    x.n,
                    raw_ptr(x), cuda_enum_map(vec_x.basic_data_type()), x.desc.inc,
                    raw_ptr(y), cuda_enum_map(vec_y.basic_data_type()), y.desc.inc,
                    raw_ptr(r), cuda_enum_map(result.basic_data_type()),
                    cuda_enum_map(result.basic_data_type())));
}

void CudaLAS::nrm2(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();

    LUISA_CHECK_CUBLAS(
        cublasNrm2Ex(_cublas_handle,
                     x.n,
                     raw_ptr(x), cuda_enum_map(vec_x.basic_data_type()), x.desc.inc,
                     raw_ptr(r), cuda_enum_map(result.basic_data_type()), cuda_enum_map(result.basic_data_type())));
}

void CudaLAS::mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto x_ = x.dense_vector_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();

    switch (A_.desc.shape) {
        case luisa::compute::tensor::DenseMatrixShape::GENERAL: {
            if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE) {
                LUISA_CHECK_CUBLAS(
                    gemv_ex(_cublas_handle,
                            cuda_enum_map(A.basic_data_type()),
                            cublas_enum_map(A_.operation),
                            A_.row, A_.col,
                            raw_ptr(alpha_),
                            raw_ptr(A_), A_.desc.lda,
                            raw_ptr(x_), x_.desc.inc,
                            raw_ptr(beta_),
                            raw_ptr(y_), y_.desc.inc));

            } else if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC) {
                LUISA_CHECK_CUBLAS(
                    symv_ex(_cublas_handle,
                            cuda_enum_map(A.basic_data_type()),
                            cublas_enum_map(A_.desc.fill_mode),
                            A_.row,
                            raw_ptr(alpha_),
                            raw_ptr(A_), A_.desc.lda,
                            raw_ptr(x_), x_.desc.inc,
                            raw_ptr(beta_),
                            raw_ptr(y_), y_.desc.inc));
            } else {
                LUISA_ERROR("unsupported matrix property: only NONE/SYMMETRIC are supported");
            }
        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR: {
            LUISA_CHECK_CUBLAS(
                trmv_ex(_cublas_handle,
                        cuda_enum_map(A.basic_data_type()),
                        cublas_enum_map(A_.desc.fill_mode),
                        cublas_enum_map(A_.operation),
                        cublas_enum_map(A_.desc.diag_type),
                        A_.row,
                        raw_ptr(A_), A_.desc.lda,
                        raw_ptr(x_), x_.desc.inc));
        } break;
        case luisa::compute::tensor::DenseMatrixShape::BAND: {
            if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE) {
                LUISA_CHECK_CUBLAS(
                    gbmv_ex(_cublas_handle,
                            cuda_enum_map(A.basic_data_type()),
                            cublas_enum_map(A_.operation),
                            A_.row, A_.col,
                            A_.desc.kl, A_.desc.ku,
                            raw_ptr(alpha_),
                            raw_ptr(A_), A_.desc.lda,
                            raw_ptr(x_), x_.desc.inc,
                            raw_ptr(beta_),
                            raw_ptr(y_), y_.desc.inc));
            } else if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC) {
                LUISA_CHECK_CUBLAS(
                    sbmv_ex(_cublas_handle,
                            cuda_enum_map(A.basic_data_type()),
                            cublas_enum_map(A_.desc.fill_mode),
                            A_.row, A_.desc.kl,
                            raw_ptr(alpha_),
                            raw_ptr(A_), A_.desc.lda,
                            raw_ptr(x_), x_.desc.inc,
                            raw_ptr(beta_),
                            raw_ptr(y_), y_.desc.inc));
            } else {
                LUISA_ERROR("unsupported matrix property: only NONE/SYMMETRIC are supported");
            };

        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR_BAND: {
            LUISA_CHECK_CUBLAS(
                tbmv_ex(_cublas_handle,
                        cuda_enum_map(A.basic_data_type()),
                        cublas_enum_map(A_.desc.fill_mode),
                        cublas_enum_map(A_.operation),
                        cublas_enum_map(A_.desc.diag_type),
                        A_.row, A_.desc.kl,
                        raw_ptr(A_), A_.desc.lda,
                        raw_ptr(x_), x_.desc.inc));
        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED_TRIANGULAR: {
            LUISA_CHECK_CUBLAS(
                tpmv_ex(_cublas_handle,
                        cuda_enum_map(A.basic_data_type()),
                        cublas_enum_map(A_.desc.fill_mode),
                        cublas_enum_map(A_.operation),
                        cublas_enum_map(A_.desc.diag_type),
                        A_.row,
                        raw_ptr(A_),
                        raw_ptr(x_), x_.desc.inc));
        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED: {
            LUISA_ASSERT(A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC, "only symmetric matrix can be packed.");
            LUISA_CHECK_CUBLAS(
                spmv_ex(_cublas_handle,
                        cuda_enum_map(A.basic_data_type()),
                        cublas_enum_map(A_.desc.fill_mode),
                        A_.row,
                        raw_ptr(alpha_),
                        raw_ptr(A_),
                        raw_ptr(x_), x_.desc.inc,
                        raw_ptr(beta_),
                        raw_ptr(y_), y_.desc.inc));

        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
            break;
    }
}

void CudaLAS::sv(DTensor &x, const DTensor &A) noexcept {
    LUISA_ERROR("NO IMPL YET");
}

void CudaLAS::mv_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto A_array = A.batch_view();
    auto x_ = x.dense_vector_view();
    auto x_array = x.batch_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();
    auto y_array = y.batch_view();
    using namespace luisa::compute::tensor;

    // LUISA_ASSERT(A.basic_data_type() == TensorBasicDataType::FLOAT32, "only float32 is supported.");
    LUISA_ASSERT(A_.desc.shape == DenseMatrixShape::GENERAL, "only general matrix is supported.");
    LUISA_ASSERT(A_.storage.size() == x_.storage.size() && A_.storage.size() == y_.storage.size(), "A, x, y must have the same batch size.");
    LUISA_CHECK_CUBLAS(
        gemv_batched_ex(_cublas_handle,
                        cuda_enum_map(A.basic_data_type()),
                        cublas_enum_map(A_.operation),
                        A_.row, A_.col,
                        raw<void>(alpha_),
                        raw<void>(A_array), A_.desc.lda,
                        raw<void>(x_array), x_.desc.inc,
                        raw<void>(beta_),
                        raw<void>(y_array), y_.desc.inc,
                        A_array.desc._batch_count));
}

void CudaLAS::mv_strided_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto A_array = A.batch_view();
    auto x_ = x.dense_vector_view();
    auto x_array = x.batch_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();
    auto y_array = y.batch_view();
    using namespace luisa::compute::tensor;

    // LUISA_ASSERT(A.basic_data_type() == TensorBasicDataType::FLOAT32, "only float32 is supported.");
    LUISA_ASSERT(A_.desc.shape == DenseMatrixShape::GENERAL, "only general matrix is supported.");

    LUISA_CHECK_CUBLAS(
        gemv_strided_batched_ex(_cublas_handle,
                                cuda_enum_map(A.basic_data_type()),
                                cublas_enum_map(A_.operation),
                                A_.row, A_.col,
                                raw_ptr(alpha_),
                                raw_ptr(A_), A_.desc.lda, A_array.desc._batch_stride,
                                raw_ptr(x_), x_.desc.inc, x_array.desc._batch_stride,
                                raw_ptr(beta_),
                                raw_ptr(y_), y_.desc.inc, y_array.desc._batch_stride,
                                A_array.desc._batch_count));
}

void CudaLAS::mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto B_ = B.dense_matrix_view();
    auto beta_ = beta.scalar_view();
    auto C_ = C.dense_matrix_view();
    using DenseMatrixShape = luisa::compute::tensor::DenseMatrixShape;

    LUISA_ASSERT(C_.desc.shape == DenseMatrixShape::GENERAL, "only general matrix is supported for C.");
    if (A_.desc.shape == DenseMatrixShape::GENERAL && B_.desc.shape == DenseMatrixShape::GENERAL) {
        LUISA_CHECK_CUBLAS(
            cublasGemmEx(
                _cublas_handle,
                cublas_enum_map(A_.operation),
                cublas_enum_map(B_.operation),
                C_.row, C_.col, A_.col,
                raw<void>(alpha_),
                raw<void>(A_), cuda_enum_map(A.basic_data_type()), A_.desc.lda,
                raw<void>(B_), cuda_enum_map(B.basic_data_type()), B_.desc.lda,
                raw<void>(beta_),
                raw<void>(C_), cuda_enum_map(C.basic_data_type()), C_.desc.lda,
                cuda_enum_map(A.basic_data_type()),
                cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP));

    } else if (A_.desc.shape == DenseMatrixShape::TRIANGULAR && B_.desc.shape == DenseMatrixShape::GENERAL) {
        LUISA_CHECK_CUBLAS(
            trmm_ex(_cublas_handle,
                    cuda_enum_map(A.basic_data_type()),
                    cublas_enum_map(options),
                    cublas_enum_map(A_.desc.fill_mode),
                    cublas_enum_map(A_.operation),
                    cublas_enum_map(A_.desc.diag_type),
                    C_.row, C_.col,
                    raw<void>(alpha_),
                    raw<void>(A_), A_.desc.lda,
                    raw<void>(B_), B_.desc.lda,
                    raw<void>(C_), C_.desc.lda));
    } else {
        LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
    }
}

void CudaLAS::sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept {
    LUISA_ERROR("NO IMPL YET");
}

void CudaLAS::mm_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto A_array = A.batch_view();
    auto B_ = B.dense_matrix_view();
    auto B_array = B.batch_view();
    auto beta_ = beta.scalar_view();
    auto C_ = C.dense_matrix_view();
    auto C_array = C.batch_view();
    using namespace luisa::compute::tensor;

    LUISA_ASSERT(A_.desc.shape == DenseMatrixShape::GENERAL &&
                     B_.desc.shape == DenseMatrixShape::GENERAL &&
                     C_.desc.shape == DenseMatrixShape::GENERAL,
                 "only general matrix is supported for A, B, C.");
    LUISA_ASSERT(A_.storage.size() == B_.storage.size() && A_.storage.size() == C_.storage.size(),
                 "A, B, C must have the same batch size.");

    LUISA_CHECK_CUBLAS(
        cublasGemmBatchedEx(_cublas_handle,
                            cublas_enum_map(A_.operation),
                            cublas_enum_map(B_.operation),
                            C_.row, C_.col, A_.col,
                            raw<float>(alpha_),
                            raw<void>(A_array), cuda_enum_map(A.basic_data_type()), A_.desc.lda,
                            raw<void>(B_array), cuda_enum_map(B.basic_data_type()), B_.desc.lda,
                            raw<float>(beta_),
                            raw<void>(C_array), cuda_enum_map(C.basic_data_type()), C_.desc.lda,
                            A_.storage.size(),
                            cuda_enum_map(A.basic_data_type()),
                            cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP));
}

void CudaLAS::mm_stride_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto A_array = A.batch_view();
    auto B_ = B.dense_matrix_view();
    auto B_array = B.batch_view();
    auto beta_ = beta.scalar_view();
    auto C_ = C.dense_matrix_view();
    auto C_array = C.batch_view();
    using namespace luisa::compute::tensor;

    LUISA_ASSERT(A_.desc.shape == DenseMatrixShape::GENERAL &&
                     B_.desc.shape == DenseMatrixShape::GENERAL &&
                     C_.desc.shape == DenseMatrixShape::GENERAL,
                 "only general matrix is supported for A, B, C.");
    LUISA_ASSERT(A_.storage.size() == B_.storage.size() && A_.storage.size() == C_.storage.size(),
                 "A, B, C must have the same batch size.");

    LUISA_CHECK_CUBLAS(
        cublasGemmStridedBatchedEx(
            _cublas_handle,
            cublas_enum_map(A_.operation),
            cublas_enum_map(B_.operation),
            C_.row, C_.col, A_.col,
            raw<float>(alpha_),
            raw<float>(A_), cuda_enum_map(A.basic_data_type()), A_.desc.lda, A_array.desc._batch_stride,
            raw<float>(B_), cuda_enum_map(B.basic_data_type()), B_.desc.lda, B_array.desc._batch_stride,
            raw<float>(beta_),
            raw<float>(C_), cuda_enum_map(C.basic_data_type()), C_.desc.lda, C_array.desc._batch_stride,
            A_array.desc._batch_count,
            cuda_enum_map(A.basic_data_type()),
            cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP));

    //);
}

cublasOperation_t cublas_enum_map(luisa::compute::tensor::MatrixOperation op) noexcept {
    cublasOperation_t ret;
    switch (op) {
        case luisa::compute::tensor::MatrixOperation::NONE:
            ret = CUBLAS_OP_N;
            break;
        case luisa::compute::tensor::MatrixOperation::TRANS:
            ret = CUBLAS_OP_T;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix operation mapping.");
            break;
    }
    return ret;
}

cublasFillMode_t cublas_enum_map(luisa::compute::tensor::DenseMatrixFillMode op) noexcept {
    cublasFillMode_t ret;
    switch (op) {
        case luisa::compute::tensor::DenseMatrixFillMode::UPPER:
            ret = CUBLAS_FILL_MODE_UPPER;
            break;
        case luisa::compute::tensor::DenseMatrixFillMode::LOWER:
            ret = CUBLAS_FILL_MODE_LOWER;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix fill mode mapping.");
            break;
    }
    return ret;
}

cublasDiagType_t cublas_enum_map(luisa::compute::tensor::DenseMatrixDiagType op) noexcept {
    cublasDiagType_t ret;
    switch (op) {
        case luisa::compute::tensor::DenseMatrixDiagType::NON_UNIT:
            ret = CUBLAS_DIAG_NON_UNIT;
        case luisa::compute::tensor::DenseMatrixDiagType::UNIT:
            ret = CUBLAS_DIAG_UNIT;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix diag type mapping.");
            break;
    }
    return ret;
}
cublasSideMode_t cublas_enum_map(luisa::compute::tensor::MatrixMulOptions op) noexcept {
    cublasSideMode_t ret;
    switch (op.side) {
        case luisa::compute::tensor::MatrixASide::LEFT:
            ret = CUBLAS_SIDE_LEFT;
            break;
        case luisa::compute::tensor::MatrixASide::RIGHT:
            ret = CUBLAS_SIDE_RIGHT;
            break;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix side mode mapping.");
            break;
    }
    return ret;
}
}// namespace luisa::compute::cuda::tensor