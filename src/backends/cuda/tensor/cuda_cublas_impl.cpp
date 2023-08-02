#include "cuda_las.h"
#include <luisa/tensor/tensor.h>
#include "../utils/cublas_check.h"

#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "cuda_tensor_res.h"

namespace luisa::compute::cuda::tensor {
static cublasOperation_t cublas_enum_map(luisa::compute::tensor::MatrixOperation op) noexcept;
static cublasFillMode_t cublas_enum_map(luisa::compute::tensor::DenseMatrixFillMode op) noexcept;
static cublasDiagType_t cublas_enum_map(luisa::compute::tensor::DenseMatrixDiagType op) noexcept;
static cublasSideMode_t cublas_enum_map(luisa::compute::tensor::MatrixMulOptions op) noexcept;

// Cublas Impl
void CudaLAS::Iamax(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasIsamax_v2(_cublas_handle,
                                       x.desc.n,
                                       raw<float>(x), x.desc.inc,
                                       raw<int>(r)));
}
void CudaLAS::Iamin(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasIsamin_v2(_cublas_handle,
                                       x.desc.n,
                                       raw<float>(x), x.desc.inc,
                                       raw<int>(r)));
}
void CudaLAS::dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept {
    auto x = vec_x.dense_vector_view();
    auto y = vec_y.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasSdot_v2(_cublas_handle,
                                     x.desc.n,
                                     raw<float>(x), x.desc.inc,
                                     raw<float>(y), y.desc.inc,
                                     raw<float>(r)));
}
void CudaLAS::nrm2(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasSnrm2_v2(_cublas_handle,
                                      x.desc.n,
                                      raw<float>(x), x.desc.inc,
                                      raw<float>(r)));
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
                cublasSgemv_v2(_cublas_handle,
                               cublas_enum_map(A_.operation),
                               A_.desc.row, A_.desc.col,
                               raw<float>(alpha_),
                               raw<float>(A_), A_.desc.lda,
                               raw<float>(x_), x_.desc.inc,
                               raw<float>(beta_),
                               raw<float>(y_), y_.desc.inc);
            } else if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC) {
                cublasSsymv_v2(_cublas_handle,
                               cublas_enum_map(A_.desc.fill_mode),
                               A_.desc.row,
                               raw<float>(alpha_),
                               raw<float>(A_), A_.desc.lda,
                               raw<float>(x_), x_.desc.inc,
                               raw<float>(beta_),
                               raw<float>(y_), y_.desc.inc);
            } else {
                LUISA_ERROR("unsupported matrix property: only NONE/SYMMETRIC are supported");
            }
        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR: {
            cublasStrmv_v2(_cublas_handle,
                           cublas_enum_map(A_.desc.fill_mode),
                           cublas_enum_map(A_.operation),
                           cublas_enum_map(A_.desc.diag_type),
                           A_.desc.row,
                           raw<float>(A_), A_.desc.lda,
                           raw<float>(x_), x_.desc.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::BAND: {
            if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE) {
                cublasSgbmv_v2(_cublas_handle,
                               cublas_enum_map(A_.operation),
                               A_.desc.row, A_.desc.col,
                               A_.desc.kl, A_.desc.ku,
                               raw<float>(alpha_),
                               raw<float>(A_), A_.desc.lda,
                               raw<float>(x_), x_.desc.inc,
                               raw<float>(beta_),
                               raw<float>(y_), y_.desc.inc);
            } else if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC) {
                cublasSsbmv_v2(_cublas_handle,
                               cublas_enum_map(A_.desc.fill_mode),
                               A_.desc.row, A_.desc.kl,
                               raw<float>(alpha_),
                               raw<float>(A_), A_.desc.lda,
                               raw<float>(x_), x_.desc.inc,
                               raw<float>(beta_),
                               raw<float>(y_), y_.desc.inc);
            } else {
                LUISA_ERROR("unsupported matrix property: only NONE/SYMMETRIC are supported");
            };

        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR_BAND: {
            cublasStbmv_v2(_cublas_handle,
                           cublas_enum_map(A_.desc.fill_mode),
                           cublas_enum_map(A_.operation),
                           cublas_enum_map(A_.desc.diag_type),
                           A_.desc.row, A_.desc.kl,
                           raw<float>(A_), A_.desc.lda,
                           raw<float>(x_), x_.desc.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED_TRIANGULAR: {
            cublasStpmv_v2(_cublas_handle,
                           cublas_enum_map(A_.desc.fill_mode),
                           cublas_enum_map(A_.operation),
                           cublas_enum_map(A_.desc.diag_type),
                           A_.desc.row,
                           raw<float>(A_),
                           raw<float>(x_), x_.desc.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED: {
            LUISA_ASSERT(A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC, "only symmetric matrix can be packed.");
            cublasSspmv_v2(_cublas_handle,
                           cublas_enum_map(A_.desc.fill_mode),
                           A_.desc.row,
                           raw<float>(alpha_),
                           raw<float>(A_),
                           raw<float>(x_), x_.desc.inc,
                           raw<float>(beta_),
                           raw<float>(y_), y_.desc.inc);

        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
            break;
    }
}

void CudaLAS::sv(DTensor &x, const DTensor &A) noexcept {
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
        cublasSgemm_v2(_cublas_handle,
                       cublas_enum_map(A_.operation),
                       cublas_enum_map(B_.operation),
                       C_.desc.row, C_.desc.col, A_.desc.col,
                       raw<float>(alpha_),
                       raw<float>(A_), A_.desc.lda,
                       raw<float>(B_), B_.desc.lda,
                       raw<float>(beta_),
                       raw<float>(C_), C_.desc.lda);
    } else if (A_.desc.shape == DenseMatrixShape::TRIANGULAR && B_.desc.shape == DenseMatrixShape::GENERAL) {
        cublasStrmm_v2(_cublas_handle,
                       cublas_enum_map(options),
                       cublas_enum_map(A_.desc.fill_mode),
                       cublas_enum_map(A_.operation),
                       cublas_enum_map(A_.desc.diag_type),
                       C_.desc.row, C_.desc.col,
                       raw<float>(alpha_),
                       raw<float>(A_), A_.desc.lda,
                       raw<float>(B_), B_.desc.lda,
                       raw<float>(C_), C_.desc.lda);
    } else {
        LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
    }
}

void CudaLAS::sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept {
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