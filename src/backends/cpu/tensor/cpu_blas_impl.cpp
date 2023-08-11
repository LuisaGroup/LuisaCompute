#include "cpu_las.h"
#include <mkl.h>

#include "enum_map.h"
#include "raw_ptr_cast.h"
#include "cpu_tensor_res.h"
#include "week_type_ex.h"

namespace luisa::compute::cpu::tensor {
using namespace luisa::compute::tensor;

static CBLAS_TRANSPOSE cblas_enum_map(MatrixOperation op) noexcept;
static CBLAS_UPLO cblas_enum_map(DenseMatrixFillMode op) noexcept;
static CBLAS_DIAG cblas_enum_map(DenseMatrixDiagType op) noexcept;
static CBLAS_SIDE cblas_enum_map(MatrixMulOptions op) noexcept;

static constexpr auto cblas_layout() noexcept { return CblasColMajor; }
static constexpr auto blas_offset() noexcept { return CblasColOffset; }

void CpuLAS::iamax(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    auto type = vec_x.basic_data_type();
    invoke([=] {
        auto r_ptr = raw<int>(r);
        iamax_ex(type, r_ptr, x.n, raw<void>(x), x.desc.inc);
        ++(*r_ptr);// workaround, because in cublas iamax the index starts from 1
    });
}

void CpuLAS::iamin(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    auto type = vec_x.basic_data_type();
    invoke([=] {
        auto r_ptr = raw<int>(r);
        iamin_ex(type, r_ptr, x.n, raw<void>(x), x.desc.inc);
        ++(*r_ptr);// workaround, because in cublas iamax the index starts from 1
    });
}

void CpuLAS::dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept {
    auto x = vec_x.dense_vector_view();
    auto y = vec_y.dense_vector_view();
    auto r = result.scalar_view();
    auto type = vec_x.basic_data_type();
    invoke([=] {
        dot_ex(type, raw<void>(r), x.n, raw<void>(x), x.desc.inc, raw<void>(y), y.desc.inc);
    });
}

void CpuLAS::nrm2(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    auto type = vec_x.basic_data_type();
    invoke([=] {
        nrm2_ex(type, raw<void>(r), x.n, raw<void>(x), x.desc.inc);
    });
}

void CpuLAS::mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto x_ = x.dense_vector_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();
    auto type = A.basic_data_type();

    switch (A_.desc.shape) {
        case luisa::compute::tensor::DenseMatrixShape::GENERAL: {
            if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE) {
                invoke([=] {
                    gemv_ex(type,
                            cblas_layout(),
                            cblas_enum_map(A_.operation),
                            A_.row, A_.col,
                            raw<void>(alpha_),
                            raw<void>(A_), A_.desc.lda,
                            raw<void>(x_), x_.desc.inc,
                            raw<void>(beta_),
                            raw<void>(y_), y_.desc.inc);
                });
            } else if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC) {
                invoke([=] {
                    symv_ex(type,
                            cblas_layout(),
                            cblas_enum_map(A_.desc.fill_mode),
                            A_.row,
                            raw<void>(alpha_),
                            raw<void>(A_), A_.desc.lda,
                            raw<void>(x_), x_.desc.inc,
                            raw<void>(beta_),
                            raw<void>(y_), y_.desc.inc);
                });
            } else {
                LUISA_ERROR("unsupported matrix property: only NONE/SYMMETRIC are supported");
            }
        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR: {
            LUISA_ASSERT(A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE,
                         "triangular matrix can't be symmetric");
            invoke([=] {
                trmv_ex(type,
                        cblas_layout(),
                        cblas_enum_map(A_.desc.fill_mode),
                        cblas_enum_map(A_.operation),
                        cblas_enum_map(A_.desc.diag_type),
                        A_.row,
                        raw<void>(A_), A_.desc.lda,
                        raw<void>(x_), x_.desc.inc);
            });
        } break;
        case luisa::compute::tensor::DenseMatrixShape::BAND: {
            if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE) {
                invoke([=] {
                    gbmv_ex(type,
                            cblas_layout(),
                            cblas_enum_map(A_.operation),
                            A_.row, A_.col,
                            A_.desc.kl, A_.desc.ku,
                            raw<void>(alpha_),
                            raw<void>(A_), A_.desc.lda,
                            raw<void>(x_), x_.desc.inc,
                            raw<void>(beta_),
                            raw<void>(y_), y_.desc.inc);
                });
            } else if (A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC) {
                invoke([=] {
                    sbmv_ex(type,
                            cblas_layout(),
                            cblas_enum_map(A_.desc.fill_mode),
                            A_.row, A_.desc.kl,
                            raw<void>(alpha_),
                            raw<void>(A_), A_.desc.lda,
                            raw<void>(x_), x_.desc.inc,
                            raw<void>(beta_),
                            raw<void>(y_), y_.desc.inc);
                });
            } else {
                LUISA_ERROR("unsupported matrix property: only NONE/SYMMETRIC are supported");
            };

        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR_BAND: {
            LUISA_ASSERT(A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE,
                         "triangular matrix can't be symmetric");

            invoke([=] {
                tbmv_ex(type,
                        cblas_layout(),
                        cblas_enum_map(A_.desc.fill_mode),
                        cblas_enum_map(A_.operation),
                        cblas_enum_map(A_.desc.diag_type),
                        A_.row, A_.desc.kl,
                        raw<void>(A_), A_.desc.lda,
                        raw<void>(x_), x_.desc.inc);
            });

        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED_TRIANGULAR: {
            LUISA_ASSERT(A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::NONE,
                         "triangular matrix can't be symmetric");

            invoke([=] {
                tpmv_ex(type,
                        cblas_layout(),
                        cblas_enum_map(A_.desc.fill_mode),
                        cblas_enum_map(A_.operation),
                        cblas_enum_map(A_.desc.diag_type),
                        A_.row,
                        raw<void>(A_),
                        raw<void>(x_), x_.desc.inc);
            });

        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED: {
            LUISA_ASSERT(A_.desc.property == luisa::compute::tensor::DenseMatrixProperty::SYMMETRIC,
                         "only symmetric matrix can be packed.");

            invoke([=] {
                tpmv_ex(type,
                        cblas_layout(),
                        cblas_enum_map(A_.desc.fill_mode),
                        cblas_enum_map(A_.operation),
                        cblas_enum_map(A_.desc.diag_type),
                        A_.row,
                        raw<void>(A_),
                        raw<void>(x_), x_.desc.inc);
            });

        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
            break;
    }
}

void CpuLAS::sv(DTensor &x, const DTensor &A) noexcept {
    auto A_ = A.dense_matrix_view();
    auto x_ = x.dense_vector_view();
    auto type = A.basic_data_type();

    switch (A_.desc.shape) {
        case DenseMatrixShape::TRIANGULAR: {
            trsv_ex(type,
                    cblas_layout(),
                    cblas_enum_map(A_.desc.fill_mode),
                    cblas_enum_map(A_.operation),
                    cblas_enum_map(A_.desc.diag_type),
                    A_.row,
                    raw<void>(A_), A_.desc.lda,
                    raw<void>(x_), x_.desc.inc);
        } break;
        case DenseMatrixShape::TRIANGULAR_BAND: {
            tbsv_ex(type,
                    cblas_layout(),
                    cblas_enum_map(A_.desc.fill_mode),
                    cblas_enum_map(A_.operation),
                    cblas_enum_map(A_.desc.diag_type),
                    A_.row, A_.desc.kl,
                    raw<void>(A_), A_.desc.lda,
                    raw<void>(x_), x_.desc.inc);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape for SolveVector(sv).");
            break;
    }
}

void CpuLAS::mv_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto A_array = A.batch_view();
    auto x_ = x.dense_vector_view();
    auto x_array = x.batch_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();
    auto y_array = y.batch_view();
    auto type = A.basic_data_type();

    LUISA_ASSERT(A_.desc.shape == DenseMatrixShape::GENERAL, "only general matrix is supported.");
    LUISA_ASSERT(A_.storage.size() == x_.storage.size() && A_.storage.size() == y_.storage.size(), "A, x, y must have the same batch size.");
    invoke([=] {
        gemv_batch_ex(type,
                      cblas_layout(),
                      cblas_enum_map(A_.operation),
                      A_.row, A_.col,
                      raw<void>(alpha_),
                      raw<const void>(A_array), A_.desc.lda,
                      raw<const void>(x_array), x_.desc.inc,
                      raw<void>(beta_),
                      raw<void>(y_array), y_.desc.inc,
                      A_array.desc.batch_count);
    });
}

void CpuLAS::mv_strided_batched(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto A_array = A.batch_view();
    auto x_ = x.dense_vector_view();
    auto x_array = x.batch_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();
    auto y_array = y.batch_view();
    auto type = A.basic_data_type();

    LUISA_ASSERT(A_.desc.shape == DenseMatrixShape::GENERAL, "only general matrix is supported.");
    LUISA_ASSERT(A_.storage.size() == x_.storage.size() && A_.storage.size() == y_.storage.size(), "A, x, y must have the same batch size.");
    invoke([=] {
        gemv_batch_strided_ex(type,
                              cblas_layout(),
                              cblas_enum_map(A_.operation),
                              A_.row, A_.col,
                              raw<void>(alpha_),
                              raw<void>(A_), A_.desc.lda, A_array.desc.batch_stride,
                              raw<void>(x_), x_.desc.inc, x_array.desc.batch_stride,
                              raw<void>(beta_),
                              raw<void>(y_), y_.desc.inc, y_array.desc.batch_stride,
                              A_array.desc.batch_count);
    });
}

void CpuLAS::mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto B_ = B.dense_matrix_view();
    auto beta_ = beta.scalar_view();
    auto C_ = C.dense_matrix_view();
    auto type = A.basic_data_type();

    LUISA_ASSERT(C_.desc.shape == DenseMatrixShape::GENERAL, "only general matrix is supported for C.");
    if (A_.desc.shape == DenseMatrixShape::GENERAL && B_.desc.shape == DenseMatrixShape::GENERAL) {
        if (A_.desc.property == DenseMatrixProperty::NONE) {
            invoke([=] {
                gemm_ex(
                    type,
                    cblas_layout(),
                    cblas_enum_map(A_.operation),
                    cblas_enum_map(B_.operation),
                    C_.row, C_.col, A_.col,
                    raw<void>(alpha_),
                    raw<void>(A_), A_.desc.lda,
                    raw<void>(B_), B_.desc.lda,
                    raw<void>(beta_),
                    raw<void>(C_), C_.desc.lda);
            });
        } else {
            invoke([=] {
                symm_ex(
                    type,
                    cblas_layout(),
                    cblas_enum_map(options),
                    cblas_enum_map(A_.desc.fill_mode),
                    C_.row, C_.col,
                    raw<void>(alpha_),
                    raw<void>(A_), A_.desc.lda,
                    raw<void>(B_), B_.desc.lda,
                    raw<void>(beta_),
                    raw<void>(C_), C_.desc.lda);
            });
        }
    }
    //else if (A_.desc.shape == DenseMatrixShape::TRIANGULAR && B_.desc.shape == DenseMatrixShape::GENERAL) {
    //    invoke([=] {
    //        trmm_ex(
    //            type,
    //            cblas_layout(),
    //            cblas_enum_map(options),
    //            cblas_enum_map(A_.desc.fill_mode),
    //            cblas_enum_map(A_.operation),
    //            cblas_enum_map(A_.desc.diag_type),
    //            C_.row, C_.col,
    //            raw<void>(alpha_),
    //            raw<void>(A_), A_.desc.lda,
    //            raw<void>(B_), B_.desc.lda);
    //    });
    //}
    else {
        LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
    }
}

void CpuLAS::sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}

void CpuLAS::mm_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}

void CpuLAS::mm_stride_batched(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}

void CpuLAS::sm_batched(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept {
    LUISA_ERROR_WITH_LOCATION("NOT IMPL YET");
}

CBLAS_TRANSPOSE cblas_enum_map(MatrixOperation op) noexcept {
    switch (op) {
        case luisa::compute::tensor::MatrixOperation::NONE:
            return CBLAS_TRANSPOSE::CblasNoTrans;
        case luisa::compute::tensor::MatrixOperation::TRANS:
            return CBLAS_TRANSPOSE::CblasTrans;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported matrix operation.");
            return CBLAS_TRANSPOSE{};
    }
}

CBLAS_UPLO cblas_enum_map(DenseMatrixFillMode op) noexcept {
    switch (op) {
        case luisa::compute::tensor::DenseMatrixFillMode::LOWER:
            return CBLAS_UPLO::CblasLower;
        case luisa::compute::tensor::DenseMatrixFillMode::UPPER:
            return CBLAS_UPLO::CblasUpper;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported matrix fill mode.");
            return CBLAS_UPLO{};
    }
}

CBLAS_DIAG cblas_enum_map(DenseMatrixDiagType op) noexcept {
    switch (op) {
        case luisa::compute::tensor::DenseMatrixDiagType::NON_UNIT:
            return CBLAS_DIAG::CblasNonUnit;
        case luisa::compute::tensor::DenseMatrixDiagType::UNIT:
            return CBLAS_DIAG::CblasUnit;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported matrix diag type.");
            return CBLAS_DIAG{};
    }
}

CBLAS_SIDE cblas_enum_map(MatrixMulOptions op) noexcept {
    switch (op.side) {
        case luisa::compute::tensor::MatrixASide::LEFT:
            return CBLAS_SIDE::CblasLeft;
        case luisa::compute::tensor::MatrixASide::RIGHT:
            return CBLAS_SIDE::CblasRight;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported matrix side.");
            return CBLAS_SIDE{};
    }
}

}// namespace luisa::compute::cpu::tensor