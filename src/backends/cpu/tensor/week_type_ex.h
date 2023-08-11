#pragma once
#include <luisa/core/logging.h>
#include <mkl.h>
#include <luisa/tensor/tensor.h>

namespace luisa::compute::cpu::tensor {
template<typename T>
T &ref(void *r) { return *reinterpret_cast <T*>(r); }
template<typename T>
const T &ref(const void *r) { return *reinterpret_cast<const T *>(r); }
template<typename T>
auto cast(void *r) { return reinterpret_cast<T *>(r); }
template<typename T>
auto cast(const void *r) { return reinterpret_cast<const T *>(r); }

inline void iamax_ex(luisa::compute::tensor::TensorBasicDataType type,
                     int *R,
                     const MKL_INT N, const void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            *R = cblas_isamax(N, cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            *R = cblas_idamax(N, cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}
inline void iamin_ex(luisa::compute::tensor::TensorBasicDataType type,
                     int *R,
                     const MKL_INT N, const void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            *R = cblas_isamin(N, cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            *R = cblas_idamin(N, cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void dot_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    void *R,
    const MKL_INT N, const void *X, const MKL_INT incX,
    const void *Y, const MKL_INT incY) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            ref<float>(R) = cblas_sdot(
                N,
                cast<float>(X), incX,
                cast<float>(Y), incY);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            ref<double>(R) = cblas_ddot(
                N,
                cast<double>(X), incX,
                cast<double>(Y), incY);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void nrm2_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    void *R,
    const MKL_INT N, const void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            ref<float>(R) = cblas_snrm2(N, cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            ref<double>(R) = cblas_dnrm2(N, cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void gemv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
    const void *alpha, const void *A, const MKL_INT lda,
    const void *X, const MKL_INT incX, const void *beta,
    void *Y, const MKL_INT incY) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_sgemv(Layout, TransA,
                        M, N, ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(X), incX,
                        ref<float>(beta),
                        cast<float>(Y), incY);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dgemv(Layout, TransA,
                        M, N, ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(X), incX,
                        ref<double>(beta),
                        cast<double>(Y), incY);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void gbmv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
    const MKL_INT KL, const MKL_INT KU, const void *alpha,
    const void *A, const MKL_INT lda, const void *X,
    const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_sgbmv(Layout, TransA,
                        M, N, KL, KU,
                        ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(X), incX,
                        ref<float>(beta),
                        cast<float>(Y), incY);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dgbmv(Layout, TransA,
                        M, N, KL, KU,
                        ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(X), incX,
                        ref<double>(beta),
                        cast<double>(Y), incY);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void symv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const MKL_INT N, const void *alpha, const void *A,
    const MKL_INT lda, const void *X, const MKL_INT incX,
    const void *beta, void *Y, const MKL_INT incY) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_ssymv(Layout, Uplo, N,
                        ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(X), incX,
                        ref<float>(beta),
                        cast<float>(Y), incY);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dsymv(Layout, Uplo, N,
                        ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(X), incX,
                        ref<double>(beta),
                        cast<double>(Y), incY);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void sbmv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const MKL_INT N, const MKL_INT K, const void *alpha,
    const void *A, const MKL_INT lda, const void *X,
    const MKL_INT incX, const void *beta, void *Y, const MKL_INT incY) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_ssbmv(Layout, Uplo, N, K,
                        ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(X), incX,
                        ref<float>(beta),
                        cast<float>(Y), incY);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dsbmv(Layout, Uplo, N, K,
                        ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(X), incX,
                        ref<double>(beta),
                        cast<double>(Y), incY);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void spmv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const MKL_INT N, const void *alpha, const void *Ap,
    const void *X, const MKL_INT incX, const void *beta,
    void *Y, const MKL_INT incY) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_sspmv(Layout, Uplo, N,
                        ref<float>(alpha),
                        cast<float>(Ap),
                        cast<float>(X), incX,
                        ref<float>(beta),
                        cast<float>(Y), incY);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dspmv(Layout, Uplo, N,
                        ref<double>(alpha),
                        cast<double>(Ap),
                        cast<double>(X), incX,
                        ref<double>(beta),
                        cast<double>(Y), incY);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void trmv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
    const MKL_INT N, const void *A, const MKL_INT lda,
    void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_strmv(Layout, Uplo, TransA, Diag, N,
                        cast<float>(A), lda,
                        cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtrmv(Layout, Uplo, TransA, Diag, N,
                        cast<double>(A), lda,
                        cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void tbmv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
    const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
    void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_stbmv(Layout, Uplo, TransA, Diag,
                        N, K,
                        cast<float>(A), lda,
                        cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtbmv(Layout, Uplo, TransA, Diag,
                        N, K,
                        cast<double>(A), lda,
                        cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void tpmv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
    const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_stpmv(Layout, Uplo, TransA, Diag,
                        N,
                        cast<float>(Ap),
                        cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtpmv(Layout, Uplo, TransA, Diag,
                        N,
                        cast<double>(Ap),
                        cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void trsv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
    const MKL_INT N, const void *A, const MKL_INT lda, void *X,
    const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_strsv(Layout, Uplo, TransA, Diag,
                        N,
                        cast<float>(A), lda,
                        cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtrsv(Layout, Uplo, TransA, Diag,
                        N,
                        cast<double>(A), lda,
                        cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void tbsv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
    const MKL_INT N, const MKL_INT K, const void *A, const MKL_INT lda,
    void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_stbsv(Layout, Uplo, TransA, Diag,
                        N, K,
                        cast<float>(A), lda,
                        cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtbsv(Layout, Uplo, TransA, Diag,
                        N, K,
                        cast<double>(A), lda,
                        cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void tpsv_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
    const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
    const MKL_INT N, const void *Ap, void *X, const MKL_INT incX) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_stpsv(Layout, Uplo, TransA, Diag,
                        N,
                        cast<float>(Ap),
                        cast<float>(X), incX);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtpsv(Layout, Uplo, TransA, Diag,
                        N,
                        cast<double>(Ap),
                        cast<double>(X), incX);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void gemv_batch_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
    const void *alpha, const void **A, const MKL_INT lda,
    const void **X, const MKL_INT incX, const void *beta,
    void **Y, const MKL_INT incY,
    const MKL_INT group_size) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_sgemv_batch(Layout, &TransA,
                              &M, &N, cast<float>(alpha),
                              cast<const float *>(A), &lda,
                              cast<const float *>(X), &incX,
                              cast<const float>(beta),
                              cast<float *>(Y), &incY,
                              1, &group_size);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dgemv_batch(Layout, &TransA,
                              &M, &N, cast<double>(alpha),
                              cast<const double *>(A), &lda,
                              cast<const double *>(X), &incX,
                              cast<const double>(beta),
                              cast<double *>(Y), &incY,
                              1, &group_size);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void gemv_batch_strided_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE TransA, const MKL_INT M, const MKL_INT N,
    const void *alpha, const void *A, const MKL_INT lda, const MKL_INT stridea,
    const void *X, const MKL_INT incX, const MKL_INT stridex, const void *beta,
    void *Y, const MKL_INT incY, const MKL_INT stridey,
    const MKL_INT batch_size) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_sgemv_batch_strided(Layout, TransA,
                                      M, N, ref<float>(alpha),
                                      cast<float>(A), lda, stridea,
                                      cast<float>(X), incX, stridex,
                                      ref<float>(beta),
                                      cast<float>(Y), incY, stridey,
                                      batch_size);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dgemv_batch_strided(Layout, TransA,
                                      M, N, ref<double>(alpha),
                                      cast<double>(A), lda, stridea,
                                      cast<double>(X), incX, stridex,
                                      ref<double>(beta),
                                      cast<double>(Y), incY, stridey,
                                      batch_size);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void gemm_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
    const MKL_INT K, const void *alpha, const void *A,
    const MKL_INT lda, const void *B, const MKL_INT ldb,
    const void *beta, void *C, const MKL_INT ldc) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_sgemm(Layout, TransA, TransB,
                        M, N, K,
                        ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(B), ldb,
                        ref<float>(beta),
                        cast<float>(C), ldc);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dgemm(Layout, TransA, TransB,
                        M, N, K,
                        ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(B), ldb,
                        ref<double>(beta),
                        cast<double>(C), ldc);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void symm_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
    const CBLAS_UPLO Uplo, const MKL_INT M, const MKL_INT N,
    const void *alpha, const void *A, const MKL_INT lda,
    const void *B, const MKL_INT ldb, const void *beta,
    void *C, const MKL_INT ldc) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_ssymm(Layout, Side, Uplo,
                        M, N,
                        ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(B), ldb,
                        ref<float>(beta),
                        cast<float>(C), ldc);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dsymm(Layout, Side, Uplo,
                        M, N,
                        ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(B), ldb,
                        ref<double>(beta),
                        cast<double>(C), ldc);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

inline void trmm_ex(
    luisa::compute::tensor::TensorBasicDataType type,
    const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
    const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
    const CBLAS_DIAG Diag, const MKL_INT M, const MKL_INT N,
    const void *alpha, const void *A, const MKL_INT lda,
    void *B, const MKL_INT ldb) noexcept {
    switch (type) {
        case luisa::compute::tensor::TensorBasicDataType::FLOAT32: {
            cblas_strmm(Layout, Side, Uplo, TransA, Diag,
                        M, N,
                        ref<float>(alpha),
                        cast<float>(A), lda,
                        cast<float>(B), ldb);
        } break;
        case luisa::compute::tensor::TensorBasicDataType::FLOAT64: {
            cblas_dtrmm(Layout, Side, Uplo, TransA, Diag,
                        M, N,
                        ref<double>(alpha),
                        cast<double>(A), lda,
                        cast<double>(B), ldb);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
}

}// namespace luisa::compute::cpu::tensor