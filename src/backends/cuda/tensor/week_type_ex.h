#pragma once
#include <luisa/core/logging.h>
#include <cublas_api.h>
#include <cusparse.h>

namespace luisa::compute::cuda::tensor {

inline cublasStatus_t gemv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasOperation_t trans,
                              int m,
                              int n,
                              const void *alpha, /* host or device pointer */
                              const void *A,
                              int lda,
                              const void *x,
                              int incx,
                              const void *beta, /* host or device pointer */
                              void *y,
                              int incy) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSgemv_v2(handle, trans, m, n,
                                 (const float *)alpha,
                                 (const float *)A, lda,
                                 (const float *)x, incx,
                                 (const float *)beta,
                                 (float *)y, incy);

        } break;
        case CUDA_R_64F: {
            ret = cublasDgemv_v2(handle, trans, m, n,
                                 (const double *)alpha,
                                 (const double *)A, lda,
                                 (const double *)x, incx,
                                 (const double *)beta,
                                 (double *)y, incy);

        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t symv_ex(
    cublasHandle_t handle,
    cudaDataType_t type,
    cublasFillMode_t uplo,
    int n,
    const void *alpha, /* host or device pointer */
    const void *A,
    int lda,
    const void *x,
    int incx,
    const void *beta, /* host or device pointer */
    void *y,
    int incy) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSsymv_v2(handle, uplo, n,
                                 (const float *)alpha,
                                 (const float *)A, lda,
                                 (const float *)x, incx,
                                 (const float *)beta,
                                 (float *)y, incy);

        } break;
        case CUDA_R_64F: {
            ret = cublasDsymv_v2(handle, uplo, n,
                                 (const double *)alpha,
                                 (const double *)A, lda,
                                 (const double *)x, incx,
                                 (const double *)beta,
                                 (double *)y, incy);

        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t trmv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const void *A,
                              int lda,
                              void *x,
                              int incx) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStrmv_v2(
                handle, uplo, trans, diag, n,
                (const float *)A, lda,
                (float *)x, incx);

        } break;
        case CUDA_R_64F: {
            ret = cublasDtrmv_v2(
                handle, uplo, trans, diag, n,
                (const double *)A, lda,
                (double *)x, incx);

        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t gbmv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasOperation_t trans,
                              int m,
                              int n,
                              int kl,
                              int ku,
                              const void *alpha, /* host or device pointer */
                              const void *A,
                              int lda,
                              const void *x,
                              int incx,
                              const void *beta, /* host or device pointer */
                              void *y,
                              int incy) noexcept {
    cublasStatus_t ret;

    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSgbmv_v2(
                handle, trans,
                m, n, kl, ku,
                (const float *)alpha,
                (const float *)A, lda,
                (const float *)x, incx,
                (const float *)beta,
                (float *)y, incy);

        } break;
        case CUDA_R_64F: {
            ret = cublasDgbmv_v2(
                handle, trans,
                m, n, kl, ku,
                (const double *)alpha,
                (const double *)A, lda,
                (const double *)x, incx,
                (const double *)beta,
                (double *)y, incy);

        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
};

inline cublasStatus_t sbmv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              int n,
                              int k,
                              const void *alpha, /* host or device pointer */
                              const void *A,
                              int lda,
                              const void *x,
                              int incx,
                              const void *beta, /* host or device pointer */
                              void *y,
                              int incy) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSsbmv_v2(
                handle, uplo, n, k,
                (const float *)alpha,
                (const float *)A, lda,
                (const float *)x, incx,
                (const float *)beta,
                (float *)y, incy);

        } break;
        case CUDA_R_64F: {
            ret = cublasDsbmv_v2(
                handle, uplo, n, k,
                (const double *)alpha,
                (const double *)A, lda,
                (const double *)x, incx,
                (const double *)beta,
                (double *)y, incy);

        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t tbmv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              int k,
                              const void *A,
                              int lda,
                              void *x,
                              int incx) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStbmv_v2(
                handle, uplo, trans, diag,
                n, k,
                (const float *)A, lda,
                (float *)x, incx);

        } break;
        case CUDA_R_64F: {
            ret = cublasDtbmv_v2(
                handle, uplo, trans, diag,
                n, k,
                (const double *)A, lda,
                (double *)x, incx);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t tpmv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const void *AP,
                              void *x,
                              int incx) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStpmv_v2(
                handle, uplo, trans, diag,
                n,
                (const float *)AP,
                (float *)x, incx);

        } break;
        case CUDA_R_64F: {
            ret = cublasDtpmv_v2(
                handle, uplo, trans, diag,
                n,
                (const double *)AP,
                (double *)x, incx);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t spmv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              int n,
                              const void *alpha, /* host or device pointer */
                              const void *AP,
                              const void *x,
                              int incx,
                              const void *beta, /* host or device pointer */
                              void *y,
                              int incy) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSspmv_v2(
                handle, uplo,
                n,
                (const float *)alpha,
                (const float *)AP,
                (const float *)x, incx,
                (const float *)beta,
                (float *)y, incy);

        } break;
        case CUDA_R_64F: {
            ret = cublasDspmv_v2(
                handle, uplo,
                n,
                (const double *)alpha,
                (const double *)AP,
                (const double *)x, incx,
                (const double *)beta,
                (double *)y, incy);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t gemv_batched_ex(cublasHandle_t handle,
                                      cudaDataType_t type,
                                      cublasOperation_t trans,
                                      int m,
                                      int n,
                                      const void *alpha, /* host or device pointer */
                                      const void *const Aarray[],
                                      int lda,
                                      const void *const xarray[],
                                      int incx,
                                      const void *beta, /* host or device pointer */
                                      void *const yarray[],
                                      int incy,
                                      int batchCount) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSgemvBatched(
                handle, trans,
                m, n,
                (const float *)alpha,
                (const float *const *)Aarray, lda,
                (const float *const *)xarray, incx,
                (const float *)beta,
                (float *const *)yarray, incy, batchCount);
        } break;
        case CUDA_R_64F: {
            ret = cublasDgemvBatched(
                handle, trans,
                m, n,
                (const double *)alpha,
                (const double *const *)Aarray, lda,
                (const double *const *)xarray, incx,
                (const double *)beta,
                (double *const *)yarray, incy, batchCount);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t gemv_strided_batched_ex(cublasHandle_t handle,
                                              cudaDataType_t type,
                                              cublasOperation_t trans,
                                              int m,
                                              int n,
                                              const void *alpha, /* host or device pointer */
                                              const void *A,
                                              int lda,
                                              long long int strideA, /* purposely signed */
                                              const void *x,
                                              int incx,
                                              long long int stridex,
                                              const void *beta, /* host or device pointer */
                                              void *y,
                                              int incy,
                                              long long int stridey,
                                              int batchCount) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSgemvStridedBatched(
                handle, trans,
                m, n,
                (const float *)alpha,
                (const float *)A, lda, strideA,
                (const float *)x, incx, stridex,
                (const float *)beta,
                (float *)y, incy, stridey, batchCount);
        } break;
        case CUDA_R_64F: {
            ret = cublasDgemvStridedBatched(
                handle, trans,
                m, n,
                (const double *)alpha,
                (const double *)A, lda, strideA,
                (const double *)x, incx, stridex,
                (const double *)beta,
                (double *)y, incy, stridey, batchCount);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t trmm_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasSideMode_t side,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int m,
                              int n,
                              const void *alpha, /* host or device pointer */
                              const void *A,
                              int lda,
                              const void *B,
                              int ldb,
                              void *C,
                              int ldc) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStrmm_v2(
                handle, side, uplo, trans, diag,
                m, n,
                (const float *)alpha,
                (const float *)A, lda,
                (const float *)B, ldb,
                (float *)C, ldc);
        } break;
        case CUDA_R_64F: {
            ret = cublasDtrmm_v2(
                handle, side, uplo, trans, diag,
                m, n,
                (const double *)alpha,
                (const double *)A, lda,
                (const double *)B, ldb,
                (double *)C, ldc);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t symm_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasSideMode_t side,
                              cublasFillMode_t uplo,
                              int m,
                              int n,
                              const void *alpha, /* host or device pointer */
                              const void *A,
                              int lda,
                              const void *B,
                              int ldb,
                              const void *beta, /* host or device pointer */
                              void *C,
                              int ldc) noexcept {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasSsymm_v2(
                handle, side, uplo,
                m, n,
                (const float *)alpha,
                (const float *)A, lda,
                (const float *)B, ldb,
                (const float *)beta,
                (float *)C, ldc);
        } break;
        case CUDA_R_64F: {
            ret = cublasDsymm_v2(
                handle, side, uplo,
                m, n,
                (const double *)alpha,
                (const double *)A, lda,
                (const double *)B, ldb,
                (const double *)beta,
                (double *)C, ldc);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t trsv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              const void *A,
                              int lda,
                              void *x,
                              int incx) {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStrsv_v2(handle, uplo, trans, diag, n,
                                 (const float *)A, lda,
                                 (float *)x, incx);
        } break;
        case CUDA_R_64F: {
            ret = cublasDtrsv_v2(handle, uplo, trans, diag, n,
                                 (const double *)A, lda,
                                 (double *)x, incx);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t tbsv_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int n,
                              int k,
                              const void *A,
                              int lda,
                              void *x,
                              int incx) {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStbsv_v2(handle, uplo, trans, diag,
                                 n, k,
                                 (const float *)A, lda,
                                 (float *)x, incx);
        } break;
        case CUDA_R_64F: {
            ret = cublasDtbsv_v2(handle, uplo, trans, diag,
                                 n, k,
                                 (const double *)A, lda,
                                 (double *)x, incx);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t trsm_ex(cublasHandle_t handle,
                              cudaDataType_t type,
                              cublasSideMode_t side,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              cublasDiagType_t diag,
                              int m,
                              int n,
                              const void *alpha, /* host or device pointer */
                              const void *A,
                              int lda,
                              void *B,
                              int ldb) {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStrsm_v2(handle, side, uplo, trans, diag,
                                 m, n,
                                 (const float *)alpha,
                                 (const float *)A, lda,
                                 (float *)B, ldb);
        } break;
        case CUDA_R_64F: {
            ret = cublasDtrsm_v2(handle, side, uplo, trans, diag,
                                 m, n,
                                 (const double *)alpha,
                                 (const double *)A, lda,
                                 (double *)B, ldb);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}

inline cublasStatus_t trsm_batched_ex(cublasHandle_t handle,
                                      cudaDataType_t type,
                                      cublasSideMode_t side,
                                      cublasFillMode_t uplo,
                                      cublasOperation_t trans,
                                      cublasDiagType_t diag,
                                      int m,
                                      int n,
                                      const void *alpha, /*Host or Device Pointer*/
                                      const void *const A[],
                                      int lda,
                                      void *const B[],
                                      int ldb,
                                      int batchCount) {
    cublasStatus_t ret;
    switch (type) {
        case CUDA_R_32F: {
            ret = cublasStrsmBatched(handle, side, uplo, trans, diag,
                                     m, n,
                                     (const float *)alpha,
                                     (const float *const *)A, lda,
                                     (float *const *)B, ldb,
                                     batchCount);
        } break;
        case CUDA_R_64F: {
            ret = cublasDtrsmBatched(handle, side, uplo, trans, diag,
                                     m, n,
                                     (const double *)alpha,
                                     (const double *const *)A, lda,
                                     (double *const *)B, ldb,
                                     batchCount);
        } break;
        //case CUDA_C_64F: {

        //} break;
        //case CUDA_C_32F: {

        //} break;
        default:
            LUISA_ERROR_WITH_LOCATION("unspported data type.");
    }
    return ret;
}
}// namespace luisa::compute::cuda::tensor