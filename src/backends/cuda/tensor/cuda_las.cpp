#include "cuda_las.h"
#include "../utils/cublas_check.h"
#include "../utils/cusparse_check.h"
#include "../cuda_device.h"
#include "../cuda_buffer.h"
#include <luisa/tensor/tensor.h>

namespace luisa::compute::cuda::tensor {
using ScalarView = luisa::compute::tensor::ScalarView;
using DenseVectorView = luisa::compute::tensor::DenseVectorView;
using DenseMatrixView = luisa::compute::tensor::DenseMatrixView;
template<typename T>
auto raw(uint64_t buffer_handle) {
    auto cuda_buffer = reinterpret_cast<CUDABuffer *>(buffer_handle);
    return reinterpret_cast<T *>(cuda_buffer->handle());
}

template<typename T>
auto raw(const ScalarView &s) { return raw<T>(s.buffer_handle) + s.buffer_offset; }

template<typename T>
auto raw(const DenseVectorView &v) { return raw<T>(v.buffer_handle) + v.buffer_offset; }

template<typename T>
auto raw(const DenseMatrixView &m) { return raw<T>(m.buffer_handle) + m.buffer_offset; }

CudaLAS::CudaLAS(CUDAStream *stream) noexcept : _stream{stream} {
    LUISA_CHECK_CUBLAS(cublasCreate(&_cublas_handle));
    LUISA_CHECK_CUBLAS(cublasSetStream(_cublas_handle, _stream->handle()));
    LUISA_CHECK_CUBLAS(cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    LUISA_CHECK_CUBLAS(cublasSetAtomicsMode(_cublas_handle, CUBLAS_ATOMICS_ALLOWED));

    LUISA_CHECK_CUSPARSE(cusparseCreate(&_cusparse_handle));
    LUISA_CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, _stream->handle()));
    LUISA_CHECK_CUSPARSE(cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE));
}

CudaLAS::~CudaLAS() noexcept {
    LUISA_CHECK_CUBLAS(cublasDestroy(_cublas_handle));
    LUISA_CHECK_CUSPARSE(cusparseDestroy(_cusparse_handle));
}
void CudaLAS::Iamax(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasIsamax_v2(_cublas_handle,
                                       x.n,
                                       raw<float>(x), x.inc,
                                       raw<int>(r)));
}
void CudaLAS::Iamin(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasIsamin_v2(_cublas_handle,
                                       x.n,
                                       raw<float>(x), x.inc,
                                       raw<int>(r)));
}
void CudaLAS::dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept {
    auto x = vec_x.dense_vector_view();
    auto y = vec_y.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasSdot_v2(_cublas_handle,
                                     x.n,
                                     raw<float>(x), x.inc,
                                     raw<float>(y), y.inc,
                                     raw<float>(r)));
}
void CudaLAS::nrm2(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasSnrm2_v2(_cublas_handle,
                                      x.n,
                                      raw<float>(x), x.inc,
                                      raw<float>(r)));
}

void CudaLAS::mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
    auto alpha_ = alpha.scalar_view();
    auto A_ = A.dense_matrix_view();
    auto x_ = x.dense_vector_view();
    auto beta_ = beta.scalar_view();
    auto y_ = y.dense_vector_view();

    switch (A_.shape) {
        case luisa::compute::tensor::DenseMatrixShape::GENERAL: {
            cublasSgemv_v2(_cublas_handle,
                           enum_map(A_.operation),
                           A_.row, A_.column,
                           raw<float>(alpha_),
                           raw<float>(A_), A_.lda,
                           raw<float>(x_), x_.inc,
                           raw<float>(beta_),
                           raw<float>(y_), y_.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR: {
            cublasStrmv_v2(_cublas_handle,
                           enum_map(A_.fill_mode),
                           enum_map(A_.operation),
                           enum_map(A_.diag_type),
                           A_.row,
                           raw<float>(A_), A_.lda,
                           raw<float>(x_), x_.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::BAND: {
            cublasSgbmv_v2(_cublas_handle,
                           enum_map(A_.operation),
                           A_.row, A_.column,
                           A_.kl, A_.ku,
                           raw<float>(alpha_),
                           raw<float>(A_), A_.lda,
                           raw<float>(x_), x_.inc,
                           raw<float>(beta_),
                           raw<float>(y_), y_.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::TRIANGULAR_BAND: {
            cublasStbmv_v2(_cublas_handle,
                           enum_map(A_.fill_mode),
                           enum_map(A_.operation),
                           enum_map(A_.diag_type),
                           A_.row, A_.kl,
                           raw<float>(A_), A_.lda,
                           raw<float>(x_), x_.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED_TRIANGULAR: {
            cublasStpmv_v2(_cublas_handle,
                           enum_map(A_.fill_mode),
                           enum_map(A_.operation),
                           enum_map(A_.diag_type),
                           A_.row,
                           raw<float>(A_),
                           raw<float>(x_), x_.inc);
        } break;
        case luisa::compute::tensor::DenseMatrixShape::PACKED: {
            cublasSspmv_v2(_cublas_handle,
                           enum_map(A_.fill_mode),
                           A_.row,
                           raw<float>(alpha_),
                           raw<float>(A_),
                           raw<float>(x_), x_.inc,
                           raw<float>(beta_),
                           raw<float>(y_), y_.inc);
        } break;
        default:
            LUISA_ERROR_WITH_LOCATION("Unsupported dense matrix shape");
            break;
    }
}

void CudaLAS::sv(DTensor &x, const DTensor &A) noexcept {
}

void CudaLAS::mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
}
void CudaLAS::sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept {
}
}// namespace luisa::compute::cuda::tensor

namespace luisa::compute::cuda::tensor {
cublasOperation_t CudaLAS::enum_map(luisa::compute::tensor::MatrixOperation op) noexcept {
    cublasOperation_t ret;
    switch (op) {
        case luisa::compute::tensor::MatrixOperation::NONE:
            ret = CUBLAS_OP_N;
        case luisa::compute::tensor::MatrixOperation::TRANS:
            ret = CUBLAS_OP_T;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix operation mapping.");
            break;
    }
    return ret;
}
cublasFillMode_t CudaLAS::enum_map(luisa::compute::tensor::DenseMatrixFillMode op) noexcept {
    cublasFillMode_t ret;
    switch (op) {
        case luisa::compute::tensor::DenseMatrixFillMode::UPPER:
            ret = CUBLAS_FILL_MODE_UPPER;
        case luisa::compute::tensor::DenseMatrixFillMode::LOWER:
            ret = CUBLAS_FILL_MODE_LOWER;
        default:
            LUISA_ERROR_WITH_LOCATION("error matrix fill mode mapping.");
            break;
    }
    return ret;
}
cublasDiagType_t CudaLAS::enum_map(luisa::compute::tensor::DenseMatrixDiagType op) noexcept {
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
}// namespace luisa::compute::cuda::tensor