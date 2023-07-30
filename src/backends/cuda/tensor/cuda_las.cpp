#include "cuda_las.h"
#include "../utils/cublas_check.h"
#include "../utils/cusparse_check.h"
#include "../cuda_device.h"
#include "../cuda_buffer.h"
#include <luisa/tensor/tensor.h>

namespace luisa::compute::cuda::tensor {
template<typename T>
auto raw(uint64_t buffer_handle) {
    auto cuda_buffer = reinterpret_cast<CUDABuffer *>(buffer_handle);
    return reinterpret_cast<T *>(cuda_buffer->handle());
}

CudaLAS::CudaLAS(CUDAStream *stream) noexcept : _stream{stream} {
    LUISA_CHECK_CUBLAS(cublasCreate(&_cublas_handle));
    LUISA_CHECK_CUBLAS(cublasSetStream(_cublas_handle, _stream->handle()));
    LUISA_CHECK_CUBLAS(cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    LUISA_CHECK_CUSPARSE(cusparseCreate(&_cusparse_handle));
    LUISA_CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, _stream->handle()));
    LUISA_CHECK_CUSPARSE(cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE));
}

CudaLAS::~CudaLAS() noexcept {
    LUISA_CHECK_CUBLAS(cublasDestroy(_cublas_handle));
    LUISA_CHECK_CUSPARSE(cusparseDestroy(_cusparse_handle));
}
void CudaLAS::Iamax(DTensor &result, const DTensor &vec_x) noexcept {
}
void CudaLAS::Iamin(DTensor &result, const DTensor &vec_x) noexcept {
}
void CudaLAS::dot(DTensor &result, const DTensor &vec_x, const DTensor &vec_y) noexcept {
}
void CudaLAS::nrm2(DTensor &result, const DTensor &vec_x) noexcept {
    auto x = vec_x.dense_vector_view();
    auto r = result.scalar_view();
    LUISA_CHECK_CUBLAS(cublasSnrm2_v2(_cublas_handle, x.size, raw<float>(x.buffer_handle), x.inc, raw<float>(r)));
}
void CudaLAS::mv(DTensor &y, const DTensor &alpha, const DTensor &A, const DTensor &x, const DTensor &beta) noexcept {
}
void CudaLAS::sv(DTensor &x, const DTensor &A) noexcept {
}
void CudaLAS::mm(DTensor &C, const DTensor &alpha, const DTensor &A, const DTensor &B, const DTensor &beta, MatrixMulOptions options) noexcept {
}
void CudaLAS::sm(DTensor &X, const DTensor &alpha, const DTensor &A, MatrixMulOptions options) noexcept {
}
}// namespace luisa::compute::cuda::tensor