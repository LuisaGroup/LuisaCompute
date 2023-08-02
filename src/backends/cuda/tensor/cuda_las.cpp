#include "cuda_las.h"
#include <luisa/tensor/tensor.h>
#include "../utils/cublas_check.h"
#include "../utils/cusparse_check.h"
#include "cuda_tensor_res.h"


using namespace luisa::compute::cuda::tensor;
// Ctor
CudaLAS::CudaLAS(CUDAStream *stream) noexcept : _stream{stream} {
    LUISA_CHECK_CUBLAS(cublasCreate(&_cublas_handle));
    LUISA_CHECK_CUBLAS(cublasSetStream(_cublas_handle, _stream->handle()));
    LUISA_CHECK_CUBLAS(cublasSetPointerMode(_cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    LUISA_CHECK_CUBLAS(cublasSetAtomicsMode(_cublas_handle, CUBLAS_ATOMICS_ALLOWED));

    LUISA_CHECK_CUSPARSE(cusparseCreate(&_cusparse_handle));
    LUISA_CHECK_CUSPARSE(cusparseSetStream(_cusparse_handle, _stream->handle()));
    LUISA_CHECK_CUSPARSE(cusparseSetPointerMode(_cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE));
}

// Dtor
CudaLAS::~CudaLAS() noexcept {
    LUISA_CHECK_CUBLAS(cublasDestroy(_cublas_handle));
    LUISA_CHECK_CUSPARSE(cusparseDestroy(_cusparse_handle));
}





CudaLAS::S<CudaLAS::BackendTensorRes> CudaLAS::alloc_backend_tensor_res(const DTensor &tensor) noexcept {
    if (tensor.is_sparse()) {
        if (tensor.is_vector())// sparse vector
            return luisa::make_unique<CusparseSpVecDescRes>(tensor);
        else if (tensor.is_matrix())// sparse matrix
            return luisa::make_unique<CusparseSpMatDescRes>(tensor);
    } else if (tensor.is_dense()) {
        if (tensor.is_vector())
            return luisa::make_unique<CusparseDnVecDescRes>(tensor);
        else if (tensor.is_matrix())
            return luisa::make_unique<CusparseDnMatDescRes>(tensor);
    }
    return nullptr;
}