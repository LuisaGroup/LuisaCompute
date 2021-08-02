//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_stream.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream() noexcept
    : _handle{nullptr},
      _upload_pool{32_mb, true} {
    LUISA_CHECK_CUDA(cuStreamCreate(&_handle, CU_STREAM_DEFAULT));
}

CUDAStream::~CUDAStream() noexcept {
    LUISA_CHECK_CUDA(cuStreamDestroy(_handle));
}

}// namespace luisa::compute::cuda
