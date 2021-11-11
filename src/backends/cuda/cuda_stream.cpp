//
// Created by Mike on 8/1/2021.
//

#include <backends/cuda/cuda_stream.h>

namespace luisa::compute::cuda {

CUDAStream::CUDAStream() noexcept
    : _handle{nullptr},
      _upload_pool{32_mb, true} {
    int lo, hi;
    LUISA_CHECK_CUDA(cuCtxGetStreamPriorityRange(&lo, &hi));
    LUISA_INFO("CUDA stream priority range: [{}, {}].", lo, hi);
    LUISA_CHECK_CUDA(cuStreamCreateWithPriority(&_handle, CU_STREAM_NON_BLOCKING, hi));
}

CUDAStream::~CUDAStream() noexcept {
    LUISA_CHECK_CUDA(cuStreamDestroy(_handle));
}

}// namespace luisa::compute::cuda
