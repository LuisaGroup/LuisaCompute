//
// Created by Mike on 3/14/2023.
//

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_buffer.h>

namespace luisa::compute::cuda {

CUDABuffer::CUDABuffer(size_t size_bytes) noexcept
    : _handle{}, _size{size_bytes} {
    LUISA_CHECK_CUDA(cuMemAlloc(&_handle, size_bytes));
}

CUDABuffer::~CUDABuffer() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_handle));
}

}// namespace luisa::compute::cuda
