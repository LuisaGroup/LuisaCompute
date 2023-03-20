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

CUDABuffer::Binding CUDABuffer::binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(offset + size <= _size, "CUDABuffer::binding() out of range.");
    return Binding{_handle + offset, size};
}

}// namespace luisa::compute::cuda
