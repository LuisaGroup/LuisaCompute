#include "cuda_error.h"
#include "cuda_buffer.h"

namespace luisa::compute::cuda {

CUDABufferBase::CUDABufferBase(size_t size_bytes) noexcept
    : _handle{}, _size_bytes{size_bytes} {
    LUISA_CHECK_CUDA(cuMemAlloc(&_handle, size_bytes));
}

CUDABufferBase::~CUDABufferBase() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_handle));
}

CUDABuffer::Binding CUDABuffer::binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(offset + size <= size_bytes(), "CUDABuffer::binding() out of range.");
    return Binding{handle() + offset, size};
}

CUDAIndirectDispatchBuffer::CUDAIndirectDispatchBuffer(size_t capacity) noexcept
    : CUDABufferBase{sizeof(Header) + sizeof(Dispatch) * capacity},
      _capacity{capacity} {}

CUDAIndirectDispatchBuffer::Binding
CUDAIndirectDispatchBuffer::binding() const noexcept {
    return {handle(), capacity()};
}

}// namespace luisa::compute::cuda

