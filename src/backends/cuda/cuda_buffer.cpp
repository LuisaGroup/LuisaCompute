#include "cuda_error.h"
#include "cuda_buffer.h"

namespace luisa::compute::cuda {

CUDABufferBase::CUDABufferBase(size_t size_bytes) noexcept
    : _handle{}, _size_bytes{size_bytes},
      _host_memory{false}, _external_memory{false} {
    if (auto ret = cuMemAlloc(&_handle, size_bytes);
        ret == CUDA_ERROR_OUT_OF_MEMORY) {
        LUISA_WARNING(
            "CUDA allocation out of device memory. Falling back to host memory.\n"
            "         THIS MAY CAUSE SIGNIFICANT PERFORMANCE DEGRADATION\n"
            "    PLEASE CONSIDER REDUCING THE WORKING SET OF YOUR APPLICATION");
        void *host_ptr = nullptr;
        LUISA_CHECK_CUDA(cuMemAllocHost(&host_ptr, size_bytes));
        _handle = reinterpret_cast<CUdeviceptr>(host_ptr);
        _host_memory = true;
    } else {
        LUISA_CHECK_CUDA(ret);
    }
    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated CUDA buffer: {} bytes @ {}",
        size_bytes, reinterpret_cast<void *>(_handle));
}

CUDABufferBase::~CUDABufferBase() noexcept {
    // we do not own external memory so do not free it
    if (_external_memory) { return; }
    if (_host_memory) {
        LUISA_CHECK_CUDA(cuMemFreeHost(reinterpret_cast<void *>(_handle)));
    } else {
        LUISA_CHECK_CUDA(cuMemFree(_handle));
    }
    auto size = _size_bytes;
    LUISA_VERBOSE_WITH_LOCATION(
        "Freed CUDA buffer: {} bytes @ {}",
        size, reinterpret_cast<void *>(_handle));
}

CUDABufferBase::CUDABufferBase(CUdeviceptr external_ptr, size_t size_bytes) noexcept
    : _handle{external_ptr}, _size_bytes{size_bytes},
      _host_memory{false}, _external_memory{true} {}

CUDABuffer::Binding CUDABuffer::binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(offset + size <= size_bytes(), "CUDABuffer::binding() out of range.");
    return Binding{handle() + offset, size};
}

CUDAIndirectDispatchBuffer::CUDAIndirectDispatchBuffer(size_t capacity) noexcept
    : CUDABufferBase{sizeof(Header) + sizeof(Dispatch) * capacity},
      _capacity{capacity} {}

CUDAIndirectDispatchBuffer::Binding
CUDAIndirectDispatchBuffer::binding(size_t offset, size_t size) const noexcept {
    size = std::min(size, std::numeric_limits<size_t>::max() - offset);// prevent overflow
    return {handle(),
            static_cast<uint>(offset),
            static_cast<uint>(std::min(offset + size, capacity()))};
}

}// namespace luisa::compute::cuda
