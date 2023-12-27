#include "cuda_error.h"
#include "cuda_buffer.h"

namespace luisa::compute::cuda {

CUDABufferBase::CUDABufferBase(size_t size_bytes,
                               Location loc,
                               int host_alloc_flags) noexcept
    : _device_address{}, _size_bytes{size_bytes}, _external_memory{false} {
    switch (loc) {
        case Location::FORCE_HOST: {
            LUISA_CHECK_CUDA(cuMemHostAlloc(&_host_address, size_bytes, host_alloc_flags));
            LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&_device_address, _host_address, 0));
            break;
        }
        case Location::PREFER_DEVICE: {
            if (auto ret = cuMemAlloc(&_device_address, size_bytes);
                ret == CUDA_ERROR_OUT_OF_MEMORY) {
                LUISA_WARNING(
                    "CUDA allocation out of device memory. Falling back to host memory.\n"
                    "         THIS MAY CAUSE SIGNIFICANT PERFORMANCE DEGRADATION\n"
                    "    PLEASE CONSIDER REDUCING THE WORKING SET OF YOUR APPLICATION");
                LUISA_CHECK_CUDA(cuMemHostAlloc(&_host_address, size_bytes,
                                                CU_MEMHOSTALLOC_DEVICEMAP |
                                                    CU_MEMHOSTALLOC_WRITECOMBINED));
                LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&_device_address, _host_address, 0));
            } else {
                LUISA_CHECK_CUDA(ret);
            }
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Invalid CUDABufferBase::Location.");
    }
    LUISA_VERBOSE_WITH_LOCATION(
        "Allocated CUDA buffer: {} bytes @ {}",
        size_bytes, reinterpret_cast<void *>(_device_address));
}

CUDABufferBase::~CUDABufferBase() noexcept {
    // we do not own external memory so do not free it
    if (_external_memory) { return; }
    if (_host_address != nullptr) {
        LUISA_CHECK_CUDA(cuMemFreeHost(_host_address));
    } else {
        LUISA_CHECK_CUDA(cuMemFree(_device_address));
    }
    auto size = _size_bytes;
    LUISA_VERBOSE_WITH_LOCATION(
        "Freed CUDA buffer: {} bytes @ {}",
        size, reinterpret_cast<void *>(_device_address));
}

CUDABufferBase::CUDABufferBase(CUdeviceptr external_ptr, size_t size_bytes) noexcept
    : _device_address{external_ptr}, _size_bytes{size_bytes}, _external_memory{true} {}

CUDABuffer::Binding CUDABuffer::binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(offset + size <= size_bytes(), "CUDABuffer::binding() out of range.");
    return Binding{device_address() + offset, size};
}

CUDAIndirectDispatchBuffer::CUDAIndirectDispatchBuffer(size_t capacity) noexcept
    : CUDABufferBase{sizeof(Header) + sizeof(Dispatch) * capacity},
      _capacity{capacity} {}

CUDAIndirectDispatchBuffer::Binding
CUDAIndirectDispatchBuffer::binding(size_t offset, size_t size) const noexcept {
    size = std::min(size, std::numeric_limits<size_t>::max() - offset);// prevent overflow
    return {device_address(),
            static_cast<uint>(offset),
            static_cast<uint>(std::min(offset + size, capacity()))};
}

}// namespace luisa::compute::cuda
