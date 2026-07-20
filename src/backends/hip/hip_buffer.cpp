//
// Created by mike on 12/25/25.
//

#include <luisa/core/stl/memory.h>

#include "hip_check.h"
#include "hip_buffer.h"

namespace luisa::compute::hip {

inline HIPBuffer::HIPBuffer() noexcept
    : _device_ptr{}, _host_ptr{}, _size_bytes{}, _indirect_capacity{},
      _ownership{Ownership::EXTERNAL_DEVICE} {}

HIPBuffer::~HIPBuffer() noexcept {
    switch (_ownership) {
        case Ownership::DEVICE:
            LUISA_CHECK_HIP(hipFree(_device_ptr));
            break;
        case Ownership::HOST_ALLOCATION:
            LUISA_CHECK_HIP(hipHostFree(_host_ptr));
            break;
        case Ownership::REGISTERED_HOST:
            LUISA_CHECK_HIP(hipHostUnregister(_host_ptr));
            break;
        case Ownership::EXTERNAL_DEVICE:
        case Ownership::EXTERNAL_HOST:
            break;
    }
}

HIPBuffer *HIPBuffer::create_device_buffer(size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    LUISA_CHECK_HIP(hipMalloc(&buffer->_device_ptr, size_bytes));
    buffer->_size_bytes = size_bytes;
    buffer->_ownership = Ownership::DEVICE;
    return buffer;
}

HIPBuffer *HIPBuffer::create_indirect_buffer(size_t capacity) noexcept {
    LUISA_ASSERT(capacity > 0u, "Indirect dispatch buffer capacity must be positive.");
    auto buffer = create_device_buffer(
        sizeof(IndirectHeader) + sizeof(IndirectDispatch) * capacity);
    buffer->_indirect_capacity = capacity;
    return buffer;
}

HIPBuffer *HIPBuffer::create_host_buffer(size_t size_bytes, bool write_combined) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    auto flags = static_cast<unsigned int>(hipHostMallocMapped);
    if (write_combined) { flags |= hipHostMallocWriteCombined; }
    LUISA_CHECK_HIP(hipHostMalloc(&buffer->_host_ptr, size_bytes, flags));
    LUISA_CHECK_HIP(hipHostGetDevicePointer(
        reinterpret_cast<void **>(&buffer->_device_ptr),
        buffer->_host_ptr, 0u));
    buffer->_size_bytes = size_bytes;
    buffer->_ownership = Ownership::HOST_ALLOCATION;
    return buffer;
}

HIPBuffer *HIPBuffer::register_host_buffer(void *host_ptr, size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    LUISA_CHECK_HIP(hipHostRegister(host_ptr, size_bytes, hipHostRegisterMapped));
    LUISA_CHECK_HIP(hipHostGetDevicePointer(
        reinterpret_cast<void **>(&buffer->_device_ptr),
        host_ptr, 0u));
    buffer->_host_ptr = host_ptr;
    buffer->_size_bytes = size_bytes;
    buffer->_ownership = Ownership::REGISTERED_HOST;
    return buffer;
}

HIPBuffer *HIPBuffer::import_external_device_buffer(hipDeviceptr_t external_ptr, size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    buffer->_device_ptr = external_ptr;
    buffer->_size_bytes = size_bytes;
    buffer->_ownership = Ownership::EXTERNAL_DEVICE;
    return buffer;
}

HIPBuffer *HIPBuffer::import_external_host_buffer(void *external_ptr, size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    LUISA_CHECK_HIP(hipHostGetDevicePointer(
        reinterpret_cast<void **>(&buffer->_device_ptr),
        external_ptr, 0u));
    buffer->_host_ptr = external_ptr;
    buffer->_size_bytes = size_bytes;
    buffer->_ownership = Ownership::EXTERNAL_HOST;
    return buffer;
}

void HIPBuffer::destroy(HIPBuffer *buffer) noexcept {
    luisa::delete_with_allocator(buffer);
}

HIPBuffer::Binding HIPBuffer::binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(offset + size <= size_bytes(), "HIPBuffer::binding() out of range.");
    return Binding{static_cast<std::byte *>(_device_ptr) + offset, size};
}

HIPBuffer::IndirectBinding HIPBuffer::indirect_binding(size_t offset, size_t size) const noexcept {
    LUISA_ASSERT(is_indirect(), "HIPBuffer is not an indirect dispatch buffer.");
    auto end = std::min(
        _indirect_capacity,
        offset + std::min(size, std::numeric_limits<size_t>::max() - offset));
    auto clamped_offset = std::min(offset, end);
    LUISA_ASSERT(clamped_offset <= std::numeric_limits<uint32_t>::max() &&
                     end <= std::numeric_limits<uint32_t>::max(),
                 "HIP indirect dispatch range exceeds the 32-bit ABI.");
    auto packed = static_cast<uint64_t>(clamped_offset) |
                  static_cast<uint64_t>(end) << 32u;
    return {_device_ptr, packed};
}

}// namespace luisa::compute::hip
