//
// Created by Mike on 12/27/2023.
//

#include "../cuda_device.h"
#include "../cuda_buffer.h"
#include "cuda_pinned_memory.h"

namespace luisa::compute::cuda {

class CUDAPinnedMemoryBuffer : public CUDABuffer {
private:
    void *_external_host_memory{};

private:
    [[nodiscard]] static CUdeviceptr _pin_host_memory(void *host_ptr, size_t size_bytes,
                                                      bool write_combined [[maybe_unused]]) noexcept {
        LUISA_CHECK_CUDA(cuMemHostRegister(host_ptr, size_bytes, CU_MEMHOSTREGISTER_DEVICEMAP));
        CUdeviceptr device_addr{};
        LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(&device_addr, host_ptr, 0));
        return device_addr;
    }
    static void _unpin_host_memory(void *host_ptr) noexcept {
        LUISA_CHECK_CUDA(cuMemHostUnregister(host_ptr));
    }

public:
    CUDAPinnedMemoryBuffer(size_t size_bytes, bool write_combined) noexcept
        : CUDABuffer{size_bytes, Location::FORCE_HOST,
                     write_combined ?
                         CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED :
                         CU_MEMHOSTALLOC_DEVICEMAP} {}
    CUDAPinnedMemoryBuffer(void *host_mem, size_t size_bytes, bool write_combined) noexcept
        : CUDABuffer{_pin_host_memory(host_mem, size_bytes, write_combined), size_bytes},
          _external_host_memory{host_mem} {}
    ~CUDAPinnedMemoryBuffer() override {
        if (_external_host_memory) {
            _unpin_host_memory(_external_host_memory);
        }
    }
    [[nodiscard]] auto native_handle() const noexcept {
        return _external_host_memory ? _external_host_memory : host_address();
    }
};

CUDAPinnedMemoryExt::CUDAPinnedMemoryExt(CUDADevice *device) noexcept
    : _device{device} {}

BufferCreationInfo CUDAPinnedMemoryExt::_pin_host_memory(
    const Type *elem_type, size_t elem_count,
    void *host_ptr, const PinnedMemoryOption &option) noexcept {
    auto elem_stride = CUDACompiler::type_size(elem_type);
    auto size_bytes = elem_stride * elem_count;
    auto buffer = _device->with_handle([=] {
        return luisa::new_with_allocator<CUDAPinnedMemoryBuffer>(
            host_ptr, size_bytes, option.write_combined);
    });
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->native_handle();
    info.element_stride = elem_stride;
    info.total_size_bytes = size_bytes;
    return info;
}

BufferCreationInfo CUDAPinnedMemoryExt::_allocate_pinned_memory(
    const Type *elem_type, size_t elem_count,
    const PinnedMemoryOption &option) noexcept {
    auto elem_stride = CUDACompiler::type_size(elem_type);
    auto size_bytes = elem_stride * elem_count;
    auto buffer = _device->with_handle([=] {
        return luisa::new_with_allocator<CUDAPinnedMemoryBuffer>(
            size_bytes, option.write_combined);
    });
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->native_handle();
    info.element_stride = elem_stride;
    info.total_size_bytes = size_bytes;
    return info;
}

DeviceInterface *CUDAPinnedMemoryExt::device() const noexcept {
    return _device;
}

}// namespace luisa::compute::cuda
