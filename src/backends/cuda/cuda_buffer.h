#pragma once

#include <cuda.h>

namespace luisa::compute::cuda {

class CUDABufferBase {

private:
    void *_host_address{nullptr};
    CUdeviceptr _device_address;
    size_t _size_bytes : 63;
    size_t _external_memory : 1;

public:
    enum struct Location {
        PREFER_DEVICE,
        FORCE_HOST
    };

public:
    explicit CUDABufferBase(size_t size_bytes,
                            Location loc = Location::PREFER_DEVICE,
                            int host_alloc_flags = 0) noexcept;
    CUDABufferBase(CUdeviceptr external_ptr, size_t size_bytes) noexcept;
    virtual ~CUDABufferBase() noexcept;
    CUDABufferBase(CUDABufferBase &&) = delete;
    CUDABufferBase(const CUDABufferBase &) = delete;
    CUDABufferBase &operator=(CUDABufferBase &&) = delete;
    CUDABufferBase &operator=(const CUDABufferBase &) = delete;
    [[nodiscard]] auto device_address() const noexcept { return _device_address; }
    [[nodiscard]] auto host_address() const noexcept { return _host_address; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    [[nodiscard]] virtual bool is_indirect() const noexcept { return false; }
    [[nodiscard]] auto is_host_memory() const noexcept { return _host_address != nullptr; }
    virtual void set_name(luisa::string &&name) noexcept {
        /* currently do nothing */
    }
};

class CUDABuffer : public CUDABufferBase {

public:
    struct Binding {
        CUdeviceptr handle;
        size_t size;
    };

public:
    using CUDABufferBase::CUDABufferBase;
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
};

class CUDAIndirectDispatchBuffer : public CUDABufferBase {

public:
    struct Binding {
        CUdeviceptr buffer;
        uint offset;
        uint capacity;
    };

    struct alignas(16) Header {
        uint size;
    };

    struct alignas(16) Dispatch {
        uint3 block_size;
        uint4 dispatch_size_and_kernel_id;
    };

private:
    size_t _capacity;

public:
    explicit CUDAIndirectDispatchBuffer(size_t capacity) noexcept;
    [[nodiscard]] auto capacity() const noexcept { return _capacity; }
    [[nodiscard]] bool is_indirect() const noexcept override { return true; }
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
};

}// namespace luisa::compute::cuda
