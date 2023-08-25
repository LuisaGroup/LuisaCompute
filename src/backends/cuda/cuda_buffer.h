#pragma once

#include <cuda.h>

namespace luisa::compute::cuda {

class CUDABufferBase {

private:
    CUdeviceptr _handle;
    size_t _size_bytes;

public:
    explicit CUDABufferBase(size_t size_bytes) noexcept;
    virtual ~CUDABufferBase() noexcept;
    CUDABufferBase(CUDABufferBase &&) = delete;
    CUDABufferBase(const CUDABufferBase &) = delete;
    CUDABufferBase &operator=(CUDABufferBase &&) = delete;
    CUDABufferBase &operator=(const CUDABufferBase &) = delete;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    [[nodiscard]] virtual bool is_indirect() const noexcept { return false; }
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

