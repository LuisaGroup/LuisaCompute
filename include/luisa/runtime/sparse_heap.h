#pragma once

#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute {

namespace detail {
LUISA_RUNTIME_API void check_sparse_heap_provenance(
    const Resource &resource,
    const Resource &heap) noexcept;
}// namespace detail

class LUISA_RUNTIME_API SparseBufferHeap : public Resource {

private:
    friend class Device;
    explicit SparseBufferHeap(DeviceInterface *device, size_t byte_size) noexcept;

public:
    SparseBufferHeap() noexcept = default;
    ~SparseBufferHeap() noexcept override { _release(); }
    using Resource::operator bool;
    using Resource::release;
    SparseBufferHeap(SparseBufferHeap &&) noexcept = default;
    SparseBufferHeap(SparseBufferHeap const &) noexcept = default;
    SparseBufferHeap &operator=(SparseBufferHeap &&rhs) noexcept = default;
    SparseBufferHeap &operator=(SparseBufferHeap const &rhs) noexcept = default;

protected:
    void destroy() noexcept override;
};

class LUISA_RUNTIME_API SparseTextureHeap : public Resource {

private:
    friend class Device;
    explicit SparseTextureHeap(DeviceInterface *device, size_t byte_size) noexcept;

public:
    SparseTextureHeap() noexcept = default;
    ~SparseTextureHeap() noexcept override { _release(); }
    using Resource::operator bool;
    using Resource::release;
    SparseTextureHeap(SparseTextureHeap &&) noexcept = default;
    SparseTextureHeap(SparseTextureHeap const &) noexcept = default;
    SparseTextureHeap &operator=(SparseTextureHeap &&rhs) noexcept = default;
    SparseTextureHeap &operator=(SparseTextureHeap const &rhs) noexcept = default;

protected:
    void destroy() noexcept override;
};

}// namespace luisa::compute
