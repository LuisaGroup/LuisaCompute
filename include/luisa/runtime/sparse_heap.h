#pragma once
#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute {

class LC_RUNTIME_API SparseBufferHeap : public Resource {

private:
    friend class Device;
    explicit SparseBufferHeap(DeviceInterface *device, size_t byte_size) noexcept;

public:
    SparseBufferHeap() noexcept = default;
    ~SparseBufferHeap() noexcept override;
    using Resource::operator bool;
    SparseBufferHeap(SparseBufferHeap &&) noexcept = default;
    SparseBufferHeap(SparseBufferHeap const &) noexcept = delete;
    SparseBufferHeap &operator=(SparseBufferHeap &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SparseBufferHeap &operator=(SparseBufferHeap const &) noexcept = delete;
};

class LC_RUNTIME_API SparseTextureHeap : public Resource {

private:
    friend class Device;
    explicit SparseTextureHeap(DeviceInterface *device, uint64_t byte_size) noexcept;

public:
    SparseTextureHeap() noexcept = default;
    ~SparseTextureHeap() noexcept override;
    using Resource::operator bool;
    SparseTextureHeap(SparseTextureHeap &&) noexcept = default;
    SparseTextureHeap(SparseTextureHeap const &) noexcept = delete;
    SparseTextureHeap &operator=(SparseTextureHeap &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    SparseTextureHeap &operator=(SparseTextureHeap const &) noexcept = delete;
};

}// namespace luisa::compute