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
    bool _is_compressed_type{};
    explicit SparseTextureHeap(DeviceInterface *device, size_t byte_size, bool is_compressed_type) noexcept;

public:
    [[nodiscard]] auto is_compressed_type() const noexcept { return _is_compressed_type; }
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