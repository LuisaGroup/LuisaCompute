#pragma once

#include <luisa/runtime/buffer.h>

namespace luisa::compute {

namespace detail {
class ByteBufferExprProxy;
}// namespace detail

class LC_RUNTIME_API ByteBuffer final : public Resource {

private:
    size_t _size_bytes{};

private:
    friend class Device;
    friend class ResourceGenerator;
    ByteBuffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept;
    ByteBuffer(DeviceInterface *device, size_t size_bytes) noexcept;

public:
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    ByteBuffer() noexcept = default;
    ~ByteBuffer() noexcept override;
    ByteBuffer(ByteBuffer &&) noexcept = default;
    ByteBuffer(ByteBuffer const &) noexcept = delete;
    ByteBuffer &operator=(ByteBuffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    ByteBuffer &operator=(ByteBuffer const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        _check_is_valid();
        return luisa::make_unique<BufferDownloadCommand>(handle(), 0u, _size_bytes, data);
    }
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        _check_is_valid();
        return luisa::make_unique<BufferUploadCommand>(handle(), 0u, _size_bytes, data);
    }
    [[nodiscard]] auto copy_from(const void *data, size_t buffer_offset, size_t size_bytes) noexcept {
        _check_is_valid();
        if (size_bytes > _size_bytes) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(size_bytes, _size_bytes);
        }
        return luisa::make_unique<BufferUploadCommand>(handle(), buffer_offset, size_bytes, data);
    }
    template<typename T>
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        _check_is_valid();
        if (source.size_bytes() != _size_bytes) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size_bytes(), _size_bytes);
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            source.offset_bytes(), 0u,
            this->size_bytes());
    }
    [[nodiscard]] auto copy_from(const ByteBuffer &source, size_t offset, size_t size_bytes) noexcept {
        _check_is_valid();
        if (size_bytes > _size_bytes) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(size_bytes, _size_bytes);
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            offset, 0u,
            size_bytes);
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::ByteBufferExprProxy *>(this);
    }
};

namespace detail {
LC_RUNTIME_API void error_buffer_size_not_aligned(size_t align) noexcept;
template<>
struct is_buffer_impl<ByteBuffer> : std::true_type {};
}// namespace detail

}// namespace luisa::compute
