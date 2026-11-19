#pragma once

#include <luisa/runtime/buffer.h>

namespace luisa::compute {

namespace detail {
class ByteBufferExprProxy;
}// namespace detail

class ByteBufferView;

class LUISA_RUNTIME_API ByteBuffer final : public Resource {

private:
    size_t _size_bytes{};

private:
    friend class Device;
    friend class ResourceGenerator;
    friend class DxCudaInterop;
    friend class VkCudaInterop;
    ByteBuffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept;
    ByteBuffer(DeviceInterface *device, size_t size_bytes) noexcept;

public:
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    ByteBuffer() noexcept = default;
    ~ByteBuffer() noexcept override { _release(); }
    ByteBuffer(ByteBuffer &&) noexcept = default;
    ByteBuffer(ByteBuffer const &) noexcept = default;
    ByteBuffer &operator=(ByteBuffer &&rhs) noexcept = default;
    ByteBuffer &operator=(ByteBuffer const &rhs) noexcept = default;
    using Resource::operator bool;
    BufferCreationInfo release() noexcept {
        return BufferCreationInfo{
            Resource::release(),
            1,
            _size_bytes};
    }
    [[nodiscard]] ByteBufferView view() const noexcept;
    [[nodiscard]] ByteBufferView view(size_t offset_bytes, size_t size_bytes) const noexcept;

    [[nodiscard]] luisa::unique_ptr<BufferUploadCommand> copy_from(const void *data) const noexcept;
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(ByteBufferView source) const noexcept;
    [[nodiscard]] luisa::unique_ptr<BufferDownloadCommand> copy_to(void *data) const noexcept;
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(ByteBufferView source) const noexcept;

    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(const Buffer<T> &source) const noexcept;

    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(BufferView<T> source) const noexcept;

    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(const Buffer<T> &dst) const noexcept;

    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(BufferView<T> dst) const noexcept;

    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::ByteBufferExprProxy *>(this);
    }

protected:
    void destroy() noexcept override;
};

class LUISA_RUNTIME_API ByteBufferView {
    friend class lc::validation::Stream;

private:
    void *_native_handle;
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;
    size_t _total_size;

public:
    ByteBufferView(void *native_handle, uint64_t handle,
                   size_t offset_bytes,
                   size_t size, size_t total_size) noexcept
        : _native_handle{native_handle}, _handle{handle}, _offset_bytes{offset_bytes},
          _size{size}, _total_size{total_size} {
    }

    ByteBufferView(const ByteBuffer &buffer) noexcept : ByteBufferView{buffer.view()} {}
    ByteBufferView(const ByteBufferView &) noexcept = default;

    template<typename T>
    explicit ByteBufferView(BufferView<T> buffer_view) noexcept
        : ByteBufferView{buffer_view.native_handle(), buffer_view.handle(),
                         buffer_view.offset_bytes(), buffer_view.size_bytes(), buffer_view.total_size_bytes()} {}

    template<typename T>
    explicit ByteBufferView(const Buffer<T> &buffer) noexcept
        : ByteBufferView{buffer.view()} {}

    ByteBufferView() noexcept : ByteBufferView{nullptr, invalid_resource_handle, 0, 0, 0} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != invalid_resource_handle; }

    // properties
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size; }
    [[nodiscard]] auto total_size_bytes() const noexcept { return _total_size; }

    [[nodiscard]] ByteBufferView original() const noexcept;
    [[nodiscard]] ByteBufferView subview(size_t offset_bytes, size_t size_bytes) const noexcept;

    template<typename U>
        requires(!is_custom_struct_v<U>)
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        return BufferView<U>{_native_handle, _handle, sizeof(U), _offset_bytes,
                             this->size_bytes() / sizeof(U), _total_size / sizeof(U)};
    }
    [[nodiscard]] luisa::unique_ptr<BufferDownloadCommand> copy_to(void *data) const noexcept;
    // copy pointer's data to buffer
    [[nodiscard]] luisa::unique_ptr<BufferUploadCommand> copy_from(const void *data) const noexcept;
    // copy source buffer's data to buffer
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(ByteBufferView source) const noexcept;
    // copy buffer's data to source buffer
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(ByteBufferView source) const noexcept;

    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(const Buffer<T> &dst) const noexcept { return copy_to(ByteBufferView{dst}); }
    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_to(BufferView<T> dst) const noexcept { return copy_to(ByteBufferView{dst}); }
    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(const Buffer<T> &source) const noexcept { return copy_from(ByteBufferView{source}); }
    template<typename T>
    [[nodiscard]] luisa::unique_ptr<BufferCopyCommand> copy_from(BufferView<T> source) const noexcept { return copy_from(ByteBufferView{source}); }
};

// some type traits
template<typename T>
struct is_byte_buffer : std::is_same<std::remove_cvref_t<T>, ByteBuffer> {};

template<typename T>
struct is_byte_buffer_view : std::is_same<std::remove_cvref_t<T>, ByteBufferView> {};

template<typename T>
struct is_byte_buffer_or_view : std::disjunction<is_byte_buffer<T>, is_byte_buffer_view<T>> {};

template<typename T>
constexpr auto is_byte_buffer_v = is_byte_buffer<T>::value;

template<typename T>
constexpr auto is_byte_buffer_view_v = is_byte_buffer_view<T>::value;

template<typename T>
constexpr auto is_byte_buffer_or_view_v = is_byte_buffer_or_view<T>::value;

// implementations for ByteBuffer
template<typename T>
luisa::unique_ptr<BufferCopyCommand> ByteBuffer::copy_from(const Buffer<T> &source) const noexcept {
    return view().copy_from(source);
}

template<typename T>
luisa::unique_ptr<BufferCopyCommand> ByteBuffer::copy_from(BufferView<T> source) const noexcept {
    return view().copy_from(source);
}

template<typename T>
luisa::unique_ptr<BufferCopyCommand> ByteBuffer::copy_to(const Buffer<T> &dst) const noexcept {
    return view().copy_to(dst);
}

template<typename T>
luisa::unique_ptr<BufferCopyCommand> ByteBuffer::copy_to(BufferView<T> dst) const noexcept {
    return view().copy_to(dst);
}

// implementations for typed Buffer
template<typename T>
luisa::unique_ptr<BufferCopyCommand> BufferView<T>::copy_to(const ByteBufferView &dst) const noexcept {
    return dst.copy_from(*this);
}

template<typename T>
luisa::unique_ptr<BufferCopyCommand> BufferView<T>::copy_from(const ByteBufferView &source) const noexcept {
    return source.copy_to(*this);
}

}// namespace luisa::compute
