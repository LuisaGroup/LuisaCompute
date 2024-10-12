#pragma once

#include <luisa/core/concepts.h>
#include <luisa/core/mathematics.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/device_interface.h>

namespace lc::validation {
class Stream;
}// namespace lc::validation

namespace luisa::compute {

namespace detail {

template<typename BufferOrView>
class BufferExprProxy;

LC_RUNTIME_API void error_buffer_copy_sizes_mismatch(size_t src, size_t dst) noexcept;
LC_RUNTIME_API void error_buffer_reinterpret_size_too_small(size_t size, size_t dst) noexcept;
LC_RUNTIME_API void error_buffer_subview_overflow(size_t offset, size_t ele_size, size_t size) noexcept;
LC_RUNTIME_API void error_buffer_invalid_alignment(size_t offset, size_t dst) noexcept;
LC_RUNTIME_API void error_buffer_size_is_zero() noexcept;

template<typename T>
struct is_buffer_impl : std::false_type {};

template<typename T>
struct is_buffer_view_impl : std::false_type {};

template<typename T>
struct buffer_element_impl {
    using type = T;
};

}// namespace detail

template<typename T>
using is_buffer = detail::is_buffer_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_view = detail::is_buffer_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::value;

template<typename T>
using buffer_element = detail::buffer_element_impl<std::remove_cvref_t<T>>;

template<typename T>
using buffer_element_t = typename buffer_element<T>::type;

template<typename T>
class SparseBuffer;

template<typename T>
class BufferView;

// check if this data type is legitimate
template<typename T>
constexpr bool is_valid_buffer_element_v =
    std::is_same_v<T, std::remove_cvref_t<T>> &&
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_destructible_v<T> &&
    (alignof(T) >= 4u);

// Buffer is a one-dimensional data structure that can be of any base data type, such as int, float2, struct or array
template<typename T>
class Buffer final : public Resource {

    static_assert(is_valid_buffer_element_v<T>);

private:
    size_t _size{};
    size_t _element_stride{};

private:
    friend class Device;
    friend class ResourceGenerator;
    friend class DxCudaInterop;
    friend class PinnedMemoryExt;
    Buffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _size{info.total_size_bytes / info.element_stride},
          _element_stride{info.element_stride} {}
    Buffer(DeviceInterface *device, size_t size) noexcept
        : Buffer{device, [&] {
                     if (size == 0) [[unlikely]] {
                         detail::error_buffer_size_is_zero();
                     }
                     return device->create_buffer(Type::of<T>(), size, nullptr);
                 }()} {}

public:
    Buffer() noexcept = default;
    ~Buffer() noexcept override {
        if (*this) { device()->destroy_buffer(handle()); }
    }
    Buffer(Buffer &&) noexcept = default;
    Buffer(Buffer const &) noexcept = delete;
    Buffer &operator=(Buffer &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    Buffer &operator=(Buffer const &) noexcept = delete;
    using Resource::operator bool;
    // properties
    [[nodiscard]] auto size() const noexcept {
        _check_is_valid();
        return _size;
    }
    [[nodiscard]] constexpr auto stride() const noexcept {
        _check_is_valid();
        return _element_stride;
    }
    [[nodiscard]] auto size_bytes() const noexcept {
        _check_is_valid();
        return _size * _element_stride;
    }
    // views
    [[nodiscard]] auto view_unchecked() const noexcept {
        return BufferView<T>{this->native_handle(), this->handle(), _element_stride, 0u, _size, _size};
    }

    [[nodiscard]] auto view() const noexcept {
        _check_is_valid();
        return BufferView<T>{this->native_handle(), this->handle(), _element_stride, 0u, _size, _size};
    }    
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept {
        return view().subview(offset, count);
    }
    // commands
    // copy buffer's data to pointer
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return this->view().copy_to(data);
    }
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return this->view().copy_from(data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        return this->view().copy_from(source);
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        _check_is_valid();
        return reinterpret_cast<const detail::BufferExprProxy<Buffer<T>> *>(this);
    }
};

// BufferView represents a reference to a Buffer. Use a BufferView that referenced to a destructed Buffer is an undefined behavior.
template<typename T>
class BufferView {
    friend class lc::validation::Stream;
    static_assert(is_valid_buffer_element_v<T>);

private:
    void *_native_handle;
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _element_stride;
    size_t _size;
    size_t _total_size;

private:
    friend class Buffer<T>;
    friend class SparseBuffer<T>;

    template<typename U>
    friend class BufferView;

public:
    BufferView(void *native_handle, uint64_t handle,
               size_t element_stride, size_t offset_bytes,
               size_t size, size_t total_size) noexcept
        : _native_handle{native_handle}, _handle{handle}, _offset_bytes{offset_bytes},
          _element_stride{element_stride}, _size{size}, _total_size{total_size} {
        if (_offset_bytes % alignof(T) != 0u) [[unlikely]] {
            detail::error_buffer_invalid_alignment(_offset_bytes, alignof(T));
        }
    }

    template<template<typename> typename B>
        requires(is_buffer_v<B<T>>)
    BufferView(const B<T> &buffer) noexcept : BufferView{buffer.view()} {}

    BufferView() noexcept : BufferView{nullptr, invalid_resource_handle, 0, 0, 0, 0} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != invalid_resource_handle; }

    // properties
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
    [[nodiscard]] constexpr auto stride() const noexcept { return _element_stride; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset() const noexcept { return _offset_bytes / _element_stride; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * _element_stride; }
    [[nodiscard]] auto total_size() const noexcept { return _total_size; }
    [[nodiscard]] auto total_size_bytes() const noexcept { return _total_size * _element_stride; }

    [[nodiscard]] auto original() const noexcept {
        return BufferView{_native_handle, _handle,
                          _element_stride, 0u,
                          _total_size, _total_size};
    }
    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements + offset_elements > _size) [[unlikely]] {
            detail::error_buffer_subview_overflow(offset_elements, size_elements, _size);
        }
        return BufferView{_native_handle, _handle, _element_stride,
                          _offset_bytes + offset_elements * _element_stride,
                          size_elements, _total_size};
    }
    // reinterpret cast buffer to another type U
    template<typename U>
        requires(!is_custom_struct_v<U>)
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        auto total_size_bytes = _total_size * _element_stride;
        return BufferView<U>{_native_handle, _handle, sizeof(U), _offset_bytes,
                             this->size_bytes() / sizeof(U), total_size_bytes / sizeof(U)};
    }
    // commands
    // copy buffer's data to pointer
    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return luisa::make_unique<BufferDownloadCommand>(_handle, offset_bytes(), size_bytes(), data);
    }
    // copy pointer's data to buffer
    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return luisa::make_unique<BufferUploadCommand>(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }
    // copy source buffer's data to buffer
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        if (source.size() != this->size()) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size(), this->size());
        }
        return luisa::make_unique<BufferCopyCommand>(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }
    // DSL interface
    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BufferExprProxy<BufferView<T>> *>(this);
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

namespace detail {
template<typename T>
struct is_buffer_impl<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view_impl<BufferView<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl<Buffer<T>> {
    using type = T;
};

template<typename T>
struct buffer_element_impl<BufferView<T>> {
    using type = T;
};

}// namespace detail

}// namespace luisa::compute
