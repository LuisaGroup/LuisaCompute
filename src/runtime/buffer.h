//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <core/concepts.h>
#include <core/mathematics.h>
#include <runtime/command.h>
#include <runtime/resource.h>
#include <runtime/device_interface.h>

namespace luisa::compute {

namespace detail {

template<typename BufferOrView>
struct BufferExprProxy;

LC_RUNTIME_API void error_buffer_copy_sizes_mismatch(size_t src, size_t dst) noexcept;
LC_RUNTIME_API void error_buffer_reinterpret_size_too_small(size_t size, size_t dst) noexcept;
LC_RUNTIME_API void error_buffer_subview_overflow(size_t offset, size_t ele_size, size_t size) noexcept;
LC_RUNTIME_API void error_buffer_invalid_alignment(size_t offset, size_t dst) noexcept;

}// namespace detail

template<typename T>
class BufferView;

template<typename T>
constexpr bool is_valid_buffer_element_v =
    std::is_same_v<T, std::remove_cvref_t<T>> &&
    std::is_trivially_copyable_v<T> &&
    std::is_trivially_destructible_v<T> &&
    (alignof(T) >= 4u);

template<typename T>
class Buffer final : public Resource {

    static_assert(is_valid_buffer_element_v<T>);

private:
    size_t _size{};
    size_t _element_stride{};

private:
    friend class Device;
    friend class ResourceGenerator;
    Buffer(DeviceInterface *device, const BufferCreationInfo &info) noexcept
        : Resource{device, Tag::BUFFER, info},
          _size{info.total_size_bytes / info.element_stride},
          _element_stride{info.element_stride} {}

public:
    Buffer(DeviceInterface *device, size_t size) noexcept
        : Buffer{device, device->create_buffer(Type::of<T>(), size)} {}
    Buffer() noexcept = default;
    Buffer(Buffer &&) noexcept = default;
    Buffer(Buffer const &) noexcept = delete;
    Buffer &operator=(Buffer &&) noexcept = default;
    Buffer &operator=(Buffer const &) noexcept = delete;
    using Resource::operator bool;
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] constexpr auto stride() const noexcept { return _element_stride; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * _element_stride; }
    [[nodiscard]] auto view() const noexcept { return BufferView<T>{this->handle(), _element_stride, 0u, _size, _size}; }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept { return view().subview(offset, count); }

    [[nodiscard]] auto copy_to(void *data) const noexcept { return this->view().copy_to(data); }
    [[nodiscard]] auto copy_from(const void *data) noexcept { return this->view().copy_from(data); }
    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept { return this->view().copy_from(source); }

    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return this->view().atomic(std::forward<I>(i));
    }

    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BufferExprProxy<Buffer<T>> *>(this);
    }
};

template<typename T>
class BufferView {

    static_assert(is_valid_buffer_element_v<T>);

private:
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _element_stride;
    size_t _size;
    size_t _total_size;

private:
    friend class Buffer<T>;
    friend class ResourceGenerator;
    template<typename U>
    friend class BufferView;
    BufferView(uint64_t handle, size_t element_stride, size_t offset_bytes, size_t size, size_t total_size) noexcept
        : _handle{handle}, _element_stride{element_stride}, _offset_bytes{offset_bytes}, _size{size}, _total_size{total_size} {
        if (_offset_bytes % alignof(T) != 0u) [[unlikely]] {
            detail::error_buffer_invalid_alignment(_offset_bytes, alignof(T));
        }
    }

public:
    BufferView() noexcept : BufferView{invalid_resource_handle, 0, 0, 0} {}
    [[nodiscard]] explicit operator bool() const noexcept { return _handle != invalid_resource_handle; }
    BufferView(const Buffer<T> &buffer) noexcept : BufferView{buffer.view()} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] constexpr auto stride() const noexcept { return _element_stride; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset() const noexcept { return _offset_bytes / _element_stride; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * _element_stride; }

    [[nodiscard]] auto original() const noexcept {
        return BufferView{_handle, _element_stride, 0u, _total_size, _total_size};
    }

    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements + offset_elements > _size) [[unlikely]] {
            detail::error_buffer_subview_overflow(offset_elements, size_elements, _size);
        }
        return BufferView{_handle, _element_stride, _offset_bytes + offset_elements * _element_stride, size_elements, _total_size};
    }

    template<typename U>
        requires(!is_custom_struct_v<U>())
    [[nodiscard]] auto as() const noexcept {
        if (this->size_bytes() < sizeof(U)) [[unlikely]] {
            detail::error_buffer_reinterpret_size_too_small(sizeof(U), this->size_bytes());
        }
        return BufferView<U>{_handle, _offset_bytes, this->size_bytes() / sizeof(U), _total_size};
    }

    [[nodiscard]] auto copy_to(void *data) const noexcept {
        return BufferDownloadCommand::create(_handle, offset_bytes(), size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(const void *data) noexcept {
        return BufferUploadCommand::create(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(BufferView<T> source) noexcept {
        if (source.size() != this->size()) [[unlikely]] {
            detail::error_buffer_copy_sizes_mismatch(source.size(), this->size());
        }
        return BufferCopyCommand::create(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }

    [[nodiscard]] auto operator->() const noexcept {
        return reinterpret_cast<const detail::BufferExprProxy<BufferView<T>> *>(this);
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

// some traits
namespace detail {

template<typename T>
struct is_buffer_impl : std::false_type {};

template<typename T>
struct is_buffer_impl<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view_impl : std::false_type {};

template<typename T>
struct is_buffer_view_impl<BufferView<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl {
    using type = T;
};

template<typename T>
struct buffer_element_impl<Buffer<T>> {
    using type = T;
};

template<typename T>
struct buffer_element_impl<BufferView<T>> {
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

}// namespace luisa::compute
