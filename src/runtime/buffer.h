//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <core/atomic.h>
#include <core/concepts.h>
#include <runtime/command.h>
#include <runtime/resource.h>

namespace luisa::compute {

template<typename T>
struct Expr;

template<typename T>
class BufferView;

template<typename T>
using is_valid_buffer_element = std::conjunction<
    std::is_same<T, std::remove_cvref_t<T>>,
    std::is_trivially_copyable<T>,
    std::is_trivially_destructible<T>,
    std::bool_constant<(alignof(T) >= 4u)>>;

template<typename T>
constexpr auto is_valid_buffer_element_v = is_valid_buffer_element<T>::value;

template<typename T>
class Buffer final : public Resource {

    static_assert(is_valid_buffer_element_v<T>);

private:
    size_t _size{};

private:
    friend class Device;
    Buffer(Device::Interface *device, size_t size) noexcept
        : Resource{
              device, Tag::BUFFER,
              device->create_buffer(size * sizeof(T))},
          _size{size} {}

public:
    Buffer() noexcept = default;
    using Resource::operator bool;
    [[nodiscard]] void *native_handle() const noexcept { return device()->buffer_native_handle(handle()); }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }
    [[nodiscard]] auto view() const noexcept { return BufferView<T>{this->handle(), 0u, _size}; }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept { return view().subview(offset, count); }

    [[nodiscard]] auto copy_to(void *data) const noexcept { return this->view().copy_to(data); }
    [[nodiscard]] auto copy_from(const void *data) { return this->view().copy_from(data); }
    [[nodiscard]] auto copy_from(BufferView<T> source) { return this->view().copy_from(source); }

    template<typename I>
    [[nodiscard]] decltype(auto) read(I &&i) const noexcept { return this->view().read(std::forward<I>(i)); }

    template<typename I, typename V>
    void write(I &&i, V &&v) const noexcept { this->view().write(std::forward<I>(i), std::forward<V>(v)); }

    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return this->view().atomic(std::forward<I>(i));
    }
};

template<typename T>
class BufferView {

    static_assert(is_valid_buffer_element_v<T>);

private:
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;

private:
    friend class Heap;
    friend class Buffer<T>;
    BufferView(uint64_t handle, size_t offset_bytes, size_t size) noexcept
        : _handle{handle}, _offset_bytes{offset_bytes}, _size{size} {
        if (_offset_bytes % alignof(T) != 0u) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid buffer view offset {} for elements with alignment {}.",
                _offset_bytes, alignof(T));
        }
    }

public:
    BufferView(const Buffer<T> &buffer) noexcept : BufferView{buffer.view()} {}

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }

    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements * sizeof(T) + offset_elements > _size) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Subview (with offset_elements = {}, size_elements = {}) "
                "overflows buffer view (with size_elements = {}).",
                offset_elements, size_elements, _size);
        }
        return BufferView{_handle, _offset_bytes + offset_elements * sizeof(T), size_elements};
    }

    template<typename U>
    [[nodiscard]] auto as() const noexcept {
        auto byte_size = this->size() * sizeof(T);
        if (byte_size < sizeof(U)) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Unable to hold any element (with size = {}) in buffer view (with size = {}).",
                sizeof(U), byte_size);
        }
        return BufferView{this->device(), this->handle(), this->offset_bytes(), byte_size / sizeof(U)};
    }

    [[nodiscard]] auto copy_to(void *data) const {
        return BufferDownloadCommand::create(_handle, offset_bytes(), size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(const void *data) {
        return BufferUploadCommand::create(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(BufferView<T> source) {
        if (source.size() != this->size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION(
                "Incompatible buffer views with different element counts (src = {}, dst = {}).",
                source.size(), this->size());
        }
        return BufferCopyCommand::create(
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes());
    }

    template<typename I>
    [[nodiscard]] decltype(auto) read(I &&i) const noexcept {
        return Expr<Buffer<T>>{*this}.read(std::forward<I>(i));
    }

    template<typename I, typename V>
    void write(I &&i, V &&v) const noexcept {
        Expr<Buffer<T>>{*this}.write(std::forward<I>(i), std::forward<V>(v));
    }

    template<typename I>
    [[nodiscard]] decltype(auto) atomic(I &&i) const noexcept {
        return Expr<Buffer<T>>{*this}.atomic(std::forward<I>(i));
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

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
