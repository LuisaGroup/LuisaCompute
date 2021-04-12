//
// Created by Mike Smith on 2021/3/2.
//

#pragma once

#include <runtime/command.h>
#include <runtime/device.h>

namespace luisa::compute {

namespace detail {
template<typename T>
struct Expr;
}

template<typename T>
class BufferView;

#define LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)                    \
    static_assert(alignof(T) <= 16u);                         \
    static_assert(std::is_same_v<T, std::remove_cvref_t<T>>); \
    static_assert(std::is_trivially_copyable_v<T>);           \
    static_assert(std::is_trivially_destructible_v<T>);

template<typename T>
class Buffer : public concepts::Noncopyable {

    LUISA_CHECK_BUFFER_ELEMENT_TYPE(T)

private:
    Device::Interface *_device;
    size_t _size;
    uint64_t _handle;

private:
    friend class Device;
    Buffer(Device &device, size_t size) noexcept
        : _device{device.interface()},
          _size{size},
          _handle{device.interface()->create_buffer(size * sizeof(T))} {}

public:
    Buffer(Buffer &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._handle} { another._device = nullptr; }

    Buffer &operator=(Buffer &&rhs) noexcept {
        if (&rhs != this) {
            _device->dispose_buffer(_handle);
            _device = rhs._device;
            _handle = rhs._handle;
            _size = rhs._handle;
            rhs._device = nullptr;
        }
        return *this;
    }

    ~Buffer() noexcept {
        if (_device != nullptr /* not moved */) {
            _device->dispose_buffer(_handle);
        }
    }

    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }
    [[nodiscard]] auto view() const noexcept { return BufferView<T>{_handle, 0u, _size}; }
    [[nodiscard]] auto view(size_t offset, size_t count) const noexcept { return view().subview(offset, count); }

    [[nodiscard]] auto copy_to(T *data) const noexcept { return this->view().copy_to(data); }
    [[nodiscard]] auto copy_from(const T *data) { return this->view().copy_from(data); }
    [[nodiscard]] auto copy_from(BufferView<T> source) { return this->view().copy_from(source); }

    template<typename I>
    [[nodiscard]] decltype(auto) operator[](I &&i) const noexcept {
        return this->view()[std::forward<I>(i)];
    }
};

template<typename T>
class BufferView {

private:
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;

private:
    friend class Buffer<T>;
    BufferView(uint64_t handle, size_t offset_bytes, size_t size) noexcept
        : _handle{handle}, _offset_bytes{offset_bytes}, _size{size} {
        if (_offset_bytes % alignof(T) != 0u) {
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
        if (size_elements * sizeof(T) + offset_elements > _size) {
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
        if (byte_size < sizeof(U)) {
            LUISA_ERROR_WITH_LOCATION(
                "Unable to hold any element (with size = {}) in buffer view (with size = {}).",
                sizeof(U), byte_size);
        }
        return BufferView{this->device(), this->handle(), this->offset_bytes(), byte_size / sizeof(U)};
    }

    [[nodiscard]] auto copy_to(T *data) const {
        if (reinterpret_cast<size_t>(data) % alignof(T) != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid host pointer {} for elements with alignment {}.",
                fmt::ptr(data), alignof(T));
        }
        return BufferDownloadCommand::create(_handle, offset_bytes(), size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(const T *data) {
        return BufferUploadCommand::create(this->handle(), this->offset_bytes(), this->size_bytes(), data);
    }

    [[nodiscard]] auto copy_from(BufferView<T> source) {
        if (source.size() != this->size()) {
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
    [[nodiscard]] decltype(auto) operator[](I &&i) const noexcept {
        return detail::Expr<Buffer<T>>{*this}[std::forward<I>(i)];
    }
};

template<typename T>
BufferView(const Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T>) -> BufferView<T>;

#undef LUISA_CHECK_BUFFER_ELEMENT_TYPE

template<typename T>
struct is_buffer : std::false_type {};

template<typename T>
struct is_buffer<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view : std::false_type {};

template<typename T>
struct is_buffer_view<BufferView<T>> : std::true_type {};

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::view;

}// namespace luisa::compute
