//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <cstddef>
#include <numeric>
#include <limits>
#include <span>
#include <utility>
#include <type_traits>

#include <core/logging.h>
#include <core/concepts.h>
#include <core/data_types.h>

#include <runtime/device.h>

namespace luisa::compute {

template<typename T>
class BufferView;

template<typename T>
class alignas(8) Buffer : public concepts::Noncopyable {

    static_assert(alignof(T) <= 16u);
    static_assert(std::is_same_v<T, std::remove_cvref_t<T>>);

private:
    Device *_device;
    uint64_t _handle;
    size_t _size;

public:
    Buffer(Device *device, size_t size) noexcept
        : _device{device},
          _handle{device->_create_buffer(size * sizeof(T))},
          _size{size} {}

    Buffer(Device *device, std::span<T> span) noexcept
        : _device{device},
          _handle{device->_create_buffer_with_data(span.size_bytes(), span.data())},
          _size{span.size()} {}

    Buffer(Buffer &&another) noexcept
        : _device{another._device},
          _handle{another._handle},
          _size{another._handle} { another._device = nullptr; }

    Buffer &operator=(Buffer &&rhs) noexcept {
        if (&rhs != this) {
            _device->_dispose_buffer(_handle);
            _device = rhs._device;
            _handle = rhs._handle;
            _size = rhs._handle;
            rhs._device = nullptr;
        }
        return *this;
    }

    ~Buffer() noexcept {
        if (_device != nullptr /* not moved */) {
            _device->_dispose_buffer(_handle);
        }
    }

    [[nodiscard]] auto view() const noexcept { return BufferView<T>{_device, _handle, 0u, _size}; }

    template<typename Index>
    [[nodiscard]] decltype(auto) operator[](Index &&index) const noexcept { return view()[std::forward<Index>(index)]; }
};

template<typename T> requires concepts::SpanConvertible<T>
Buffer(Device *, T &&) -> Buffer<typename std::remove_cvref_t<T>::value_type>;

template<typename T>
Buffer(Buffer<T> &&) -> Buffer<T>;

class BufferUploadCommand {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    const void *_data;

private:
    BufferUploadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, const void *data) noexcept
        : _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
};

class BufferDownloadCommand {

private:
    uint64_t _handle;
    size_t _offset;
    size_t _size;
    void *_data;

private:
    BufferDownloadCommand(uint64_t handle, size_t offset_bytes, size_t size_bytes, void *data) noexcept
        : _handle{handle}, _offset{offset_bytes}, _size{size_bytes}, _data{data} {}

public:
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto offset() const noexcept { return _offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto data() const noexcept { return _data; }
};

class BufferCopyCommand {

private:
    uint64_t _src_handle;
    uint64_t _dst_handle;
    size_t _src_offset;
    size_t _dst_offset;
    size_t _size;

private:
    BufferCopyCommand(uint64_t src, uint64_t dst, size_t src_offset, size_t dst_offset, size_t size) noexcept
        : _src_handle{src}, _dst_handle{dst}, _src_offset{src_offset}, _dst_offset{dst_offset}, _size{size} {}

public:
    [[nodiscard]] auto src_handle() const noexcept { return _src_handle; }
    [[nodiscard]] auto dst_handle() const noexcept { return _dst_handle; }
    [[nodiscard]] auto src_offset() const noexcept { return _src_offset; }
    [[nodiscard]] auto dst_offset() const noexcept { return _dst_offset; }
    [[nodiscard]] auto size() const noexcept { return _size; }
};

namespace detail {

template<typename Index>
struct BufferAccess {
    static_assert(always_false<Index>, "Invalid BufferAccess");
};

}// namespace detail

template<typename T>
class alignas(8) BufferView {

private:
    Device *_device;
    uint64_t _handle;
    size_t _offset_bytes;
    size_t _size;

protected:
    friend class Buffer<T>;
    BufferView(Device *device, uint64_t handle, size_t offset_bytes, size_t size) noexcept
        : _device{device}, _handle{handle}, _offset_bytes{offset_bytes}, _size{size} {
        if (_offset_bytes % alignof(T) != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid buffer view offset {} for elements with alignment {}.",
                _offset_bytes, alignof(T));
        }
    }

public:
    BufferView(Buffer<T> &buffer) noexcept : BufferView{buffer.view()} {}

    [[nodiscard]] auto device() const noexcept { return _device; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size * sizeof(T); }

    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        if (size_elements * sizeof(T) + offset_elements > _size) {
            LUISA_ERROR_WITH_LOCATION(
                "Subview (with offset_elements = {}, size_elements = {}) overflows buffer view (with size_elements = {}).",
                offset_elements, size_elements, _size);
        }
        return BufferView{_device, _handle, _offset_bytes + offset_elements * sizeof(T), size_elements};
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

    [[nodiscard]] auto download(T *data) const {
        if (reinterpret_cast<size_t>(data) % alignof(T) != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid host pointer {} for elements with alignment {}.",
                fmt::ptr(data), alignof(T));
        }
        BufferDownloadCommand{_handle, offset_bytes(), size_bytes(), data};
    }

    [[nodiscard]] auto upload(const T *data) {
        return BufferUploadCommand{this->handle(), this->offset_bytes(), this->size_bytes(), data};
    }

    [[nodiscard]] auto copy(BufferView<T> source) {
        if (source.device() != this->device()) {
            LUISA_ERROR_WITH_LOCATION("Incompatible buffer views created on different devices.");
        }
        if (source.size() != this->size()) {
            LUISA_ERROR_WITH_LOCATION(
                "Incompatible buffer views with different element counts (src = {}, dst = {}).",
                source.size(), this->size());
        }
        return BufferCopyCommand{
            source.handle(), this->handle(),
            source.offset_bytes(), this->offset_bytes(),
            this->size_bytes()};
    }

    template<typename Index>
    [[nodiscard]] decltype(auto) operator[](Index &&index) const noexcept {
        return detail::BufferAccess<std::remove_cvref_t<Index>>{}(*static_cast<const BufferView *>(this), std::forward<Index>(index));
    }
};

template<typename T>
BufferView(Buffer<T> &) -> BufferView<T>;

template<typename T>
BufferView(BufferView<T> &) -> BufferView<T>;

}// namespace luisa::compute
