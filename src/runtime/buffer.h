//
// Created by Mike Smith on 2021/2/3.
//

#pragma once

#include <cstddef>
#include <span>
#include <numeric>
#include <limits>

#include <core/logging.h>
#include <core/concepts.h>

namespace luisa::compute {

class Stream;

template<typename T>
class BufferView;

template<typename T>
class ConstBufferView;

class Buffer : public Noncopyable {

private:
    size_t _size_bytes;

public:
    explicit Buffer(size_t size_bytes) noexcept : _size_bytes{size_bytes} {}
    virtual ~Buffer() noexcept = default;

    Buffer(Buffer &&) noexcept = default;
    Buffer &operator=(Buffer &&) noexcept = default;

    virtual void upload(Stream *stream, const void *data, size_t offset, size_t size) = 0;
    virtual void download(Stream *stream, void *data, size_t offset, size_t size) const = 0;

    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }

    template<typename T>
    [[nodiscard]] auto view() noexcept { return BufferView<T>{this, 0u, _size_bytes}; }

    template<typename T>
    [[nodiscard]] auto view() const noexcept { return ConstBufferView<T>{this, 0u, _size_bytes}; }

    template<typename T>
    [[nodiscard]] auto const_view() const noexcept { return view<T>(); }
};

namespace detail {

template<typename Buffer, typename Element, template<typename> typename View>
class BufferViewInterface {

private:
    Buffer *_buffer;
    size_t _offset_bytes;
    size_t _size_bytes;

public:
    BufferViewInterface(Buffer *buffer, size_t offset_bytes, size_t size_bytes) noexcept
        : _buffer{buffer}, _offset_bytes{offset_bytes}, _size_bytes{size_bytes} {
        static constexpr auto element_alignment = alignof(Element);
        static constexpr auto element_size = sizeof(Element);
        if (_offset_bytes % element_alignment != 0u) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid offset {} for element with alignment {}.", _offset_bytes, element_alignment);
        }
        if (_size_bytes % element_size != 0u) {
            auto fit_size = _size_bytes / element_size * element_size;
            LUISA_WARNING_WITH_LOCATION(
                "Rounding buffer view size {} to {} to fit elements with size {}.", _size_bytes, fit_size, element_size);
            _size_bytes = fit_size;
        }
        if (_offset_bytes + _size_bytes > buffer->size_bytes()) {
            LUISA_ERROR_WITH_LOCATION(
                "Buffer view (with offset = {}, size = {}) overflows buffer with size {}.",
                _offset_bytes, _size_bytes, buffer->size_bytes());
        }
    }

    [[nodiscard]] auto buffer() const noexcept { return _buffer; }
    [[nodiscard]] auto size() const noexcept { return _size_bytes / sizeof(Element); }
    [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }

    void download(Stream *stream, Element *data) { _buffer->download(stream, data, offset_bytes(), size_bytes()); }

    [[nodiscard]] auto subview(size_t offset_elements, size_t size_elements) const noexcept {
        auto sub_offset = offset_elements * sizeof(Element);
        auto sub_size = size_elements * sizeof(Element);
        return View<Element>{_buffer, _offset_bytes + sub_offset, sub_size};
    }

    template<typename T>
    [[nodiscard]] auto as() const noexcept { return View<T>{_buffer, _offset_bytes, _size_bytes}; }
};

}// namespace detail

template<typename T>
struct ConstBufferView : public detail::BufferViewInterface<const Buffer, T, ConstBufferView> {
    using detail::BufferViewInterface<const Buffer, T, ConstBufferView>::BufferViewInterface;
};

template<typename T>
class BufferView : public detail::BufferViewInterface<Buffer, T, BufferView> {

public:
    using detail::BufferViewInterface<Buffer, T, BufferView>::BufferViewInterface;

    [[nodiscard]] auto const_view() const noexcept {
        return ConstBufferView<T>{this->buffer(), this->offset_bytes(), this->size_bytes()};
    }

    void upload(Stream *stream, const T *data) {
        this->buffer()->upload(stream, data, this->offset_bytes(), this->size_bytes());
    }
};

}// namespace luisa::compute
