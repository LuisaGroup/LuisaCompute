//
// Created by Mike Smith on 2021/9/24.
//

#pragma once

#include <span>
#include <vector>
#include <memory>

#include <asio.hpp>

#include <core/basic_types.h>
#include <core/mathematics.h>
#include <core/logging.h>

namespace luisa::compute {

class BinaryBuffer {

    using size_field = uint32_t;

private:
    std::vector<std::byte> _data;
    size_t _cursor{sizeof(size_field)};

private:
    void _reserve_for(size_t n) noexcept {
        if (auto s = next_pow2(_data.size() + n);
            s > _data.capacity()) {
            _data.reserve(s);
        }
    }

    [[nodiscard]] auto _resize_for(size_t n) noexcept {
        _reserve_for(n);
        auto s = _data.size();
        _data.resize(s + n);
        return _data.data() + s;
    }

    auto &_write(const void *p, size_t n) noexcept {
        if (auto s = _resize_for(n); p != nullptr) {
            std::memcpy(s, p, n);
        }
        return *this;
    }

    auto &_read(void *p, size_t n) noexcept {
        if (_cursor + n > _data.size()) [[unlikely]] {
            LUISA_ERROR_WITH_LOCATION("BinaryBuffer overflow.");
        }
        if (p != nullptr) {
            std::memcpy(p, _data.data() + _cursor, n);
        }
        _cursor += n;
        return *this;
    }

public:
    BinaryBuffer() noexcept : _data(sizeof(size_field)){};

    void clear() noexcept {
        _data.resize(sizeof(size_field));
        _cursor = sizeof(size_field);
        // clear size field
        write_size();
    }

    // read
    template<typename T>
        requires std::is_trivially_copyable_v<T>
    auto &read(T &x) noexcept { return _read(&x, sizeof(T)); }
    auto &read(void *p, size_t n) noexcept { return _read(p, n); }

    // write
    template<typename T>
        requires std::is_trivially_copyable_v<T>
    auto &write(const T &x) noexcept { return _write(&x, sizeof(T)); }
    auto &write(const void *p, size_t n) noexcept { return _write(p, n); }

    // skip some bytes when reading/writing
    auto &read_skip(size_t n) noexcept { return _read(nullptr, n); }
    auto &write_skip(size_t n) noexcept { return _write(nullptr, n); }

    // views (std::span)
    [[nodiscard]] auto view() noexcept { return std::span{_data}; }
    [[nodiscard]] auto view() const noexcept { return std::span{_data}; }
    [[nodiscard]] auto tail() noexcept { return std::span{_data}.subspan(_cursor); }
    [[nodiscard]] auto tail() const noexcept { return std::span{_data}.subspan(_cursor); }
    [[nodiscard]] auto asio_buffer() noexcept { return asio::buffer(_data); }
    [[nodiscard]] auto asio_buffer() const noexcept { return asio::buffer(_data); }
    [[nodiscard]] auto asio_buffer_tail() noexcept {
        auto v = tail();
        return asio::buffer(v.data(), v.size_bytes());
    }
    [[nodiscard]] auto asio_buffer_tail() const noexcept {
        auto v = tail();
        return asio::buffer(v.data(), v.size_bytes());
    }

    // write size into the first sizeof(size_field) bytes of the buffer
    void write_size() noexcept {
        auto size = static_cast<size_field>(_data.size() - sizeof(size_field));
        std::memcpy(_data.data(), &size, sizeof(size_field));
    }

    // reads the size from the first sizeof(size_field) bytes of the buffer
    [[nodiscard]] auto read_size() const noexcept {
        auto size = static_cast<size_field>(0u);
        std::memcpy(&size, _data.data(), sizeof(size_field));
        return size;
    }
};

}// namespace luisa::compute