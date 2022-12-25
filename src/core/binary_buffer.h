//
// Created by Mike Smith on 2022/6/18.
//

#pragma once

#include <bit>
#include <core/stl/vector.h>
#include <core/stl/memory.h>

namespace luisa {

class BinaryBuffer {

private:
    luisa::vector<std::byte> _bytes;

private:
    void _write_bytes(const void *data, size_t size, size_t alignment) noexcept;

public:
    explicit BinaryBuffer(size_t capacity = 1_mb) noexcept
        : _bytes{capacity} {}
    explicit BinaryBuffer(luisa::vector<std::byte> &&bytes) noexcept
        : _bytes{std::move(bytes)} {}
    template<typename T>
    void write(T value) noexcept {
        _write_bytes(&value, sizeof(T), alignof(T));
    }
    template<typename T>
    void write(const T *data, size_t size) noexcept {
        _write_bytes(data, size * sizeof(T), alignof(T));
    }
    void clear() noexcept { _bytes.clear(); }
    [[nodiscard]] auto data() const noexcept { return _bytes.data(); }
    [[nodiscard]] auto size() const noexcept { return _bytes.size(); }
    [[nodiscard]] auto move() noexcept { return std::move(_bytes); }
};

class BinaryBufferReader {

private:
    const std::byte *_bytes;
    size_t _offset;
    size_t _size;

public:
    BinaryBufferReader(const std::byte *bytes, size_t size) noexcept
        : _bytes{bytes}, _offset{0u}, _size{size} {}
    BinaryBufferReader(luisa::span<const std::byte> bytes) noexcept
        : BinaryBufferReader{bytes.data(), bytes.size()} {}
    BinaryBufferReader(const BinaryBuffer &buffer) noexcept
        : BinaryBufferReader{buffer.data(), buffer.size()} {}
    template<typename T>
    [[nodiscard]] auto read() noexcept {
        _offset = align(_offset, alignof(T));
        assert(_offset + sizeof(T) <= _size);
        auto ptr = _bytes + _offset;
        _offset += sizeof(T);
        return *reinterpret_cast<const T *>(ptr);// FIXME: this is UB
    }
    template<typename T>
    void read(T *dst, size_t n) noexcept {
        _offset = align(_offset, alignof(T));
        auto size = n * sizeof(T);
        assert(_offset + size <= _size);
        auto ptr = _bytes + _offset;
        _offset += size;
        memcpy(dst, ptr, size);
    }
};

}// namespace luisa
