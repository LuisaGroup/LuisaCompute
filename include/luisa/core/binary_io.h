#pragma once

#include <cstdio>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/filesystem.h>

namespace luisa {

class BinaryBlob {
    std::byte *_ptr{nullptr};
    size_t _size{0};
    void (*_disposer)(void *){nullptr};

public:
    BinaryBlob() noexcept = default;
    BinaryBlob(
        std::byte *ptr,
        size_t size,
        void (*disposer)(void *)) noexcept
        : _ptr{ptr},
          _size{size},
          _disposer{disposer} {}
    BinaryBlob(BinaryBlob const &) noexcept = delete;
    BinaryBlob(BinaryBlob &&rhs) noexcept
        : _ptr{rhs._ptr},
          _size{rhs._size},
          _disposer{rhs._disposer} {
        rhs._ptr = nullptr;
        rhs._size = 0;
        rhs._disposer = nullptr;
    }
    BinaryBlob &operator=(BinaryBlob const &rhs) noexcept = delete;
    BinaryBlob &operator=(BinaryBlob &&rhs) noexcept {
        this->~BinaryBlob();
        new (std::launder(this)) BinaryBlob{std::move(rhs)};
        return *this;
    }
    ~BinaryBlob() noexcept {
        if (_disposer) { _disposer(_ptr); }
    }
    [[nodiscard]] explicit operator luisa::span<const std::byte>() const noexcept {
        return {_ptr, _size};
    }
    [[nodiscard]] explicit operator luisa::span<std::byte>() noexcept {
        return {_ptr, _size};
    }
    [[nodiscard]] std::byte const *data() const noexcept {
        return _ptr;
    }
    [[nodiscard]] std::byte *data() noexcept {
        return _ptr;
    }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto empty() const noexcept { return _size == 0; }
};

class BinaryStream {

public:
    [[nodiscard]] virtual size_t length() const noexcept = 0;
    [[nodiscard]] virtual size_t pos() const noexcept = 0;
    virtual void read(luisa::span<std::byte> dst) noexcept = 0;
    [[nodiscard]] virtual BinaryBlob read(size_t expected_max_size) noexcept {
        auto len = std::min(expected_max_size, length());
        BinaryBlob blob{
            reinterpret_cast<std::byte *>(luisa::detail::allocator_allocate(len, 0)),
            len,
            [](void *ptr) { luisa::detail::allocator_deallocate(ptr, 0); }};
        read(blob);
        return blob;
    }
    virtual ~BinaryStream() noexcept = default;
};

class BinaryIO {

public:
    virtual ~BinaryIO() noexcept = default;
    virtual void clear_shader_cache() const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept = 0;
    // returns the path of the written file (if stored on disk, otherwise returns empty path)
    [[nodiscard]] virtual luisa::filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
    [[nodiscard]] virtual luisa::filesystem::path write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
    [[nodiscard]] virtual luisa::filesystem::path write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
};

}// namespace luisa
