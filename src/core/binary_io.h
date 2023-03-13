#pragma once

#include <cstdio>
#include <core/stl/memory.h>
#include <core/stl/string.h>

namespace luisa {

class BinaryStream {

public:
    [[nodiscard]] virtual size_t length() const noexcept = 0;
    [[nodiscard]] virtual size_t pos() const noexcept = 0;
    virtual void read(luisa::span<std::byte> dst) noexcept = 0;
    virtual ~BinaryStream() noexcept = default;
};

class LC_CORE_API BinaryStringStream : public BinaryStream {

private:
    luisa::string _data;
    size_t _pos{0u};

public:
    explicit BinaryStringStream(luisa::string data) noexcept
        : _data{std::move(data)} {}
    [[nodiscard]] size_t length() const noexcept override { return _data.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    void read(luisa::span<std::byte> dst) noexcept override;
};

class LC_CORE_API BinaryFileStream : public BinaryStream {

private:
    ::FILE *_file{nullptr};
    size_t _length{0u};
    size_t _pos{0u};

public:
    explicit BinaryFileStream(const luisa::string &path) noexcept;
    ~BinaryFileStream() noexcept override;
    BinaryFileStream(BinaryFileStream &&another) noexcept;
    BinaryFileStream &operator=(BinaryFileStream &&rhs) noexcept;
    BinaryFileStream(const BinaryFileStream &) noexcept = delete;
    BinaryFileStream &operator=(const BinaryFileStream &) noexcept = delete;
    [[nodiscard]] auto valid() const noexcept { return _file != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    [[nodiscard]] size_t length() const noexcept override { return _length; }
    [[nodiscard]] size_t pos() const noexcept override { return _pos; }
    void read(luisa::span<std::byte> dst) noexcept override;
    void close() noexcept;
};

class BinaryIO {

public:
    virtual ~BinaryIO() = default;
    virtual luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) noexcept = 0;
    virtual luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) noexcept = 0;
    virtual luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) noexcept = 0;
    virtual void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
};

class BinaryFileIO : public BinaryIO {



};

}// namespace luisa::compute
