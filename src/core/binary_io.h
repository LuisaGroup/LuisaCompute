#pragma once

#include <cstdio>
#include <core/stl/memory.h>
#include <core/stl/string.h>

namespace luisa {

class BinaryStream {

public:
    [[nodiscard]] virtual size_t length() const = 0;
    [[nodiscard]] virtual size_t pos() const = 0;
    virtual void read(luisa::span<std::byte> dst) = 0;
    virtual ~BinaryStream() noexcept = default;
};

class BinaryStringStream : public BinaryStream {

private:
    luisa::string _data;
    size_t _pos{0u};

public:
    explicit BinaryStringStream(luisa::string data) noexcept
        : _data{std::move(data)} {}
    [[nodiscard]] size_t length() const override { return _data.size(); }
    [[nodiscard]] size_t pos() const override { return _pos; }
    void read(luisa::span<std::byte> dst) override;
};

class BinaryFileStream : public BinaryStream {

private:
    ::FILE *_file{nullptr};

public:
    explicit BinaryFileStream(luisa::string_view path) noexcept;
    ~BinaryFileStream() noexcept override;
    [[nodiscard]] size_t length() const override;
    [[nodiscard]] size_t pos() const override;
    void read(luisa::span<std::byte> dst) override;
};

class BinaryIO {
public:
    virtual ~BinaryIO() noexcept = default;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) noexcept = 0;
    virtual void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
};

class BinaryFileIO : public BinaryIO {



};

}// namespace luisa::compute
