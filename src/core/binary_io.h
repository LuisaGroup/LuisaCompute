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

class BinaryIO {

public:
    virtual ~BinaryIO() noexcept = default;
    virtual luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) noexcept = 0;
    virtual luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) noexcept = 0;
    virtual luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) noexcept = 0;
    virtual void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
};
}// namespace luisa::compute
