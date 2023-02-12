#pragma once

#include <core/stl/memory.h>
#include <core/stl/string.h>

namespace luisa::compute {

class IBinaryStream {

public:
    [[nodiscard]] virtual size_t length() const = 0;
    [[nodiscard]] virtual size_t pos() const = 0;
    virtual void read(luisa::span<std::byte> dst) = 0;
    virtual ~IBinaryStream() = default;
};

class BinaryIO {

public:
    virtual ~BinaryIO() = default;
    virtual luisa::unique_ptr<IBinaryStream> read_shader_bytecode(luisa::string_view name) noexcept = 0;
    virtual luisa::unique_ptr<IBinaryStream> read_shader_cache(luisa::string_view name) noexcept = 0;
    virtual luisa::unique_ptr<IBinaryStream> read_internal_shader(luisa::string_view name) noexcept = 0;
    virtual void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
    virtual void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept = 0;
};

}// namespace luisa::compute
