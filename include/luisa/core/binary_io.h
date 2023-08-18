#pragma once

#include <cstdio>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/filesystem.h>

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
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept = 0;
    // returns the path of the written file (if stored on disk, otherwise returns empty path)
    [[nodiscard]] virtual luisa::filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
    [[nodiscard]] virtual luisa::filesystem::path write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
    [[nodiscard]] virtual luisa::filesystem::path write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
};

}// namespace luisa
