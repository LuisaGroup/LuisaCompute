#pragma once

#include <core/binary_io.h>
#include <core/stl/filesystem.h>

namespace luisa::compute {

class Context;

class DefaultBinaryIO final : public BinaryIO {
    Context &_ctx;
    std::filesystem::path _data_path;
    luisa::unique_ptr<BinaryStream> _read(luisa::string const &file_path) noexcept;
    void _write(luisa::string const &file_path, luisa::span<std::byte const> data) noexcept;

public:
    DefaultBinaryIO(Context &ctx, luisa::string_view backend_name) noexcept;
    luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) noexcept override;
    luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) noexcept override;
    void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept override;
    void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept override;
    void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept override;
};

}// namespace luisa::compute
