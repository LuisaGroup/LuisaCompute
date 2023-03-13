#include <backends/common/default_binary_io.h>
#include <core/binary_file_stream.h>
#include <runtime/context.h>
#include <runtime/context_paths.h>
#include <core/stl/filesystem.h>

namespace luisa::compute {

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::_read(luisa::string const &file_path) noexcept {
    return luisa::make_unique<BinaryFileStream>(file_path);
}

void DefaultBinaryIO::_write(luisa::string const &file_path, luisa::span<std::byte const> data) noexcept {
    auto f = fopen(file_path.c_str(), "wb");
    if (f) [[likely]] {
#ifdef _WIN32
#define LUISA_FWRITE _fwrite_nolock
#define LUISA_FCLOSE _fclose_nolock
#else
#define LUISA_FWRITE fwrite
#define LUISA_FCLOSE fclose
#endif
        LUISA_FWRITE(data.data(), data.size(), 1, f);
        LUISA_FCLOSE(f);
#undef LUISA_FWRITE
#undef LUISA_FCLOSE
    }
}

DefaultBinaryIO::DefaultBinaryIO(Context &ctx, luisa::string_view backend_name) noexcept : _ctx(ctx) {
    using namespace std::string_view_literals;
    luisa::string dir_name{backend_name};
    dir_name += "_builtin"sv;
    _data_path = _ctx.paths().data_directory() / dir_name;
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_bytecode(luisa::string_view name) noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        return _read(luisa::to_string(name));
    }
    auto file_path = luisa::to_string(_ctx.paths().runtime_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_cache(luisa::string_view name) noexcept {
    auto file_path = luisa::to_string(_ctx.paths().cache_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_internal_shader(luisa::string_view name) noexcept {
    auto file_path = luisa::to_string(_data_path / name);
    return _read(file_path);
}

void DefaultBinaryIO::write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        _write(luisa::to_string(name), data);
        return;
    }
    auto file_path = luisa::to_string(_ctx.paths().runtime_directory() / name);
    _write(file_path, data);
}

void DefaultBinaryIO::write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) noexcept {
    auto file_path = luisa::to_string(_ctx.paths().cache_directory() / name);
    _write(file_path, data);
}

void DefaultBinaryIO::write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) noexcept {
    auto file_path = luisa::to_string(_data_path / name);
    _write(file_path, data);
}

}// namespace luisa::compute
