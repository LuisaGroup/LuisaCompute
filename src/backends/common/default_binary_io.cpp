#include <backends/common/default_binary_io.h>
#include <runtime/context.h>
#include <runtime/context_paths.h>
#include <core/stl/filesystem.h>

namespace luisa::compute {
LockedBinaryFileStream::LockedBinaryFileStream(DefaultBinaryIO const *binary_io, const luisa::string &path) noexcept
    : _stream{path},
      _binary_io{binary_io},
      _idx{binary_io->_lock(path, false)} {
}
LockedBinaryFileStream::~LockedBinaryFileStream() noexcept {
    _binary_io->_unlock(_idx, false);
}
luisa::unique_ptr<BinaryStream> DefaultBinaryIO::_read(luisa::string const &file_path) const noexcept {
    return luisa::make_unique<LockedBinaryFileStream>(this, file_path);
}
DefaultBinaryIO::MapIndex DefaultBinaryIO::_lock(luisa::string const &name, bool is_write) const noexcept {
    MapIndex iter;
    FileMutex *ptr;
    {
        std::lock_guard lck{_global_mtx};
        iter = _mutex_map.emplace(name);
        ptr = &iter.value();
        ptr->ref_count++;
    }
    if (is_write) {
        ptr->mtx.lock();
    } else {
        ptr->mtx.lock_shared();
    }
    return iter;
}
void DefaultBinaryIO::_unlock(MapIndex const &idx, bool is_write) const noexcept {
    auto &v = idx.value();
    if (is_write) {
        v.mtx.unlock();
    } else {
        v.mtx.unlock_shared();
    }
    {
        std::lock_guard lck{_global_mtx};
        if (--v.ref_count == 0) {
            _mutex_map.remove(idx);
        }
    }
}
void DefaultBinaryIO::_write(luisa::string const &file_path, luisa::span<std::byte const> data) const noexcept {
    auto idx = _lock(file_path, true);
    auto disposer = vstd::scope_exit([&]() { _unlock(idx, true); });
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

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_bytecode(luisa::string_view name) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        return _read(luisa::to_string(name));
    }
    auto file_path = luisa::to_string(_ctx.paths().runtime_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_cache(luisa::string_view name) const noexcept {
    auto file_path = luisa::to_string(_ctx.paths().cache_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_internal_shader(luisa::string_view name) const noexcept {
    auto file_path = luisa::to_string(_data_path / name);
    return _read(file_path);
}

void DefaultBinaryIO::write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        _write(luisa::to_string(name), data);
        return;
    }
    auto file_path = luisa::to_string(_ctx.paths().runtime_directory() / name);
    _write(file_path, data);
}

void DefaultBinaryIO::write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    auto file_path = luisa::to_string(_ctx.paths().cache_directory() / name);
    _write(file_path, data);
}

void DefaultBinaryIO::write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    auto file_path = luisa::to_string(_data_path / name);
    _write(file_path, data);
}
}// namespace luisa::compute
