#include <new>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/logging.h>
#include <luisa/core/binary_file_stream.h>

#include "default_binary_io.h"

namespace luisa::compute {
class LMDBBinaryStream final : public BinaryStream {
    std::byte const *_begin;
    std::byte const *_ptr;
    std::byte const *_end;
public:
    LMDBBinaryStream(
        std::byte const *ptr,
        size_t size) noexcept : _begin(ptr), _ptr(ptr), _end(ptr + size) {}
    size_t length() const noexcept override {
        return _end - _ptr;
    }
    size_t pos() const noexcept override {
        return _ptr - _begin;
    }
    void read(luisa::span<std::byte> dst) noexcept override {
        std::memcpy(dst.data(), _ptr, dst.size());
        _ptr += dst.size();
    }
    BinaryBlob read(size_t expected_max_size) noexcept override {
        auto len = std::min(expected_max_size, length());
        BinaryBlob blob{const_cast<std::byte *>(_ptr), len, nullptr};
        _ptr += len;
        return blob;
    }
};

class LockedBinaryFileStream : public BinaryStream {

private:
    BinaryFileStream _stream;
    DefaultBinaryIO const *_binary_io;
    DefaultBinaryIO::MapIndex _idx;

public:
    explicit LockedBinaryFileStream(DefaultBinaryIO const *binary_io, ::FILE *file, size_t length, const luisa::string &path, DefaultBinaryIO::MapIndex &&idx) noexcept
        : _stream{file, length},
          _binary_io{binary_io},
          _idx{idx} {}
    ~LockedBinaryFileStream() noexcept override {
        _binary_io->_unlock(_idx, false);
    }
    [[nodiscard]] size_t length() const noexcept override { return _stream.length(); }
    [[nodiscard]] size_t pos() const noexcept override { return _stream.pos(); }
    void read(luisa::span<std::byte> dst) noexcept override {
        _stream.read(dst);
    }
};

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::_read(luisa::string const &file_path) const noexcept {
    auto idx = _lock(file_path, false);
    auto file = std::fopen(file_path.c_str(), "rb");
    if (file) {
        auto length = luisa::detail::get_c_file_length(file);
        if (length == 0) [[unlikely]] {
            _unlock(idx, false);
            return nullptr;
        }
        return luisa::make_unique<LockedBinaryFileStream>(this, file, length, file_path, std::move(idx));
    } else {
        _unlock(idx, false);
        LUISA_VERBOSE("Read file {} failed.", file_path);
        return nullptr;
    }
}

DefaultBinaryIO::MapIndex DefaultBinaryIO::_lock(luisa::string const &name, bool is_write) const noexcept {
    MapIndex iter;
    FileMutex *ptr;
    auto abs_path = luisa::filesystem::absolute(name).string();
    {
        std::lock_guard lck{_global_mtx};
        iter = _mutex_map.emplace(abs_path);
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
    std::lock_guard lck{_global_mtx};
    if ((--v.ref_count) == 0) {
        _mutex_map.remove(idx);
    }
}

void DefaultBinaryIO::_write(const luisa::string &file_path, luisa::span<std::byte const> data) const noexcept {
    auto folder = luisa::filesystem::path{file_path}.parent_path();
    std::error_code ec;
    luisa::filesystem::create_directories(folder, ec);
    if (ec) { LUISA_WARNING("Create directory {} failed.", folder.string()); }
    auto idx = _lock(file_path, true);
    if (auto f = fopen(file_path.c_str(), "wb")) [[likely]] {
        fwrite(data.data(), data.size(), 1, f);
        fclose(f);
    } else {
        LUISA_WARNING("Write file {} failed.", file_path);
    }
    _unlock(idx, true);
}

DefaultBinaryIO::DefaultBinaryIO(Context &&ctx, bool headless, bool use_lmdb) noexcept
    : _ctx(std::move(ctx)),
      _headless(headless),
      _use_lmdb{
#if defined(LUISA_PLATFORM_IOS)
          false
#else
          use_lmdb
#endif
      } {
#if defined(LUISA_PLATFORM_IOS)
    if (use_lmdb) {
        LUISA_WARNING_WITH_LOCATION(
            "LMDB is unavailable in the iOS application sandbox; "
            "using the file-backed shader cache instead.");
    }
#endif
    if (!headless) {
        _cache_dir = _ctx.create_data_subdir(".cache"sv);
        _data_dir = _ctx.create_data_subdir(".data"sv);
        if (_use_lmdb) {
            _data_lmdb.create(_data_dir, std::max<size_t>(126ull, std::thread::hardware_concurrency() * 2));
            _cache_lmdb.create(_cache_dir, std::max<size_t>(126ull, std::thread::hardware_concurrency() * 2));
        }
    }
}

DefaultBinaryIO::~DefaultBinaryIO() noexcept {
    if (!_headless) {
        if (_use_lmdb) {
            _data_lmdb.destroy();
            _cache_lmdb.destroy();
        }
    }
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_bytecode(luisa::string_view name) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        return _read(luisa::to_string(name));
    }
    auto file_path = luisa::to_string(_ctx.data_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_cache(luisa::string_view name) const noexcept {
    if (_use_lmdb) {
        auto r = _cache_lmdb->read(name);
        if (r.empty()) return {};
        return luisa::make_unique<LMDBBinaryStream>(r.data(), r.size());
    } else {
        std::filesystem::path local_path{name};
        if (local_path.is_absolute()) {
            return _read(luisa::to_string(name));
        }
        auto file_path = luisa::to_string(_cache_dir / name);
        return _read(file_path);
    }
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_internal_shader(luisa::string_view name) const noexcept {
    if (_use_lmdb) {
        auto r = _data_lmdb->read(name);
        if (r.empty()) return {};
        return luisa::make_unique<LMDBBinaryStream>(r.data(), r.size());
    } else {
        std::filesystem::path local_path{name};
        if (local_path.is_absolute()) {
            return _read(luisa::to_string(name));
        }
        auto file_path = luisa::to_string(_data_dir / name);
        return _read(file_path);
    }
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_source(luisa::string_view name) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) { return _read(luisa::to_string(name)); }
    return _read(luisa::to_string(_cache_dir / name));
}

luisa::filesystem::path DefaultBinaryIO::write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        _write(luisa::to_string(name), data);
        return local_path;
    }
    auto file_path = _ctx.data_directory() / name;
    _write(luisa::to_string(file_path), data);
    return file_path;
}

luisa::filesystem::path DefaultBinaryIO::write_shader_source(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        _write(luisa::to_string(name), data);
        return local_path;
    }
    auto file_path = _cache_dir / name;
    _write(luisa::to_string(file_path), data);
    return file_path;
}

luisa::filesystem::path DefaultBinaryIO::write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    if (_use_lmdb) {
        _cache_lmdb->write(name, data);
        return _cache_dir / name;
    } else {
        std::filesystem::path local_path{name};
        if (local_path.is_absolute()) {
            _write(luisa::to_string(name), data);
            return local_path;
        }
        auto file_path = _cache_dir / name;
        _write(luisa::to_string(file_path), data);
        return file_path;
    }
}

luisa::filesystem::path DefaultBinaryIO::write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    if (_use_lmdb) {
        _data_lmdb->write(name, data);
        return _data_dir / name;
    } else {
        std::filesystem::path local_path{name};
        if (local_path.is_absolute()) {
            _write(luisa::to_string(name), data);
            return local_path;
        }
        auto file_path = _data_dir / name;
        _write(luisa::to_string(file_path), data);
        return file_path;
    }
}

void DefaultBinaryIO::clear_shader_cache() const noexcept {
    std::lock_guard lck{_global_mtx};
    if (_use_lmdb) {
        _cache_lmdb.destroy();
    }
    std::error_code ec;
    for (auto &&dir : std::filesystem::directory_iterator(_cache_dir)) {
        std::filesystem::remove_all(dir, ec);
        if (ec) [[unlikely]] {
            LUISA_ERROR(
                "Failed to remove dir '{}': {}.",
                to_string(dir), ec.message());
        }
    }
    if (_use_lmdb) {
        _cache_lmdb.create(_cache_dir);
    }
}

}// namespace luisa::compute
