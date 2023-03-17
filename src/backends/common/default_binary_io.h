#pragma once

#include <core/binary_io.h>
#include <core/stl/filesystem.h>
#include <vstl/common.h>
#include <shared_mutex>
#include <core/binary_file_stream.h>

namespace luisa::compute {

class Context;

class DefaultBinaryIO final : public BinaryIO {

public:
    friend class LockedBinaryFileStream;
    struct FileMutex {
        std::shared_mutex mtx;
        size_t ref_count{0};
    };
    using MutexMap = vstd::HashMap<luisa::string, FileMutex>;
    using MapIndex = MutexMap::Index;

private:
    Context &_ctx;
    std::filesystem::path _data_path;
    mutable std::mutex _global_mtx;
    mutable MutexMap _mutex_map;

private:
    luisa::unique_ptr<BinaryStream> _read(luisa::string const &file_path) const noexcept;
    void _write(luisa::string const &file_path, luisa::span<std::byte const> data) const noexcept;
    MapIndex _lock(luisa::string const &name, bool is_write) const noexcept;
    void _unlock(MapIndex const &idx, bool is_write) const noexcept;

public:
    DefaultBinaryIO(Context &ctx, luisa::string_view backend_name) noexcept;
    luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept override;
    void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
};

class LockedBinaryFileStream : public BinaryStream {

private:
    BinaryFileStream _stream;
    DefaultBinaryIO const* _binary_io;
    DefaultBinaryIO::MapIndex _idx;

public:
    explicit LockedBinaryFileStream(DefaultBinaryIO const* binary_io, const luisa::string &path) noexcept;
    ~LockedBinaryFileStream() noexcept override;
    [[nodiscard]] size_t length() const noexcept override { return _stream.length(); }
    [[nodiscard]] size_t pos() const noexcept override { return _stream.pos(); }
    void read(luisa::span<std::byte> dst) noexcept override {
        _stream.read(dst);
    }
};

}// namespace luisa::compute
