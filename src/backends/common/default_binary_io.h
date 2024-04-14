#pragma once

#include <shared_mutex>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/binary_io.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>
#include <luisa/runtime/context.h>
#include <luisa/vstl/lmdb.hpp>

namespace luisa::compute {

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
    Context _ctx;
    mutable luisa::spin_mutex _global_mtx;
    mutable MutexMap _mutex_map;
    std::filesystem::path _cache_dir;
    std::filesystem::path _data_dir;
    mutable vstd::LMDB _data_lmdb;
    mutable vstd::LMDB _cache_lmdb;

private:
    luisa::unique_ptr<BinaryStream> _read(luisa::string const &file_path) const noexcept;
    void _write(luisa::string const &file_path, luisa::span<std::byte const> data) const noexcept;
    MapIndex _lock(luisa::string const &name, bool is_write) const noexcept;
    void _unlock(MapIndex const &idx, bool is_write) const noexcept;

public:
    explicit DefaultBinaryIO(Context &&ctx, void *ext = nullptr) noexcept;
    ~DefaultBinaryIO() noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept override;
    luisa::filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    luisa::filesystem::path write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    luisa::filesystem::path write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_source(luisa::string_view name) const noexcept override;
    luisa::filesystem::path write_shader_source(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void clear_shader_cache() const noexcept override;
};

}// namespace luisa::compute

