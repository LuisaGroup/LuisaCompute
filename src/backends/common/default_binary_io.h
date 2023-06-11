#pragma once

#include <shared_mutex>

#include <luisa/core/binary_io.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>
#include <luisa/runtime/context.h>
#include <luisa/core/dynamic_module.h>

namespace luisa::compute {

class DefaultBinaryIO final : public BinaryIO {

public:
    friend class LockedBinaryFileStream;
    friend class DStorageLockedFileStream;
    struct FileMutex {
        std::shared_mutex mtx;
        std::atomic_size_t ref_count{1};
    };
    using MutexMap = vstd::HashMap<luisa::string, FileMutex>;
    using MapIndex = MutexMap::Index;

private:
    Context _ctx;
    mutable std::mutex _global_mtx;
    mutable MutexMap _mutex_map;
    std::filesystem::path _cache_dir;
    std::filesystem::path _data_dir;

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
    void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
};

}// namespace luisa::compute

