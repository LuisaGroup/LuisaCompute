#pragma once
#ifdef LUISA_PLATFORM_WINDOWS
#define LUISA_USE_DIRECT_STORAGE
#endif

#include <core/binary_io.h>
#include <core/stl/filesystem.h>
#include <vstl/common.h>
#include <shared_mutex>
#include <runtime/context.h>
#include <core/dynamic_module.h>
namespace luisa::compute {
class DefaultBinaryIO final : public BinaryIO {
public:
    friend class LockedBinaryFileStream;
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
#ifdef LUISA_USE_DIRECT_STORAGE
    DynamicModule dstorage_lib;
    void *dstorage_impl;
    BinaryStream *(*create_dstorage_stream)(void *impl, luisa::string_view path);
#endif

private:
    luisa::unique_ptr<BinaryStream> _read(luisa::string const &file_path) const noexcept;
    void _write(luisa::string const &file_path, luisa::span<std::byte const> data) const noexcept;
    MapIndex _lock(luisa::string const &name, bool is_write) const noexcept;
    void _unlock(MapIndex const &idx, bool is_write) const noexcept;

public:
    DefaultBinaryIO(Context &&ctx, void* ext = nullptr) noexcept;
    ~DefaultBinaryIO() noexcept;
    luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept override;
    void write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    void write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
};

}// namespace luisa::compute
