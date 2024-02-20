#pragma once
#include <luisa/core/dll_export.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/filesystem.h>
#include <shared_mutex>
struct MDB_env;
struct MDB_txn;
struct MDB_cursor;
struct MDB_val;
struct MDB_stat;
struct MDB_envinfo;
namespace vstd {
struct LMDBWriteCommand {
    luisa::vector<std::byte> key;
    luisa::vector<std::byte> value;
};
class LC_VSTL_API LMDB {
    luisa::string _path;
    size_t _map_size;
    MDB_env *_env{nullptr};
    luisa::optional<uint32_t> _dbi{};
    uint32_t _flag;
    void _dispose() noexcept;

public:
    LMDB(
        std::filesystem::path db_dir,
        // 64G as default (should be enough for shader?)
        size_t map_size = 1024ull * 1024ull * 1024ull * 64ull) noexcept;
    LMDB(LMDB const &) = delete;
    LMDB &operator=(LMDB const &) = delete;
    LMDB(LMDB &&rhs) noexcept;
    LMDB &operator=(LMDB &&rhs) noexcept {
        this->~LMDB();
        new (std::launder(this)) LMDB{std::move(rhs)};
        return *this;
    }
    luisa::span<const std::byte> read(luisa::span<const std::byte> key) noexcept;
    void write(luisa::span<const std::byte> key, luisa::span<std::byte> value) noexcept;
    void write_all(luisa::vector<LMDBWriteCommand> &&commands) noexcept;
    void remove(luisa::span<const std::byte> key) noexcept;
    void remove_all(luisa::vector<luisa::vector<std::byte>> &&keys) noexcept;
    void copy_to(std::filesystem::path path) noexcept;
    ~LMDB() noexcept;
};
};// namespace vstd