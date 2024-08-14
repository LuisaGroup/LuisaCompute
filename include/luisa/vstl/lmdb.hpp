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

struct LMDBIteratorEndTag {};

struct LC_VSTL_API LMDBIterator {
    friend class LMDB;
    struct Value {
        luisa::span<const std::byte> key;
        luisa::span<const std::byte> value;
    };
    Value const &operator*() const noexcept {
        return _value;
    }
    void operator++() noexcept;
    void operator++(int) noexcept {
        return operator++();
    }
    bool operator==(LMDBIteratorEndTag) const noexcept {
        return _finished;
    }
    LMDBIterator(LMDBIterator const &) = delete;
    LMDBIterator(LMDBIterator &&rhs) noexcept;
    LMDBIterator &operator=(LMDBIterator const &) = delete;
    LMDBIterator &operator=(LMDBIterator &&rhs) noexcept {
        this->~LMDBIterator();
        new (std::launder(this)) LMDBIterator{std::move(rhs)};
        return *this;
    }
    ~LMDBIterator() noexcept;

private:
    MDB_txn *_txn{nullptr};
    MDB_cursor *_cursor{nullptr};
    Value _value{};
    bool _finished{false};
    LMDBIterator(MDB_env *env, uint32_t dbi) noexcept;
};

class LC_VSTL_API LMDB {
    luisa::string _path;
    size_t _map_size{};
    MDB_env *_env{nullptr};
    luisa::optional<uint32_t> _dbi{};
    uint32_t _flag{};
    void _dispose() noexcept;

public:
    LMDB(
        std::filesystem::path const &db_dir,
        size_t max_reader = 126ull,
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
    [[nodiscard]] luisa::span<const std::byte> read(luisa::span<const std::byte> key) const noexcept;
    [[nodiscard]] luisa::span<const std::byte> read(luisa::string_view key) const noexcept {
        return read(luisa::span{reinterpret_cast<std::byte const *>(key.data()), key.size()});
    }
    void write(luisa::span<const std::byte> key, luisa::span<const std::byte> value) const noexcept;
    void write(luisa::string_view key, luisa::span<const std::byte> value) const noexcept {
        write(luisa::span{reinterpret_cast<std::byte const *>(key.data()), key.size()}, value);
    }
    void write_all(luisa::vector<LMDBWriteCommand> &&commands) const noexcept;
    void remove(luisa::span<const std::byte> key) const noexcept;
    void remove_all(luisa::vector<luisa::vector<std::byte>> &&keys) const noexcept;
    void copy_to(std::filesystem::path path) const noexcept;
    ~LMDB() noexcept;
    [[nodiscard]] LMDBIterator begin() const noexcept;
    [[nodiscard]] LMDBIteratorEndTag end() const noexcept { return {}; }
};
};// namespace vstd