#include <luisa/vstl/lmdb.hpp>
#include <lmdb.h>
#include <luisa/core/logging.h>

namespace vstd {

#define LUISA_CHECK_LMDB_ERROR(call)                            \
    do {                                                        \
        if (auto rc = (call); rc != MDB_SUCCESS) [[unlikely]] { \
            LUISA_ERROR_WITH_LOCATION(                          \
                "MDB error in call '{}': {}",                   \
                #call, mdb_strerror(rc));                       \
        }                                                       \
    } while (false)

LMDB::LMDB(
    std::filesystem::path const &db_dir,
    size_t max_reader,
    size_t map_size) noexcept
    : _path(luisa::to_string(db_dir)),
      _map_size(map_size) {
    LUISA_CHECK_LMDB_ERROR(mdb_env_create(&_env));
    LUISA_CHECK_LMDB_ERROR(mdb_env_set_maxreaders(_env, max_reader));
    LUISA_CHECK_LMDB_ERROR(mdb_env_set_mapsize(_env, _map_size));
    if (!std::filesystem::exists(db_dir)) {
        std::filesystem::create_directories(db_dir);
    }
    LUISA_CHECK_LMDB_ERROR(mdb_env_open(_env, _path.c_str(), MDB_NORDAHEAD, 0664));
    MDB_txn *txn;
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(_env, nullptr, MDB_RDONLY, &txn));
    _dbi = 0;
    LUISA_CHECK_LMDB_ERROR(mdb_dbi_open(txn, nullptr, 0, &*_dbi));
    mdb_txn_abort(txn);
}
luisa::span<const std::byte> LMDB::read(luisa::span<const std::byte> key) const noexcept {
    MDB_txn *txn;
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(_env, nullptr, MDB_RDONLY, &txn));
    MDB_val key_v{
        .mv_size = key.size_bytes(),
        .mv_data = const_cast<std::byte *>(key.data())};
    MDB_val value_v;
    uint32_t r = mdb_get(txn, *_dbi, &key_v, &value_v);
    mdb_txn_abort(txn);
    if (r == MDB_NOTFOUND) {
        return {};
    }
    return {static_cast<std::byte const *>(value_v.mv_data), value_v.mv_size};
}
void LMDB::write(luisa::span<const std::byte> key, luisa::span<const std::byte> value) const noexcept {
    MDB_txn *txn;
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(_env, nullptr, 0, &txn));
    MDB_val key_v{
        .mv_size = key.size_bytes(),
        .mv_data = const_cast<std::byte *>(key.data())};
    MDB_val value_v{
        .mv_size = value.size_bytes(),
        .mv_data = const_cast<std::byte *>(value.data())};
    mdb_put(txn, *_dbi, &key_v, &value_v, 0);
    LUISA_CHECK_LMDB_ERROR(mdb_txn_commit(txn));
}
void LMDB::write_all(luisa::vector<LMDBWriteCommand> &&commands) const noexcept {
    MDB_txn *txn;
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(_env, nullptr, 0, &txn));
    for (auto &i : commands) {
        MDB_val key_v{
            .mv_size = i.key.size_bytes(),
            .mv_data = const_cast<std::byte *>(i.key.data())};
        MDB_val value_v{
            .mv_size = i.value.size_bytes(),
            .mv_data = const_cast<std::byte *>(i.value.data())};
        mdb_put(txn, *_dbi, &key_v, &value_v, 0);
    }
    LUISA_CHECK_LMDB_ERROR(mdb_txn_commit(txn));
}
LMDB::LMDB(LMDB &&rhs) noexcept
    : _path(std::move(rhs._path)),
      _map_size(rhs._map_size),
      _env(rhs._env),
      _dbi(std::move(rhs._dbi)),
      _flag(rhs._flag) {
    rhs._env = nullptr;
}
void LMDB::_dispose() noexcept {
    if (_env) {
        if (_dbi) {
            mdb_dbi_close(_env, *_dbi);
            _dbi.reset();
        }
        mdb_env_close(_env);
        _env = nullptr;
    }
}
void LMDB::copy_to(std::filesystem::path path) const noexcept {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    } else {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    LUISA_CHECK_LMDB_ERROR(mdb_env_copy2(_env, luisa::to_string(path).c_str(), MDB_CP_COMPACT));
}
void LMDB::remove(luisa::span<const std::byte> key) const noexcept {
    MDB_txn *txn;
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(_env, nullptr, 0, &txn));
    MDB_val key_v{
        .mv_size = key.size_bytes(),
        .mv_data = const_cast<std::byte *>(key.data())};
    mdb_del(txn, *_dbi, &key_v, nullptr);
    LUISA_CHECK_LMDB_ERROR(mdb_txn_commit(txn));
}
void LMDB::remove_all(luisa::vector<luisa::vector<std::byte>> &&keys) const noexcept {
    MDB_txn *txn;
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(_env, nullptr, 0, &txn));
    for (auto &i : keys) {
        MDB_val key_v{
            .mv_size = i.size_bytes(),
            .mv_data = const_cast<std::byte *>(i.data())};
        mdb_del(txn, *_dbi, &key_v, nullptr);
    }
    LUISA_CHECK_LMDB_ERROR(mdb_txn_commit(txn));
}
LMDB::~LMDB() noexcept {
    _dispose();
}
LMDBIterator LMDB::begin() const noexcept {
    return LMDBIterator{_env, *_dbi};
}
LMDBIterator::LMDBIterator(LMDBIterator &&rhs) noexcept
    : _txn(rhs._txn),
      _cursor(rhs._cursor),
      _value(rhs._value),
      _finished(rhs._finished) {
    rhs._txn = nullptr;
    rhs._cursor = nullptr;
    rhs._finished = true;
}

LMDBIterator::~LMDBIterator() noexcept {
    if (_txn) {
        mdb_cursor_close(_cursor);
        mdb_txn_abort(_txn);
    }
}
LMDBIterator::LMDBIterator(MDB_env *env, uint32_t dbi) noexcept {
    LUISA_CHECK_LMDB_ERROR(mdb_txn_begin(env, nullptr, MDB_RDONLY, &_txn));
    LUISA_CHECK_LMDB_ERROR(mdb_cursor_open(_txn, dbi, &_cursor));
    operator++();
}
void LMDBIterator::operator++() noexcept {
    MDB_val key{};
    MDB_val data{};
    auto rc = mdb_cursor_get(_cursor, &key, &data, MDB_NEXT);
    _finished = rc != 0;
    if (_finished) {
        mdb_cursor_close(_cursor);
        mdb_txn_abort(_txn);
        _txn = nullptr;
    } else {
        _value.key = {
            reinterpret_cast<std::byte const *>(key.mv_data),
            key.mv_size};
        _value.value = {
            reinterpret_cast<std::byte const *>(data.mv_data),
            data.mv_size};
    }
}

#undef LUISA_CHECK_LMDB_ERROR

}// namespace vstd
