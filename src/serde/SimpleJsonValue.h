#pragma once
#include <serde/IJsonObject.h>
#include <serde/SimpleJsonLoader.h>
#include <vstl/variant_util.h>
namespace toolhub::db {
class ConcurrentBinaryJson;
class SimpleBinaryJson;
struct SimpleJsonKey {
    using ValueType = eastl::variant<int64,
                                     vstd::string,
                                     vstd::Guid>;
    using VarType = typename vstd::VariantVisitor<ValueType>::Type;

    ValueType value;
    SimpleJsonKey(ValueType const &value)
        : value(value) {}
    SimpleJsonKey(ValueType &&value)
        : value(std::move(value)) {}
    SimpleJsonKey(Key const &v) {
        if (v.index() < eastl::variant_size_v<ValueType>) {
            switch (v.index()) {
                case VarType::IndexOf<int64>:
                    value = int64(eastl::get<int64>(v));
                    break;
                case VarType::IndexOf<vstd::string>:
                    value = vstd::string(eastl::get<std::string_view>(v));
                    break;
                case VarType::IndexOf<vstd::Guid>:
                    value = vstd::Guid(eastl::get<vstd::Guid>(v));
                    break;
            }
        }
    }
    SimpleJsonKey(Key &&v)
        : SimpleJsonKey(v) {}
    static Key GetKey(ValueType const &value) {
        switch (value.index()) {
            case VarType::IndexOf<int64>:
                return Key(eastl::get<int64>(value));
            case VarType::IndexOf<vstd::Guid>:
                return Key(eastl::get<vstd::Guid>(value));
            case VarType::IndexOf<vstd::string>:
                return Key(eastl::get<vstd::string>(value));
            default:
                return Key();
        }
    }
    Key GetKey() const {
        return GetKey(value);
    }
    int32 compare(Key const &key) const {
        if (key.index() == value.index()) {
            switch (value.index()) {
                case VarType::IndexOf<int64>: {
                    static const vstd::compare<int64> cm;
                    return cm(eastl::get<int64>(value), eastl::get<int64>(key));
                }
                case VarType::IndexOf<vstd::Guid>: {
                    static const vstd::compare<vstd::Guid> cm;
                    return cm(eastl::get<vstd::Guid>(value), eastl::get<vstd::Guid>(key));
                }
                case VarType::IndexOf<vstd::string>: {
                    static const vstd::compare<std::string_view> cm;
                    return cm(eastl::get<vstd::string>(value), eastl::get<std::string_view>(key));
                }
            }
            return 0;
        } else
            return (key.index() > value.index()) ? 1 : -1;
    }
    int32 compare(ValueType const &key) const {
        if (key.index() == value.index()) {
            switch (value.index()) {
                case VarType::IndexOf<int64>: {
                    static const vstd::compare<int64> cm;
                    return cm(eastl::get<int64>(value), eastl::get<int64>(key));
                }
                case VarType::IndexOf<vstd::Guid>: {
                    static const vstd::compare<vstd::Guid> cm;
                    return cm(eastl::get<vstd::Guid>(value), eastl::get<vstd::Guid>(key));
                }
                case VarType::IndexOf<vstd::string>: {
                    static const vstd::compare<std::string_view> cm;
                    return cm(eastl::get<vstd::string>(value), eastl::get<vstd::string>(key));
                }
                default:
                    return 0;
            }
        } else
            return (key.index() > value.index()) ? 1 : -1;
    }
    size_t GetHashCode() const {
        size_t hash;
        eastl::visit(
            [&](auto &&v) {
                hash = vstd::hash<std::remove_cvref_t<decltype(v)>>()(v);
            },
            value);
        return hash;
    }
};
struct SimpleJsonKeyHash {
    template<typename T>
    size_t operator()(T const &key) const {
        if constexpr (std::is_same_v<std::remove_cvref_t<T>, SimpleJsonKey>)
            return key.GetHashCode();
        else {
            return vstd::hash<T>()(key);
        }
    }
};
struct SimpleJsonKeyEqual {
    int32 operator()(SimpleJsonKey const &key, SimpleJsonKey const &v) const {
        return key.compare(v.value);
    }
    int32 operator()(SimpleJsonKey const &key, SimpleJsonKey::ValueType const &v) const {
        return key.compare(v);
    }

    int32 operator()(SimpleJsonKey const &key, Key const &t) const {
        return key.compare(t);
    }
    int32 operator()(SimpleJsonKey const &key, int64 const &t) const {
        if (key.value.index() == 0) {
            vstd::compare<int64> c;
            return c(eastl::get<0>(key.value), t);
        } else {
            return -1;
        }
    }
    int32 operator()(SimpleJsonKey const &key, std::string_view const &t) const {
        if (key.value.index() == 1) {
            vstd::compare<std::string_view> c;
            return c(eastl::get<1>(key.value), t);
        } else {
            return (key.value.index() < 1) ? 1 : -1;
        }
    }
    int32 operator()(SimpleJsonKey const &key, vstd::Guid const &t) const {
        if (key.value.index() == 2) {
            vstd::compare<vstd::Guid> c;
            return c(eastl::get<2>(key.value), t);
        } else {
            return 1;
        }
    }
};
using KVMap = vstd::HashMap<SimpleJsonKey, SimpleJsonVariant, SimpleJsonKeyHash, SimpleJsonKeyEqual>;
class SimpleJsonValue {
protected:
    SimpleBinaryJson *db;
};

class SimpleJsonValueDict final : public IJsonDict, public SimpleJsonValue {
public:
    void Dispose() override;
    SimpleBinaryJson *MGetDB() const { return db; }
    IJsonDatabase *GetDB() const override;
    KVMap vars;
    SimpleJsonValueDict(SimpleBinaryJson *db);
    bool Contains(Key const &key) const override;
    //void Add(Key const& key, WriteJsonVariant&& value) override;

    ~SimpleJsonValueDict();
    /* SimpleJsonValueDict(
		SimpleBinaryJson* db,
		IJsonDict* src);*/
    ReadJsonVariant Get(Key const &key) const override;
    vstd::vector<ReadJsonVariant> Get(vstd::span<Key> keys) const override;
    vstd::vector<bool> Contains(vstd::span<Key> keys) const override;
    void Set(vstd::span<std::pair<Key, WriteJsonVariant>> kv) override;
    void Remove(vstd::span<Key> keys) override;
    void Set(Key const &key, WriteJsonVariant &&value) override;
    ReadJsonVariant TrySet(Key const &key, vstd::function<WriteJsonVariant()> const &value) override;
    void TryReplace(Key const &key, vstd::function<WriteJsonVariant(ReadJsonVariant const &)> const &value) override;
    vstd::vector<std::pair<Key, ReadJsonVariant>> ToVector() const override;
    void Remove(Key const &key) override;
    eastl::optional<WriteJsonVariant> GetAndRemove(Key const &key) override;
    eastl::optional<WriteJsonVariant> GetAndSet(Key const &key, WriteJsonVariant &&newValue) override;
    vstd::Iterator<JsonKeyPair> begin() const & override;
    vstd::Iterator<MoveJsonKeyPair> begin() && override;
    size_t Length() const override;
    vstd::vector<uint8_t> Serialize() const override;
    void Serialize(vstd::vector<uint8_t> &vec) const override;
    void M_GetSerData(vstd::vector<uint8_t> &arr) const;
    void LoadFromSer(vstd::span<uint8_t const> &arr);
    void LoadFromSer_DiffEnding(vstd::span<uint8_t const> &arr);
    bool Read(vstd::span<uint8_t const> sp,
              bool clearLast) override;
    void Reset() override;
    void Reserve(size_t capacity) override;
    eastl::optional<ParsingException> Parse(
        std::string_view str, bool clearLast) override;
    bool IsEmpty() const override { return vars.size() == 0; }
    void M_Print(vstd::string &str) const;
    void M_Print_Compress(vstd::string &str) const;
    vstd::string FormattedPrint() const override {
        vstd::string str;
        M_Print(str);
        return str;
    }
    vstd::string Print() const override {
        vstd::string str;
        M_Print_Compress(str);
        return str;
    }
    vstd::MD5 GetMD5() const override;
};

class SimpleJsonValueArray final : public IJsonArray, public SimpleJsonValue {
public:
    void Dispose() override;
    SimpleBinaryJson *MGetDB() const { return db; }
    IJsonDatabase *GetDB() const override;
    vstd::vector<SimpleJsonVariant, VEngine_AllocType::VEngine, 4> arr;
    SimpleJsonValueArray(SimpleBinaryJson *db);
    ~SimpleJsonValueArray();
    /* SimpleJsonValueArray(
		SimpleBinaryJson* db,
		IJsonArray* src);*/
    size_t Length() const override;
    void Reserve(size_t capacity) override;
    eastl::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) override;
    vstd::vector<uint8_t> Serialize() const override;
    void Serialize(vstd::vector<uint8_t> &vec) const override;
    void M_GetSerData(vstd::vector<uint8_t> &result) const;
    void LoadFromSer(vstd::span<uint8_t const> &arr);
    void LoadFromSer_DiffEnding(vstd::span<uint8_t const> &arr);
    bool Read(vstd::span<uint8_t const> sp, bool clearLast) override;
    void Reset() override;
    vstd::vector<ReadJsonVariant> Get(vstd::span<size_t> indices) const override;
    void Set(vstd::span<std::pair<size_t, WriteJsonVariant>> values) override;
    void Remove(vstd::span<size_t> indices) override;
    void Add(vstd::span<WriteJsonVariant> values) override;
    eastl::optional<WriteJsonVariant> GetAndRemove(size_t index) override;
    eastl::optional<WriteJsonVariant> GetAndSet(size_t index, WriteJsonVariant &&newValue) override;
    vstd::vector<ReadJsonVariant> ToVector() const override;
    void Set(size_t index, WriteJsonVariant &&value) override;
    ReadJsonVariant Get(size_t index) const override;
    void Remove(size_t index) override;
    void Add(WriteJsonVariant &&value) override;
    vstd::Iterator<ReadJsonVariant> begin() const & override;
    vstd::Iterator<WriteJsonVariant> begin() && override;
    bool IsEmpty() const override { return arr.size() == 0; }
    void M_Print(vstd::string &str) const;
    vstd::string FormattedPrint() const override {
        vstd::string str;
        M_Print(str);
        return str;
    }
    void M_Print_Compress(vstd::string &str) const;
    vstd::string Print() const override {
        vstd::string str;
        M_Print_Compress(str);
        return str;
    }
    vstd::MD5 GetMD5() const override;
};
}// namespace toolhub::db