#pragma once
#include <serde/IJsonObject.h>
#include <serde/SimpleJsonLoader.h>
namespace toolhub::db {
class ConcurrentBinaryJson;
class SimpleBinaryJson;
struct SimpleJsonKey {
    using ValueType = vstd::variant<int64,
                                    vstd::string,
                                    vstd::Guid>;

    ValueType value;
    SimpleJsonKey(ValueType const &value)
        : value(value) {}
    SimpleJsonKey(ValueType &&value)
        : value(std::move(value)) {}
    SimpleJsonKey(Key const &v) {
        if (v.GetType() < ValueType::argSize) {
            value.update(v.GetType(), [&](void *ptr) {
                switch (v.GetType()) {
                    case ValueType::IndexOf<int64>:
                        new (ptr) int64(v.template force_get<int64>());
                        break;
                    case ValueType::IndexOf<vstd::string>:
                        new (ptr) vstd::string(v.template force_get<std::string_view>());
                        break;
                    case ValueType::IndexOf<vstd::Guid>:
                        new (ptr) vstd::Guid(v.template force_get<vstd::Guid>());
                        break;
                }
            });
        }
    }
    SimpleJsonKey(Key &&v)
        : SimpleJsonKey(v) {}
    static Key GetKey(ValueType const &value) {
        switch (value.GetType()) {
            case ValueType::IndexOf<int64>:
                return Key(value.template force_get<int64>());
            case ValueType::IndexOf<vstd::Guid>:
                return Key(value.template force_get<vstd::Guid>());
            case ValueType::IndexOf<vstd::string>:
                return Key(value.template force_get<vstd::string>());
            default:
                return Key();
        }
    }
    Key GetKey() const {
        return GetKey(value);
    }
    int32 compare(Key const &key) const {
        if (key.GetType() == value.GetType()) {
            switch (value.GetType()) {
                case ValueType::IndexOf<int64>: {
                    static const vstd::compare<int64> cm;
                    return cm(value.template force_get<int64>(), key.template force_get<int64>());
                }
                case ValueType::IndexOf<vstd::Guid>: {
                    static const vstd::compare<vstd::Guid> cm;
                    return cm(value.template force_get<vstd::Guid>(), key.template force_get<vstd::Guid>());
                }
                case ValueType::IndexOf<vstd::string>: {
                    static const vstd::compare<std::string_view> cm;
                    return cm(value.template force_get<vstd::string>(), key.template force_get<std::string_view>());
                }
            }
            return 0;
        } else
            return (key.GetType() > value.GetType()) ? 1 : -1;
    }
    int32 compare(ValueType const &key) const {
        if (key.GetType() == value.GetType()) {
            switch (value.GetType()) {
                case ValueType::IndexOf<int64>: {
                    static const vstd::compare<int64> cm;
                    return cm(value.template force_get<int64>(), key.template force_get<int64>());
                }
                case ValueType::IndexOf<vstd::Guid>: {
                    static const vstd::compare<vstd::Guid> cm;
                    return cm(value.template force_get<vstd::Guid>(), key.template force_get<vstd::Guid>());
                }
                case ValueType::IndexOf<vstd::string>: {
                    static const vstd::compare<std::string_view> cm;
                    return cm(value.template force_get<vstd::string>(), key.template force_get<vstd::string>());
                }
                default:
                    return 0;
            }
        } else
            return (key.GetType() > value.GetType()) ? 1 : -1;
    }
    size_t GetHashCode() const {
        auto getHash = [](auto &&v) {
            vstd::hash<std::remove_cvref_t<decltype(v)>> h;
            return h(v);
        };
        switch (value.GetType()) {
            case ValueType::IndexOf<int64>:
                return getHash(*reinterpret_cast<int64 const *>(value.GetPlaceHolder()));

            case ValueType::IndexOf<vstd::Guid>:
                return getHash(*reinterpret_cast<vstd::Guid const *>(value.GetPlaceHolder()));

            case ValueType::IndexOf<vstd::string>:
                return getHash(*reinterpret_cast<vstd::string const *>(value.GetPlaceHolder()));
            default:
                return 0;
        }
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
            return c(key.value.get<0>(), t);
        } else {
            return -1;
        }
    }
    int32 operator()(SimpleJsonKey const &key, std::string_view const &t) const {
        if (key.value.index() == 1) {
            vstd::compare<std::string_view> c;
            return c(key.value.get<1>(), t);
        } else {
            return (key.value.index() < 1) ? 1 : -1;
        }
    }
    int32 operator()(SimpleJsonKey const &key, vstd::Guid const &t) const {
        if (key.value.index() == 2) {
            vstd::compare<vstd::Guid> c;
            return c(key.value.get<2>(), t);
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
    vstd::vector<ReadJsonVariant> Get(std::span<Key> keys) const override;
    vstd::vector<bool> Contains(std::span<Key> keys) const override;
    void Set(std::span<std::pair<Key, WriteJsonVariant>> kv) override;
    void Remove(std::span<Key> keys) override;
    vstd::string PrintYaml() const override;
    void PrintYaml(vstd::string &str, SimpleJsonKey::ValueType const &key, size_t space) const;
    //Single line dict
    void PrintYaml(vstd::string &str) const;
    void PrintYaml(vstd::string &str, size_t space) const;
    void Set(Key const &key, WriteJsonVariant &&value) override;
    ReadJsonVariant TrySet(Key const &key, vstd::function<WriteJsonVariant()> const &value) override;
    void TryReplace(Key const &key, vstd::function<WriteJsonVariant(ReadJsonVariant const &)> const &value) override;
    vstd::vector<std::pair<Key, ReadJsonVariant>> ToVector() const override;
    void Remove(Key const &key) override;
    vstd::optional<WriteJsonVariant> GetAndRemove(Key const &key) override;
    vstd::optional<WriteJsonVariant> GetAndSet(Key const &key, WriteJsonVariant &&newValue) override;
    vstd::Iterator<JsonKeyPair> begin() const & override;
    vstd::Iterator<MoveJsonKeyPair> begin() && override;
    size_t Length() const override;
    vstd::vector<uint8_t> Serialize() const override;
    void Serialize(vstd::vector<uint8_t> &vec) const override;
    void M_GetSerData(vstd::vector<uint8_t> &arr) const;
    void LoadFromSer(std::span<uint8_t const> &arr);
    void LoadFromSer_DiffEnding(std::span<uint8_t const> &arr);
    bool Read(std::span<uint8_t const> sp,
              bool clearLast) override;
    void Reset() override;
    void Reserve(size_t capacity) override;
    vstd::optional<ParsingException> Parse(
        std::string_view str, bool clearLast) override;
    bool IsEmpty() const override { return vars.size() == 0; }
    void M_Print(vstd::string &str) const;
    void M_Print_Compress(vstd::string &str) const;
    vstd::optional<ParsingException> ParseYaml(
        std::string_view str,
        bool clearLast) override;
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
    vstd::vector<SimpleJsonVariant, VEngine_AllocType::VEngine, false, 4> arr;
    SimpleJsonValueArray(SimpleBinaryJson *db);
    ~SimpleJsonValueArray();
    /* SimpleJsonValueArray(
		SimpleBinaryJson* db,
		IJsonArray* src);*/
    size_t Length() const override;
    void Reserve(size_t capacity) override;
    vstd::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) override;
    vstd::vector<uint8_t> Serialize() const override;
    void Serialize(vstd::vector<uint8_t> &vec) const override;
    void M_GetSerData(vstd::vector<uint8_t> &result) const;
    void LoadFromSer(std::span<uint8_t const> &arr);
    void LoadFromSer_DiffEnding(std::span<uint8_t const> &arr);
    bool Read(std::span<uint8_t const> sp, bool clearLast) override;
    void Reset() override;
    vstd::vector<ReadJsonVariant> Get(std::span<size_t> indices) const override;
    void Set(std::span<std::pair<size_t, WriteJsonVariant>> values) override;
    void Remove(std::span<size_t> indices) override;
    void Add(std::span<WriteJsonVariant> values) override;
    vstd::optional<WriteJsonVariant> GetAndRemove(size_t index) override;
    vstd::optional<WriteJsonVariant> GetAndSet(size_t index, WriteJsonVariant &&newValue) override;
    vstd::vector<ReadJsonVariant> ToVector() const override;
    void Set(size_t index, WriteJsonVariant &&value) override;
    ReadJsonVariant Get(size_t index) const override;
    void Remove(size_t index) override;
    void Add(WriteJsonVariant &&value) override;
    vstd::Iterator<ReadJsonVariant> begin() const & override;
    vstd::Iterator<WriteJsonVariant> begin() && override;
    bool IsEmpty() const override { return arr.size() == 0; }
    void M_Print(vstd::string &str) const;
    void PrintYaml(vstd::string &str) const;
    void PrintYaml(vstd::string &str, size_t space) const;
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