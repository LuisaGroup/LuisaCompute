#pragma once
#include <vstl/VGuid.h>
#include <serde/IJsonDatabase.h>
#include <serde/ParserException.h>
#include <EASTL/variant.h>
#include <EASTL/optional.h>
namespace toolhub::db {
class IJsonDict;
class IJsonArray;

using ReadJsonVariant = vstd::variant<int64,
                                      double,
                                      std::string_view,
                                      IJsonDict *,
                                      IJsonArray *,
                                      vstd::Guid,
                                      bool,
                                      std::nullptr_t>;
using WriteJsonVariant = vstd::variant<int64,
                                       double,
                                       vstd::string,
                                       vstd::unique_ptr<IJsonDict>,
                                       vstd::unique_ptr<IJsonArray>,
                                       vstd::Guid,
                                       bool>;
using Key = vstd::variant<int64,
                          std::string_view,
                          vstd::Guid>;
struct JsonKeyPair {
    Key key;
    ReadJsonVariant value;
    JsonKeyPair(
        Key &&key,
        ReadJsonVariant &&value) : key(std::move(key)), value(std::move(value)) {}
};
struct MoveJsonKeyPair {
    Key key;
    WriteJsonVariant value;
    MoveJsonKeyPair(
        Key &&key,
        WriteJsonVariant &&value) : key(std::move(key)), value(std::move(value)) {}
};
class IJsonObject : public vstd::IDisposable {
protected:
    virtual ~IJsonObject() = default;

public:
    ////////// Basic
    virtual bool IsDict() const { return false; }
    virtual bool IsArray() const { return false; }
    virtual IJsonDatabase *GetDB() const = 0;
    virtual size_t Length() const = 0;
    virtual vstd::vector<uint8_t> Serialize() const = 0;
    virtual void Serialize(vstd::vector<uint8_t> &vec) const = 0;
    virtual void Reset() = 0;
    virtual bool IsEmpty() const = 0;
    virtual vstd::string FormattedPrint() const = 0;
    virtual vstd::string Print() const = 0;
    virtual bool Read(std::span<uint8_t const> sp,
                      bool clearLast) = 0;
    virtual void Reserve(size_t capacity) = 0;
    virtual vstd::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) = 0;
    virtual vstd::MD5 GetMD5() const = 0;
    vstd::IteEndTag end() const { return vstd::IteEndTag(); }
};

class IJsonDict : public IJsonObject {

protected:
    ~IJsonDict() = default;

public:
    bool IsDict() const override { return true; }
    virtual ReadJsonVariant Get(Key const &key) const = 0;
    virtual vstd::vector<ReadJsonVariant> Get(std::span<Key> keys) const = 0;
    virtual bool Contains(Key const &key) const = 0;
    virtual vstd::vector<bool> Contains(std::span<Key> keys) const = 0;
    virtual void Set(Key const &key, WriteJsonVariant &&value) = 0;
    virtual void Set(std::span<std::pair<Key, WriteJsonVariant>> kv) = 0;
    virtual ReadJsonVariant TrySet(Key const &key, vstd::function<WriteJsonVariant()> const &value) = 0;
    virtual void TryReplace(Key const &key, vstd::function<WriteJsonVariant(ReadJsonVariant const &)> const &value) = 0;
    virtual void Remove(Key const &key) = 0;
    virtual vstd::optional<WriteJsonVariant> GetAndRemove(Key const &key) = 0;
    virtual vstd::optional<WriteJsonVariant> GetAndSet(Key const &key, WriteJsonVariant &&newValue) = 0;
    virtual void Remove(std::span<Key> keys) = 0;
    virtual vstd::vector<std::pair<Key, ReadJsonVariant>> ToVector() const = 0;
    virtual vstd::Iterator<JsonKeyPair> begin() const & = 0;
    virtual vstd::Iterator<MoveJsonKeyPair> begin() && = 0;
    virtual vstd::optional<ParsingException> ParseYaml(
        std::string_view str,
        bool clearLast) = 0;
    virtual vstd::string PrintYaml() const = 0;

    IJsonDict &operator<<(std::pair<Key, WriteJsonVariant> &&value) {
        Set(value.first, std::move(value.second));
        return *this;
    }
    ReadJsonVariant operator[](Key const &key) const {
        return Get(key);
    }
};

class IJsonArray : public IJsonObject {

protected:
    ~IJsonArray() = default;

public:
    bool IsArray() const override { return true; }
    virtual ReadJsonVariant Get(size_t index) const = 0;
    virtual vstd::vector<ReadJsonVariant> Get(std::span<size_t> indices) const = 0;
    virtual void Set(size_t index, WriteJsonVariant &&value) = 0;
    virtual void Set(std::span<std::pair<size_t, WriteJsonVariant>> values) = 0;
    virtual void Remove(size_t index) = 0;
    virtual void Remove(std::span<size_t> indices) = 0;
    virtual vstd::optional<WriteJsonVariant> GetAndRemove(size_t index) = 0;
    virtual vstd::optional<WriteJsonVariant> GetAndSet(size_t key, WriteJsonVariant &&newValue) = 0;
    virtual void Add(WriteJsonVariant &&value) = 0;
    virtual void Add(std::span<WriteJsonVariant> values) = 0;
    virtual vstd::vector<ReadJsonVariant> ToVector() const = 0;
    virtual vstd::Iterator<ReadJsonVariant> begin() const & = 0;
    virtual vstd::Iterator<WriteJsonVariant> begin() && = 0;
    IJsonArray &operator<<(WriteJsonVariant &&value) {
        Add(std::move(value));
        return *this;
    }
    ReadJsonVariant operator[](size_t index) const {
        return Get(index);
    }
};
}// namespace toolhub::db