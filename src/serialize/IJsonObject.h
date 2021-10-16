#pragma once
#include <vstl/VGuid.h>
#include <serialize/IJsonDatabase.h>

namespace toolhub::db {
struct ParsingException {
    vstd::string message;
    ParsingException() {}
    ParsingException(vstd::string &&msg)
        : message(std::move(msg)) {}
};
class IJsonDict;
class IJsonArray;

using ReadJsonVariant = vstd::variant<int64,
                                      double,
                                      vstd::string_view,
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
                                       bool,
                                       std::nullptr_t>;
using Key = vstd::variant<int64,
                          vstd::string_view,
                          vstd::Guid>;
struct JsonKeyPair {
    Key key;
    ReadJsonVariant value;
    JsonKeyPair(
        Key &&key,
        ReadJsonVariant &&value) : key(std::move(key)), value(std::move(value)) {}
};
class IJsonObject : public vstd::IDisposable {
protected:
    virtual ~IJsonObject() = default;

public:
    ////////// Basic
    virtual size_t Length() = 0;
    virtual vstd::vector<uint8_t> Serialize() = 0;
    virtual void Serialize(vstd::vector<uint8_t> &vec) = 0;
    virtual void Reset() = 0;
    virtual bool IsEmpty() = 0;
    virtual vstd::string FormattedPrint() = 0;
    virtual vstd::string Print() = 0;
    virtual bool Read(std::span<uint8_t const> sp,
                      bool clearLast) = 0;
    virtual void Reserve(size_t capacity) = 0;
    virtual vstd::optional<ParsingException> Parse(
        vstd::string_view str,
        bool clearLast) = 0;
    virtual vstd::MD5 GetMD5() = 0;
    vstd::IteEndTag end() const { return vstd::IteEndTag(); }
};

class IJsonDict : public IJsonObject {

protected:
    ~IJsonDict() = default;

public:
    virtual ReadJsonVariant Get(Key const &key) = 0;
    virtual void Set(Key const &key, WriteJsonVariant &&value) = 0;
    virtual bool TrySet(Key const &key, WriteJsonVariant &&value) = 0;
    virtual void Remove(Key const &key) = 0;
    virtual WriteJsonVariant GetAndSet(Key const &key, WriteJsonVariant &&newValue) = 0;
    virtual WriteJsonVariant GetAndRemove(Key const &key) = 0;
    virtual vstd::Iterator<JsonKeyPair> begin() const = 0;
};

class IJsonArray : public IJsonObject {

protected:
    ~IJsonArray() = default;

public:
    virtual ReadJsonVariant Get(size_t index) = 0;
    virtual void Set(size_t index, WriteJsonVariant &&value) = 0;
    virtual void Remove(size_t index) = 0;
    virtual void Add(WriteJsonVariant &&value) = 0;
    virtual WriteJsonVariant GetAndSet(size_t index, WriteJsonVariant &&newValue) = 0;
    virtual WriteJsonVariant GetAndRemove(size_t) = 0;
    virtual vstd::Iterator<ReadJsonVariant> begin() const = 0;
};
}// namespace toolhub::db