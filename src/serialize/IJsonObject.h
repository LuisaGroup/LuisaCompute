#pragma once

#include <util/VGuid.h>
#include <serialize/IJsonDatabase.h>

namespace toolhub::db {

class IJsonDict;
class IJsonArray;

using ReadJsonVariant = vstd::variant<
    int64,
    double,
    std::string_view,
    IJsonDict *,
    IJsonArray *,
    vstd::Guid>;

using WriteJsonVariant = vstd::variant<
    int64,
    double,
    std::string,
    UniquePtr<IJsonDict>,
    UniquePtr<IJsonArray>,
    vstd::Guid>;

using Key = vstd::variant<
    int64,
    std::string_view,
    vstd::Guid>;

struct JsonKeyPair {
    Key key;
    ReadJsonVariant value;
    JsonKeyPair(
        Key &&key,
        ReadJsonVariant &&value) : key(std::move(key)), value(std::move(value)) {}
};

class IJsonObject : protected vstd::IDisposable {

protected:
    ~IJsonObject() = default;

public:
    virtual size_t Length() = 0;
    virtual std::vector<uint8_t> Serialize() = 0;
    virtual void Reset() = 0;
    virtual bool IsEmpty() = 0;
    virtual std::string Print() = 0;
    virtual bool Read(std::span<uint8_t const> sp,
                      bool clearLast) = 0;
    virtual void Reserve(size_t capacity) = 0;
    virtual vstd::optional<ParsingException> Parse(
        std::string_view str,
        bool clearLast) = 0;
    virtual vstd::MD5 GetMD5() = 0;
    static vstd::IteEndTag end() { return {}; }
};

class IJsonDict : public IJsonObject {
    friend class std::unique_ptr<IJsonDict, Disposer>;

protected:
    ~IJsonDict() noexcept = default;

public:
    virtual ReadJsonVariant Get(Key const &key) = 0;
    virtual void Set(Key const &key, WriteJsonVariant &&value) = 0;
    virtual bool TrySet(Key const &key, WriteJsonVariant &&value) = 0;
    virtual void Remove(Key const &key) = 0;
    virtual WriteJsonVariant GetAndSet(Key const &key, WriteJsonVariant &&newValue) = 0;
    virtual WriteJsonVariant GetAndRemove(Key const &key) = 0;
    [[nodiscard]] virtual vstd::Iterator<JsonKeyPair> begin() const = 0;
};

class IJsonArray : public IJsonObject {
    friend class std::unique_ptr<IJsonArray, Disposer>;

protected:
    ~IJsonArray() noexcept = default;

public:
    virtual ReadJsonVariant Get(size_t index) = 0;
    virtual void Set(size_t index, WriteJsonVariant &&value) = 0;
    virtual void Remove(size_t index) = 0;
    virtual void Add(WriteJsonVariant &&value) = 0;
    virtual WriteJsonVariant GetAndSet(size_t index, WriteJsonVariant &&newValue) = 0;
    virtual WriteJsonVariant GetAndRemove(size_t) = 0;
    [[nodiscard]] virtual vstd::Iterator<ReadJsonVariant> begin() const = 0;
};

}// namespace toolhub::db