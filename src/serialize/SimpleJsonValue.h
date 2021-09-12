#pragma once
#include <serialize/Common.h>
#include <serialize/IJsonObject.h>
#include <serialize/SimpleJsonLoader.h>
namespace toolhub::db {
class ConcurrentBinaryJson;
class SimpleBinaryJson;
struct SimpleJsonKey {
	using ValueType = vstd::variant<int64,
									std::string,
									vstd::Guid>;

	ValueType value;
	SimpleJsonKey(ValueType const& value)
		: value(value) {}
	SimpleJsonKey(ValueType&& value)
		: value(std::move(value)) {}
	SimpleJsonKey(Key const& v) {
		if (v.GetType() < ValueType::argSize) {
			value.update(v.GetType(), [&](void* ptr) {
				switch (v.GetType()) {
					case ValueType::IndexOf<int64>:
						new (ptr) int64(v.force_get<int64>());
						break;
					case ValueType::IndexOf<std::string>:
						new (ptr) std::string(v.force_get<std::string_view>());
						break;
					case ValueType::IndexOf<vstd::Guid>:
						new (ptr) vstd::Guid(v.force_get<vstd::Guid>());
						break;
				}
			});
		}
	}
	SimpleJsonKey(Key&& v)
		: SimpleJsonKey(v) {}
	Key GetKey() const {
		switch (value.GetType()) {
			case ValueType::IndexOf<int64>:
				return Key(value.force_get<int64>());
			case ValueType::IndexOf<vstd::Guid>:
				return Key(value.force_get<vstd::Guid>());
			case ValueType::IndexOf<std::string>:
				return Key(value.force_get<std::string>());
			default:
				return Key();
		}
	}
	bool operator==(SimpleJsonKey const& key) const {
		if (key.value.GetType() != value.GetType()) return false;
		switch (value.GetType()) {
			case ValueType::IndexOf<int64>:
				return value.force_get<int64>() == key.value.force_get<int64>();
			case ValueType::IndexOf<vstd::Guid>:
				return value.force_get<vstd::Guid>() == key.value.force_get<vstd::Guid>();
			case ValueType::IndexOf<std::string>:
				return value.force_get<std::string>() == key.value.force_get<std::string>();
		}
		return true;
	}

	bool EqualToKey(Key const& key) const {
		if (key.GetType() != value.GetType()) return false;
		switch (value.GetType()) {
			case ValueType::IndexOf<int64>:
				return value.force_get<int64>() == key.force_get<int64>();
			case ValueType::IndexOf<vstd::Guid>:
				return value.force_get<vstd::Guid>() == key.force_get<vstd::Guid>();
			case ValueType::IndexOf<std::string>:
				return value.force_get<std::string>() == key.force_get<std::string_view>();
		}
		return true;
	}
	size_t GetHashCode() const {
		auto getHash = [](auto&& v) {
			vstd::hash<std::remove_cvref_t<decltype(v)>> h;
			return h(v);
		};
		switch (value.GetType()) {
			case ValueType::IndexOf<int64>:
				return getHash(*reinterpret_cast<int64 const*>(value.GetPlaceHolder()));

			case ValueType::IndexOf<vstd::Guid>:
				return getHash(*reinterpret_cast<vstd::Guid const*>(value.GetPlaceHolder()));

			case ValueType::IndexOf<std::string>:
				return getHash(*reinterpret_cast<std::string const*>(value.GetPlaceHolder()));
		}
        return 0u;
	}

	bool operator!=(SimpleJsonKey const& key) const {
		return !operator==(key);
	}
};
struct SimpleJsonKeyHash {
	template<typename T>
	size_t operator()(T const& key) const {
		if constexpr (std::is_same_v<std::remove_cvref_t<T>, SimpleJsonKey>)
			return key.GetHashCode();
		else {
			auto getHash = [](auto&& v) {
				vstd::hash<std::remove_cvref_t<decltype(v)>> h;
				return h(v);
			};
			switch (key.GetType()) {
				case Key::IndexOf<int64>:
					return getHash(*reinterpret_cast<int64 const*>(key.GetPlaceHolder()));

				case Key::IndexOf<vstd::Guid>:
					return getHash(*reinterpret_cast<vstd::Guid const*>(key.GetPlaceHolder()));

				case Key::IndexOf<std::string_view>:
					return getHash(*reinterpret_cast<std::string const*>(key.GetPlaceHolder()));
			}
            return 0u;
		}
	}
};
struct SimpleJsonKeyEqual {
	template<typename T>
	bool operator()(SimpleJsonKey const& key, T const& t) const {
		if constexpr (std::is_same_v<std::remove_cvref_t<T>, Key>) {
			return key.EqualToKey(t);
		} else {
			return key == t;
		}
	}
};
using KVMap = vstd::HashMap<SimpleJsonKey, SimpleJsonVariant, SimpleJsonKeyHash, SimpleJsonKeyEqual>;
class SimpleJsonValue {
protected:
	SimpleBinaryJson* db;
};

class SimpleJsonValueDict final : public IJsonDict, public SimpleJsonValue {

public:
	void Dispose() override;
	[[nodiscard]] SimpleBinaryJson* GetDB() const { return db; }
	KVMap vars;
	explicit SimpleJsonValueDict(SimpleBinaryJson* db);
	~SimpleJsonValueDict();
	/* SimpleJsonValueDict(
		SimpleBinaryJson* db,
		IJsonDict* src);*/
	ReadJsonVariant Get(Key const& key) override;
	void Set(Key const& key, WriteJsonVariant&& value) override;
	bool TrySet(Key const& key, WriteJsonVariant&& value) override;
	void Remove(Key const& key) override;
	[[nodiscard]] vstd::Iterator<JsonKeyPair> begin() const override;
	size_t Length() override;
	std::vector<uint8_t> Serialize() override;
	void M_GetSerData(std::vector<uint8_t>& arr);
	void LoadFromSer(std::span<uint8_t const>& arr);
	bool Read(std::span<uint8_t const> sp,
			  bool clearLast) override;
	void Reset() override;
	void Reserve(size_t capacity) override {
		vars.reserve(capacity);
	}
	vstd::optional<ParsingException> Parse(
		std::string_view str, bool clearLast) override;
	bool IsEmpty() override { return vars.size() == 0; }
	WriteJsonVariant GetAndSet(Key const& key, WriteJsonVariant&& newValue) override;
	WriteJsonVariant GetAndRemove(Key const& key) override;
	void M_Print(std::string& str, size_t space);
	std::string Print() override {
		std::string str;
		M_Print(str, 0);
		return str;
	}
	vstd::MD5 GetMD5() override;
};

class SimpleJsonValueArray final : public IJsonArray, public SimpleJsonValue {

public:
	void Dispose() override;
	[[nodiscard]] SimpleBinaryJson* GetDB() const { return db; }
	std::vector<SimpleJsonVariant> arr;
	explicit SimpleJsonValueArray(SimpleBinaryJson* db);
	~SimpleJsonValueArray();
	/* SimpleJsonValueArray(
		SimpleBinaryJson* db,
		IJsonArray* src);*/
	size_t Length() override;
	void Reserve(size_t capacity) override {
		arr.reserve(capacity);
	}
	vstd::optional<ParsingException> Parse(
		std::string_view str,
		bool clearLast) override;
	std::vector<uint8_t> Serialize() override;
	void M_GetSerData(std::vector<uint8_t>& result);
	void LoadFromSer(std::span<uint8_t const>& arr);
	bool Read(std::span<uint8_t const> sp, bool clearLast) override;
	void Reset() override;

	ReadJsonVariant Get(size_t index) override;
	void Set(size_t index, WriteJsonVariant&& value) override;
	void Remove(size_t index) override;
	void Add(WriteJsonVariant&& value) override;
	[[nodiscard]] vstd::Iterator<ReadJsonVariant> begin() const override;
	bool IsEmpty() override { return arr.empty(); }
	WriteJsonVariant GetAndSet(size_t index, WriteJsonVariant&& newValue) override;
	WriteJsonVariant GetAndRemove(size_t) override;
	void M_Print(std::string& str, size_t space);
	std::string Print() override {
		std::string str;
		M_Print(str, 0);
		return str;
	}
	vstd::MD5 GetMD5() override;
};
class ConcurrentJsonValue {
protected:
	ConcurrentBinaryJson* db;
	std::mutex mtx;
};
class ConcurrentJsonValueDict final : public IJsonDict, public ConcurrentJsonValue {

public:
	void Dispose() override;
	[[nodiscard]] ConcurrentBinaryJson* GetDB() const { return db; }
	KVMap vars;
	explicit ConcurrentJsonValueDict(ConcurrentBinaryJson* db);
	~ConcurrentJsonValueDict();
	ReadJsonVariant Get(Key const& key) override;
	void Set(Key const& key, WriteJsonVariant&& value) override;
	bool TrySet(Key const& key, WriteJsonVariant&& value) override;
	void Remove(Key const& key) override;
	[[nodiscard]] vstd::Iterator<JsonKeyPair> begin() const override;
	size_t Length() override;
	std::vector<uint8_t> Serialize() override;
	void M_GetSerData(std::vector<uint8_t>& arr);
	void LoadFromSer(std::span<uint8_t const>& arr);
	bool Read(std::span<uint8_t const> sp, bool clearLast) override;
	void Reset() override;
	void Reserve(size_t capacity) override {
		std::lock_guard lck(mtx);
		vars.reserve(capacity);
	}
	vstd::optional<ParsingException> Parse(
		std::string_view str, bool clearLast) override;
	bool IsEmpty() override { return vars.size() == 0; }
	WriteJsonVariant GetAndSet(Key const& key, WriteJsonVariant&& newValue) override;
	WriteJsonVariant GetAndRemove(Key const& key) override;
	void M_Print(std::string& str, size_t space);
	std::string Print() override {
		std::string str;
		M_Print(str, 0);
		return str;
	}
	vstd::MD5 GetMD5() override;
};

class ConcurrentJsonValueArray final : public IJsonArray, public ConcurrentJsonValue {

public:
	void Dispose() override;
	[[nodiscard]] ConcurrentBinaryJson* GetDB() const { return db; }
	std::vector<SimpleJsonVariant> arr;
	explicit ConcurrentJsonValueArray(ConcurrentBinaryJson* db);
	~ConcurrentJsonValueArray();
	size_t Length() override;
	void Reserve(size_t capacity) override {
		std::lock_guard lck(mtx);
		arr.reserve(capacity);
	}
	vstd::optional<ParsingException> Parse(
		std::string_view str, bool clearLast) override;
	std::vector<uint8_t> Serialize() override;
	void M_GetSerData(std::vector<uint8_t>& result);
	void LoadFromSer(std::span<uint8_t const>& arr);
	bool Read(std::span<uint8_t const> sp, bool clearLast) override;
	void Reset() override;

	ReadJsonVariant Get(size_t index) override;
	void Set(size_t index, WriteJsonVariant&& value) override;
	void Remove(size_t index) override;
	void Add(WriteJsonVariant&& value) override;
	[[nodiscard]] vstd::Iterator<ReadJsonVariant> begin() const override;
	bool IsEmpty() override { return arr.empty(); }
	WriteJsonVariant GetAndSet(size_t index, WriteJsonVariant&& newValue) override;
	WriteJsonVariant GetAndRemove(size_t) override;
	void M_Print(std::string& str, size_t space);
	std::string Print() override {
		std::string str;
		M_Print(str, 0);
		return str;
	}
	vstd::MD5 GetMD5() override;
};
}// namespace toolhub::db