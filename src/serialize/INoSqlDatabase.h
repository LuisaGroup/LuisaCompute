#pragma once
#include <serialize/IJsonObject.h>
namespace toolhub::db {
enum class CompareFlag : uint8_t {
	Less,
	LessEqual,
	Equal,
	GreaterEqual,
	Greater,
	Always
};
class INoSqlDatabase : public vstd::IDisposable {
public:
	using KeyType = vstd::variant<
		int64,
		vstd::string,
		vstd::Guid>;
	using ValueType = vstd::variant<
		int64,
		double,
		vstd::string,
		vstd::Guid,
		bool,
		std::nullptr_t>;
	using KeyValue = std::pair<KeyType, ValueType>;
	using CompareKey = std::tuple<KeyType, ValueType, CompareFlag>;
	using Table = vstd::HashMap<KeyType, ValueType>;
	virtual void AddNode(vstd::IEnumerable<KeyValue>* keyValues) = 0;
	virtual void AddNode(IJsonDict* dictNode) = 0;
	virtual vstd::optional<Table> FindOne(CompareKey const& key) = 0;
	virtual vstd::vector<Table> FindAll(CompareKey const& key) = 0;
	virtual vstd::optional<Table> FindOne_And(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::vector<Table> FindAll_And(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::optional<Table> FindOne_Or(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::vector<Table> FindAll_Or(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::optional<Table> FindOneAndDelete(CompareKey const& key) = 0;
	virtual vstd::vector<Table> FindAllAndDelete(CompareKey const& key) = 0;
	virtual vstd::optional<Table> FindOneAndDelete_And(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::vector<Table> FindAllAndDelete_And(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::optional<Table> FindOneAndDelete_Or(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual vstd::vector<Table> FindAllAndDelete_Or(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual void DeleteAll(CompareKey const& key) = 0;
	virtual void DeleteAll_And(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual void DeleteAll_Or(vstd::IEnumerable<CompareKey>* keys) = 0;
	virtual void Clear() = 0;
	virtual vstd::vector<Table> GetAll() = 0;
};
class INoSqlExecutor : public vstd::IDisposable {
public:
	virtual void Execute(INoSqlDatabase* db, vstd::string_view cmd) const = 0;
};
}// namespace toolhub::db