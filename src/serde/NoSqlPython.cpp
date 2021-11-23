#pragma vengine_package vengine_database
#include <serde/config.h>
#include <vstl/config.h>

#ifdef XX_VENGINE_PYTHON_SUPPORT
#include <serde/INoSqlDatabase.h>
#include <serde/DatabaseInclude.h>
#include <Utility/CommonIterators.h>
#include <serde/IJsonObject.h>
#include <vstl/StringUtility.h>
namespace toolhub::db::py {
 Database const* Database_GetFactory();

class SearchResultHandle : public vstd::IOperatorNewBase {
public:
	using ExternMap = INoSqlDatabase::Table;
	vstd::vector<ExternMap> results;
	ExternMap* mapPtr;
	ExternMap::Iterator idx = nullptr;
};

static vstd::vector<INoSqlDatabase::KeyType> keyStack;
static vstd::vector<INoSqlDatabase::ValueType> valueStack;
static vstd::vector<CompareFlag> compFlagStack;
 INoSqlDatabase* pynosql_create_db(char const* name, uint64 nameSize) {
	auto ptr = Database_GetFactory()->CreateSimpleNoSql(std::string_view(name, nameSize));
	return ptr;
}
 void pynosql_dispose_handle(SearchResultHandle* handle) {
	delete handle;
}
 void pynosql_dispose_db(INoSqlDatabase* ptr) {
	ptr->Dispose();
}
 void pynosql_clear_db(INoSqlDatabase* ptr) {
	ptr->Clear();
}
 void pynosql_print(INoSqlDatabase* ptr) {
	auto tables = ptr->GetAll();
	std::cout << "[\n";
	for (auto& i : tables) {
		std::cout << "{\n";
		for (auto& kv : i) {
			kv.first.visit([](auto&& v) { std::cout << v; });
			std::cout << ": ";
			kv.second.multi_visit(
				[](auto&& v) { std::cout << v; },
				[](auto&& v) { std::cout << v; },
				[](auto&& v) { std::cout << '"' << v << '"'; },
				[](auto&& v) { std::cout << v.ToString(); },
				[](auto&& v) {
					std::cout << (v ? "True" : "False");
				},
				[](auto&&) { std::cout << "null"; });
			std::cout << '\n';
		}
		std::cout << "}\n";
	}
	std::cout << "]\n";
}
// Search key , set value to valuePtr
/*
 uint64 pynosql_get_key_size(SearchResultHandle* handle) {
	return handle->idx->first.size();
}
 void pynosql_get_key(SearchResultHandle* handle, char* chr) {
	memcpy(chr, handle->idx->first.data(), handle->idx->first.size());
}*/
 uint8_t pynosql_get_key_type(SearchResultHandle* handle) {
	return handle->idx->first.GetType();
}
 int64 pynosql_get_int_key(SearchResultHandle* handle) {
	return handle->idx->first.multi_visit_or(
		int64(0),
		[](auto&& v) { return v; },
		[](auto&& v) { return v.size(); },
		[](auto&& v) { return 0; });
}
 void pynosql_get_str_key(SearchResultHandle* handle, char* ptr) {
	auto&& str = handle->idx->first.template force_get<vstd::string>();
	memcpy(ptr, str.data(), str.size());
}
 void pynosql_get_guid_key(SearchResultHandle* handle, vstd::Guid::GuidData* ptr) {
	*ptr = handle->idx->first.template force_get<vstd::Guid>().ToBinary();
}
 uint8_t pynosql_get_value_type(SearchResultHandle* handle) {
	return handle->idx->second.GetType();
}
 int64 pynosql_get_int_value(SearchResultHandle* handle) {
	return handle->idx->second.multi_visit_or(
		int64(0),
		[](auto&& it) { return it; },
		[](auto&& it) { return it; },
		[](auto&& it) { return it.size(); },
		[](auto&&) { return 0; },
		[](auto&& it) { return it; },
		[](auto&&) { return 0; });
}
 double pynosql_get_double_value(SearchResultHandle* handle) {
	return handle->idx->second.multi_visit_or(
		double(0),
		[](auto&& it) { return it; },
		[](auto&& it) { return it; },
		[](auto&& it) { return it.size(); },
		[](auto&&) { return 0; },
		[](auto&& it) { return it; },
		[](auto&&) { return 0; });
}
 void pynosql_get_str(SearchResultHandle* handle, char* ptr) {
	auto&& str = handle->idx->second.template force_get<vstd::string>();
	memcpy(ptr, str.data(), str.size());
}
 void pynosql_get_guid(SearchResultHandle* handle, vstd::Guid::GuidData* ptr) {
	*ptr = handle->idx->second.template force_get<vstd::Guid>().ToBinary();
}
 void pynosql_begin_map(SearchResultHandle* handle) {
	handle->mapPtr = handle->results.begin();
}
 void pynosql_next_map(SearchResultHandle* handle) {
	++handle->mapPtr;
}
 bool pynosql_map_end(SearchResultHandle* handle) {
	return handle->mapPtr == handle->results.end();
}

 void pynosql_begin_ele(SearchResultHandle* handle) {
	handle->idx = handle->mapPtr->begin();
}
 void pynosql_next_ele(SearchResultHandle* handle) {
	++handle->idx;
}
 bool pynosql_ele_end(SearchResultHandle* handle) {
	return handle->idx == handle->mapPtr->end();
}

enum class SelectLogic : uint8_t {
	Or,
	And
};
 void pynosql_clear_condition() {
	keyStack.clear();
	valueStack.clear();
	compFlagStack.clear();
}
 void pynosql_add_key_int(int64 value) {
	keyStack.emplace_back(value);
}
 void pynosql_add_key_str(char const* ptr, uint64 sz) {
	keyStack.emplace_back(std::string_view(ptr, sz));
}
 void pynosql_add_key_guid(vstd::Guid* guid) {
	keyStack.emplace_back(*guid);
}
 void pynosql_add_value_int(int64 value) {
	valueStack.emplace_back(value);
}
 void pynosql_add_value_double(double value) {
	valueStack.emplace_back(value);
}
 void pynosql_add_value_str(char const* ptr, uint64 sz) {
	valueStack.emplace_back(std::string_view(ptr, sz));
}
 void pynosql_add_value_bool(bool value) {
	valueStack.emplace_back(value);
}
 void pynosql_add_value_null() {
	valueStack.emplace_back(nullptr);
}
 void pynosql_add_value_guid(vstd::Guid* guid) {
	valueStack.emplace_back(*guid);
}
 void pynosql_add_flag(CompareFlag flag) {
	compFlagStack.emplace_back(flag);
}
class CompareKeyValueIterator final : public vstd::IEnumerable<INoSqlDatabase::CompareKey> {
public:
	INoSqlDatabase::KeyType* keyPtr;
	INoSqlDatabase::ValueType* valuePtr;
	CompareFlag* flagPtr;
	CompareKeyValueIterator() {
		keyPtr = keyStack.data();
		valuePtr = valueStack.data();
		flagPtr = compFlagStack.data();
	}
	using T = INoSqlDatabase::CompareKey;
	T GetValue() override {
		return {std::move(*keyPtr), std::move(*valuePtr), std::move(*flagPtr)};
	}
	bool End() override {
		return keyPtr == keyStack.end() || valuePtr == valueStack.end() || flagPtr == compFlagStack.end();
	}
	void GetNext() override {
		keyPtr++;
		valuePtr++;
		flagPtr++;
	}
	vstd::optional<size_t> Length() { return keyStack.size(); }
	void Dispose() override { delete this; }
};
class KeyValueIterator final : public vstd::IEnumerable<INoSqlDatabase::KeyValue> {
public:
	INoSqlDatabase::KeyType* keyPtr;
	INoSqlDatabase::ValueType* valuePtr;
	KeyValueIterator() {
		keyPtr = keyStack.data();
		valuePtr = valueStack.data();
	}
	using T = INoSqlDatabase::KeyValue;
	T GetValue() override {
		return {std::move(*keyPtr), std::move(*valuePtr)};
	}
	bool End() override {
		return keyPtr == keyStack.end() || valuePtr == valueStack.end();
	}
	void GetNext() override {
		keyPtr++;
		valuePtr++;
	}
	vstd::optional<size_t> Length() { return keyStack.size(); }
	void Dispose() override { delete this; }
};

 SearchResultHandle* pynosql_findone(INoSqlDatabase* db, bool deleteAfterFind, SelectLogic logic) {
	SearchResultHandle* handle = new SearchResultHandle();
	auto Add = [&](auto&& value) {
		if (value)
			handle->results.emplace_back(std::move(*value));
	};
	if (logic == SelectLogic::And) {
		if (deleteAfterFind)
			Add(db->FindOneAndDelete_And(vstd::get_rvalue_ptr(CompareKeyValueIterator())));
		else
			Add(db->FindOne_And(vstd::get_rvalue_ptr(CompareKeyValueIterator())));
	} else {
		if (deleteAfterFind)
			Add(db->FindOneAndDelete_Or(vstd::get_rvalue_ptr(CompareKeyValueIterator())));
		else
			Add(db->FindOne_Or(vstd::get_rvalue_ptr(CompareKeyValueIterator())));
	}
	return handle;
}
 SearchResultHandle* pynosql_findall(INoSqlDatabase* db, bool deleteAfterFind, SelectLogic logic) {
	SearchResultHandle* handle = new SearchResultHandle();
	auto&& results = handle->results;
	if (logic == SelectLogic::And) {
		if (deleteAfterFind) {
			results = db->FindAllAndDelete_And(vstd::get_rvalue_ptr(CompareKeyValueIterator()));
		} else
			results = db->FindAll_And(vstd::get_rvalue_ptr(CompareKeyValueIterator()));
	} else {
		if (deleteAfterFind) {
			results = db->FindAllAndDelete_Or(vstd::get_rvalue_ptr(CompareKeyValueIterator()));
		} else
			results = db->FindAll_Or(vstd::get_rvalue_ptr(CompareKeyValueIterator()));
	}
	return handle;
}
 void pynosql_deleteall(INoSqlDatabase* db, SelectLogic logic) {
	if (logic == SelectLogic::And) {
		db->DeleteAll_And(vstd::get_rvalue_ptr(CompareKeyValueIterator()));
	} else {
		db->DeleteAll_Or(vstd::get_rvalue_ptr(CompareKeyValueIterator()));
	}
}
 void pynosql_addall(INoSqlDatabase* db) {
	db->AddNode(vstd::get_rvalue_ptr(KeyValueIterator()));
}
 void pynosql_addall_json(INoSqlDatabase* db, IJsonDict* dct) {
	db->AddNode(dct);
}
}// namespace toolhub::db::py
#endif