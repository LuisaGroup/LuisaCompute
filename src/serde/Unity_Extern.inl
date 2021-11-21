#pragma once
#include <serde/SimpleBinaryJson.h>
#include <serde/SimpleJsonValue.h>
#include <Utility/BinaryReader.h>
namespace toolhub::db {
template<typename T>
void DBParse(T* db, std::string_view strv, vstd::funcPtr_t<void(std::string_view)> errorCallback, bool clearLast) {
	auto errorMsg = db->Parse(strv, clearLast);
	if (errorMsg) {
		errorCallback(errorMsg->message);
	} else {
		errorCallback(std::string_view(nullptr, 0));
	}
}

 void db_get_new(SimpleBinaryJson** pp) {
	*pp = new SimpleBinaryJson();
}
 void db_dispose(SimpleBinaryJson* p) {
	delete p;
}
 void db_compile_from_py(SimpleBinaryJson* p, wchar_t const* datas, int64 sz, bool* result) {
	vstd::string str;
	str.resize(sz);
	for (auto i : vstd::range(sz)) {
		str[i] = datas[i];
	}
	*result = p->CompileFromPython(str.data());
}
 void db_get_rootnode(SimpleBinaryJson* db, SimpleJsonValueDict** pp) {
	*pp = static_cast<SimpleJsonValueDict*>(db->GetRootNode());
}
 void db_dict_print(SimpleJsonValueDict* db, vstd::funcPtr_t<void(std::string_view)> callback) {
	auto str = db->Print();
	callback(str);
}
 void db_dict_print_yaml(SimpleJsonValueDict* db, vstd::funcPtr_t<void(std::string_view)> callback) {
	auto str = db->PrintYaml();
	callback(str);
}
 void db_arr_print(SimpleJsonValueArray* db, vstd::funcPtr_t<void(char const*, uint64)> callback) {
	auto str = db->Print();
	callback(str.data(), str.size());
}
 void db_dict_format_print(SimpleJsonValueDict* db, vstd::funcPtr_t<void(std::string_view)> callback) {
	auto str = db->FormattedPrint();
	callback(str);
}
 void db_arr_format_print(SimpleJsonValueArray* db, vstd::funcPtr_t<void(char const*, uint64)> callback) {
	auto str = db->FormattedPrint();
	callback(str.data(), str.size());
}
 void db_create_dict(SimpleBinaryJson* db, SimpleJsonValueDict** pp) {
	*pp = db->CreateDict_Nake();
}
 void db_create_array(SimpleBinaryJson* db, SimpleJsonValueArray** pp) {
	*pp = db->CreateArray_Nake();
}

 void db_arr_ser(SimpleJsonValueArray* db, vstd::funcPtr_t<void(uint8_t*, uint64)> callback) {
	auto vec = db->Serialize();
	callback(vec.data(), vec.size());
}
 void db_dict_ser(SimpleJsonValueDict* db, vstd::funcPtr_t<void(uint8_t*, uint64)> callback) {
	auto vec = db->Serialize();
	callback(vec.data(), vec.size());
}
 void db_arr_deser(SimpleJsonValueArray* db, uint8_t* ptr, uint64 len, bool* success, bool clearLast) {
	*success = db->Read(std::span<uint8_t const>(ptr, len), clearLast);
}
 void db_dict_deser(SimpleJsonValueDict* db, uint8_t* ptr, uint64 len, bool* success, bool clearLast) {
	*success = db->Read(std::span<uint8_t const>(ptr, len), clearLast);
}

 void db_dispose_arr(SimpleJsonValueArray* p) {
	p->Dispose();
}

////////////////// Dict Area

 void db_dispose_dict(SimpleJsonValueDict* ptr) {
	ptr->Dispose();
}
 void db_dict_set(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType,
	void* valuePtr,
	CSharpValueType valueType) {
	dict->Set(GetCSharpKey(keyPtr, keyType), GetCSharpWriteValue(valuePtr, valueType));
}
 void db_dict_tryset(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType,
	void* valuePtr,
	CSharpValueType valueType,
	bool* isTry) {
	Key key;
	WriteJsonVariant value;
	*isTry = false;
	dict->TrySet(GetCSharpKey(keyPtr, keyType), [&]() {
		*isTry = true;
		return GetCSharpWriteValue(valuePtr, valueType);
	});
}
 void db_dict_get(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType,
	CSharpValueType targetValueType,
	void* valuePtr) {
	SetCSharpReadValue(valuePtr, targetValueType, dict->Get(GetCSharpKey(keyPtr, keyType)));
}
 void db_dict_get_variant(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType,
	CSharpValueType* targetValueType,
	void* valuePtr) {
	auto value = dict->Get(GetCSharpKey(keyPtr, keyType));
	*targetValueType = SetCSharpReadValue(valuePtr, value);
}
 void db_dict_remove(SimpleJsonValueDict* dict, void* keyPtr, CSharpKeyType keyType) {
	dict->Remove(GetCSharpKey(keyPtr, keyType));
}
 void db_dict_len(SimpleJsonValueDict* dict, int32* sz) { *sz = dict->Length(); }
 void db_dict_itebegin(SimpleJsonValueDict* dict, DictIterator* ptr) { *ptr = dict->vars.begin(); }
 void db_dict_iteend(SimpleJsonValueDict* dict, DictIterator* end, bool* result) { *result = (*end == dict->vars.end()); }
 void db_dict_ite_next(DictIterator* end) { (*end)++; }
 void db_dict_ite_get(DictIterator ite, void* valuePtr, CSharpValueType valueType) {
	SetCSharpReadValue(valuePtr, valueType, ite->second.GetVariant());
}
 void db_dict_ite_get_variant(DictIterator ite, void* valuePtr, CSharpValueType* valueType) {
	*valueType = SetCSharpReadValue(valuePtr, ite->second.GetVariant());
}

 void db_dict_ite_getkey(DictIterator ite, void* keyPtr, CSharpKeyType keyType) {
	SetCSharpKey(keyPtr, keyType, ite->first.GetKey());
}
 void db_dict_ite_getkey_variant(DictIterator ite, void* keyPtr, CSharpKeyType* keyType) {
	*keyType = SetCSharpKey(keyPtr, ite->first.GetKey());
}
 void db_dict_reset(
	SimpleJsonValueDict* dict) {
	dict->Reset();
}
 void db_dict_parse(SimpleJsonValueDict* db, std::string_view strv, vstd::funcPtr_t<void(std::string_view)> errorCallback, bool clearLast) {
	DBParse(db, strv, errorCallback, clearLast);
}
 void db_dict_parse_yaml(SimpleJsonValueDict* db, std::string_view strv, vstd::funcPtr_t<void(std::string_view)> errorCallback, bool clearLast) {
	auto errorMsg = db->ParseYaml(strv, clearLast);
	if (errorMsg) {
		errorCallback(errorMsg->message);
	} else {
		errorCallback(std::string_view(nullptr, 0));
	}
}
////////////////// Array Area
 void db_arr_reset(
	SimpleJsonValueArray* arr) {
	arr->Reset();
}
 void db_arr_len(SimpleJsonValueArray* arr, int32* sz) {
	*sz = arr->Length();
}
 void db_arr_get_value(SimpleJsonValueArray* arr, int32 index, void* valuePtr, CSharpValueType valueType) {
	SetCSharpReadValue(valuePtr, valueType, arr->Get(index));
}
 void db_arr_get_value_variant(SimpleJsonValueArray* arr, int32 index, void* valuePtr, CSharpValueType* valueType) {
	*valueType = SetCSharpReadValue(valuePtr, arr->Get(index));
}
 void db_arr_set_value(SimpleJsonValueArray* arr, int32 index, void* valuePtr, CSharpValueType valueType) {
	arr->Set(index, GetCSharpWriteValue(valuePtr, valueType));
}
 void db_arr_add_value(SimpleJsonValueArray* arr, void* valuePtr, CSharpValueType valueType) {
	arr->Add(GetCSharpWriteValue(valuePtr, valueType));
}
 void db_arr_remove(SimpleJsonValueArray* arr, int32 index) {
	arr->Remove(index);
}
 void db_arr_itebegin(SimpleJsonValueArray* arr, ArrayIterator* ptr) {
	*ptr = arr->arr.begin();
}
 void db_arr_iteend(SimpleJsonValueArray* arr, ArrayIterator* ptr, bool* result) {
	*result = (*ptr == arr->arr.end());
}
 void db_arr_ite_next(SimpleJsonValueArray* arr, ArrayIterator* ptr) {
	(*ptr)++;
}
 void db_arr_ite_get(ArrayIterator ite, void* valuePtr, CSharpValueType valueType) {
	SetCSharpReadValue(valuePtr, valueType, ite->GetVariant());
}
 void db_arr_ite_get_variant(ArrayIterator ite, void* valuePtr, CSharpValueType* valueType) {
	*valueType = SetCSharpReadValue(valuePtr, ite->GetVariant());
}
 void db_arr_parse(SimpleJsonValueArray* db, std::string_view strv, vstd::funcPtr_t<void(std::string_view)> errorCallback, bool clearLast) {
	DBParse(db, strv, errorCallback, clearLast);
}
}// namespace toolhub::db