#pragma once
#include <serde/SimpleBinaryJson.h>
#include <serde/SimpleJsonValue.h>
#include <Utility/BinaryReader.h>
namespace toolhub::db {
struct PyStruct {
	union {
		vstd::StackObject<int64> intValue;
		vstd::StackObject<double> floatValue;
		vstd::StackObject<std::string_view> strv;
		vstd::StackObject<vstd::Guid> guid;
		vstd::StackObject<IJsonDict*> dict;
		vstd::StackObject<IJsonArray*> arr;
	};
	PyStruct() {}
};
static PyStruct pyStruct;
static vstd::vector<uint8_t> pyDataCache;
 uint64 pydb_get_data_len() {

	return pyDataCache.size();
}
 void pydb_cpy_data(uint8_t* ptr) {
	memcpy(ptr, pyDataCache.data(), pyDataCache.size());
}
 IJsonDatabase* pydb_get_new() {
	return new SimpleBinaryJson();
}
 void pydb_dispose(IJsonDatabase* p) {
	delete static_cast<SimpleBinaryJson*>(p);
}
 void pydb_dict_dispose(SimpleJsonValueDict* p) {
	p->Dispose();
}
 void pydb_arr_dispose(SimpleJsonValueArray* p) {
	p->Dispose();
}
 void* pydb_get_rootnode(IJsonDatabase* db) {
	return db->GetRootNode();
}
 void* pydb_create_dict(IJsonDatabase* db) {
	return static_cast<SimpleBinaryJson*>(db)->CreateDict_Nake();
}
 void* pydb_create_array(IJsonDatabase* db) {
	return static_cast<SimpleBinaryJson*>(db)->CreateArray_Nake();
}
 void* pydb_dict_create_dict(SimpleJsonValueDict* db) {
	return db->MGetDB()->CreateDict_Nake();
}
 void* pydb_dict_create_array(SimpleJsonValueDict* db) {
	return db->MGetDB()->CreateArray_Nake();
}
 void* pydb_arr_create_dict(SimpleJsonValueArray* db) {
	return db->MGetDB()->CreateDict_Nake();
}
 void* pydb_arr_create_array(SimpleJsonValueArray* db) {
	return db->MGetDB()->CreateArray_Nake();
}
template<typename T>
void TSer(T* v) {
	pyDataCache = v->Serialize();
}
template<typename T>
bool TDeser(T* v, std::span<uint8_t const> data, bool clearLast) {
	return v->Read(data, clearLast);
}
static vstd::optional<vstd::string> printableStr;
template<typename T>
void TFormattedPrint(T* v) {
	printableStr = v->FormattedPrint();
	pyStruct.strv.New(*printableStr);
}
template<typename T>
void TPrint(T* v) {
	printableStr = v->Print();
	pyStruct.strv.New(*printableStr);
}
template<typename T>
bool TParse(T* v, char const* str, uint64 strLen, bool clearLast) {
	auto error = v->Parse(std::string_view(str, strLen), clearLast);
	if (error) {
		std::cout << error->message;
	}
	return !error;
}
template<typename T>
bool TParseYaml(T* v, char const* str, uint64 strLen, bool clearLast) {
	auto error = v->ParseYaml(std::string_view(str, strLen), clearLast);
	if (error) {
		std::cout << error->message;
	}
	return !error;
}
 void py_clear_printstr() {
	printableStr.Delete();
}
 void pydb_dict_ser(SimpleJsonValueDict* v) {
	TSer(v);
}
 void pydb_arr_ser(SimpleJsonValueArray* v) {
	TSer(v);
}
 bool pydb_dict_deser(SimpleJsonValueDict* v, uint8_t const* ptr, uint64 size, bool clearLast) {
	return TDeser(v, std::span<uint8_t const>(ptr, size), clearLast);
}
 bool pydb_arr_deser(SimpleJsonValueArray* v, uint8_t const* ptr, uint64 size, bool clearLast) {
	return TDeser(v, std::span<uint8_t const>(ptr, size), clearLast);
}
 void pydb_dict_print(SimpleJsonValueDict* v) {
	TPrint(v);
}
 void pydb_arr_print(SimpleJsonValueArray* v) {
	TPrint(v);
}
 void pydb_dict_print_formatted(SimpleJsonValueDict* v) {
	TFormattedPrint(v);
}
 void pydb_arr_print_formatted(SimpleJsonValueArray* v) {
	TFormattedPrint(v);
}
 bool pydb_dict_parse(SimpleJsonValueDict* v, char const* str, uint64 strLen, bool clearLast) {
	return TParse(v, str, strLen, clearLast);
}
 bool pydb_dict_parse_yaml(SimpleJsonValueDict* v, char const* str, uint64 strLen, bool clearLast) {
	return TParseYaml(v, str, strLen, clearLast);
}
 void pydb_dict_print_yaml(SimpleJsonValueDict* v) {
	printableStr = v->PrintYaml();
	pyStruct.strv.New(*printableStr);
}
 bool pydb_arr_parse(SimpleJsonValueArray* v, char const* str, uint64 strLen, bool clearLast) {
	return TParse(v, str, strLen, clearLast);
}

 void pydb_dict_set(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType,
	void* valuePtr,
	CSharpValueType valueType) {
	dict->Set(GetCSharpKey(keyPtr, keyType), GetCSharpWriteValue(valuePtr, valueType));
}

 CSharpValueType pydb_dict_get_variant(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType) {
	auto value = dict->Get(GetCSharpKey(keyPtr, keyType));
	return SetCSharpReadValue(&pyStruct, value);
}
 void pydb_dict_remove(SimpleJsonValueDict* dict, void* keyPtr, CSharpKeyType keyType) {
	dict->Remove(GetCSharpKey(keyPtr, keyType));
}

 uint pydb_dict_len(SimpleJsonValueDict* dict) { return dict->Length(); }
 uint64 pydb_dict_itebegin(SimpleJsonValueDict* dict) {
	auto ite = dict->vars.begin();
	return *reinterpret_cast<uint64 const*>(&ite);
}
 bool pydb_dict_iteend(SimpleJsonValueDict* dict, DictIterator end) { return (end == dict->vars.end()); }
 uint64 pydb_dict_ite_next(DictIterator end) {
	end++;
	return *reinterpret_cast<uint64 const*>(&end);
}
 CSharpValueType pydb_dict_ite_get_variant(DictIterator ite) {
	return SetCSharpReadValue(&pyStruct, ite->second.GetVariant());
}
 CSharpKeyType pydb_dict_ite_getkey_variant(DictIterator ite) {
	return SetCSharpKey(&pyStruct, ite->first.GetKey());
}
 void pydb_dict_reset(
	SimpleJsonValueDict* dict) {
	dict->Reset();
}
////////////////// Array Area
 void pydb_arr_reset(
	SimpleJsonValueArray* arr) {
	arr->Reset();
}
 uint pydb_arr_len(SimpleJsonValueArray* arr) {
	return arr->Length();
}
 CSharpValueType pydb_arr_get_value_variant(SimpleJsonValueArray* arr, uint index) {
	return SetCSharpReadValue(&pyStruct, arr->Get(index));
}
 void pydb_arr_set_value(SimpleJsonValueArray* arr, uint index, void* valuePtr, CSharpValueType valueType) {
	arr->Set(index, GetCSharpWriteValue(valuePtr, valueType));
}
 void pydb_arr_add_value(SimpleJsonValueArray* arr, void* valuePtr, CSharpValueType valueType) {
	arr->Add(GetCSharpWriteValue(valuePtr, valueType));
}
 void pydb_arr_remove(SimpleJsonValueArray* arr, uint index) {
	arr->Remove(index);
}

 uint64 pydb_arr_itebegin(SimpleJsonValueArray* arr) {
	auto a = arr->arr.begin();
	return *reinterpret_cast<uint64 const*>(&a);
}
 bool pydb_arr_iteend(SimpleJsonValueArray* arr, ArrayIterator ptr) {
	return (ptr == arr->arr.end());
}
 uint64 pydb_arr_ite_next(ArrayIterator ptr) {
	(ptr)++;
	return *reinterpret_cast<uint64 const*>(&ptr);
}
 CSharpValueType pydb_arr_ite_get_variant(ArrayIterator ite) {
	return SetCSharpReadValue(&pyStruct, ite->GetVariant());
}
 uint64 py_get_str_size() {
	return pyStruct.strv->size();
}
 void py_get_chars(char* ptr) {
	memcpy(ptr, pyStruct.strv->data(), pyStruct.strv->size());
}
 void py_bind_64(int64* ptr) {
	*ptr = *pyStruct.intValue;
}
 void py_bind_128(int64* ptr) {
	int64* otherPtr = reinterpret_cast<int64*>(&pyStruct);
	ptr[0] = otherPtr[0];
	ptr[1] = otherPtr[1];
}

 void py_bind_strview(
	std::string_view* strv,
	char* ptr,
	uint64 len) { *strv = std::string_view(ptr, len); }
}// namespace toolhub::db