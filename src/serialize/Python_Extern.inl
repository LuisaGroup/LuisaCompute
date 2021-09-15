#pragma once
#include <serialize/Common.h>
#include <serialize/SimpleBinaryJson.h>
#include <serialize/SimpleJsonValue.h>

namespace toolhub::db {
struct PyStruct {
	union {
		vstd::StackObject<int64> intValue;
		vstd::StackObject<double> floatValue;
		vstd::StackObject<CSharpStringView> strv;
		vstd::StackObject<vstd::Guid> guid;
		vstd::StackObject<IJsonDict*> dict;
		vstd::StackObject<IJsonArray*> arr;
	};
	PyStruct() {}
};
static PyStruct pyStruct;

LUISA_EXPORT_API SimpleBinaryJson* pydb_get_new() {
	return new SimpleBinaryJson();
}
LUISA_EXPORT_API void pydb_dispose(SimpleBinaryJson* p) {
	p->Dispose();
}
LUISA_EXPORT_API void pydb_dict_dispose(SimpleJsonValueDict* p) {
	p->Dispose();
}
LUISA_EXPORT_API void pydb_arr_dispose(SimpleJsonValueArray* p) {
	p->Dispose();
}
LUISA_EXPORT_API void* pydb_get_rootnode(SimpleBinaryJson* db) {
	return db->GetRootNode();
}
LUISA_EXPORT_API void* pydb_create_dict(SimpleBinaryJson* db) {
	return db->CreateDict_Nake();
}
LUISA_EXPORT_API void* pydb_create_array(SimpleBinaryJson* db) {
	return db->CreateArray_Nake();
}
LUISA_EXPORT_API void* pydb_dict_create_dict(SimpleJsonValueDict* db) {
	return db->GetDB()->CreateDict_Nake();
}
LUISA_EXPORT_API void* pydb_dict_create_array(SimpleJsonValueDict* db) {
	return db->GetDB()->CreateArray_Nake();
}
LUISA_EXPORT_API void* pydb_arr_create_dict(SimpleJsonValueArray* db) {
	return db->GetDB()->CreateDict_Nake();
}
LUISA_EXPORT_API void* pydb_arr_create_array(SimpleJsonValueArray* db) {
	return db->GetDB()->CreateArray_Nake();
}
template<typename T>
void TSer(T* v, char const* filePath) {
	auto vec = v->Serialize();
	auto file = fopen(filePath, "wb");
	if (file == nullptr) {
		std::cout << "Cannot write to path: " << filePath << '\n';
		return;
	}
	auto disp = vstd::create_disposer([&]() {
		fclose(file);
	});
	fwrite(vec.data(), vec.size(), 1, file);
}
template<typename T>
void TDeser(T* v, char const* filePath, bool clearLast) {
    auto ifs = fopen(filePath, "wb");
    if (!ifs) {
        std::cout << "Cannot open path: " << filePath << '\n';
        return;
    }
    auto disp = vstd::create_disposer([&]() {
        fclose(ifs);
    });
    fseek(ifs, 0, SEEK_END);
    auto len = ftell(ifs);
    fseek(ifs, 0, SEEK_SET);
    std::vector<uint8_t> result(len);
    fread(result.data(), len, 1, ifs);
	
   if (!v->Read(result, clearLast)) {
        std::cout << "Illegal deserialize file: " << filePath;
    }
}
static vstd::StackObject<luisa::string, true> printableStr;
template<typename T>
void TPrint(T* v) {
	printableStr = v->Print();
	pyStruct.strv.New(*printableStr);
}
template<typename T>
void TParse(T* v, char const* str, uint64 strLen, bool clearLast) {
	auto error = v->Parse(std::string_view(str, strLen), clearLast);
}
LUISA_EXPORT_API void py_clear_printstr() {
	printableStr.Delete();
}
LUISA_EXPORT_API void pydb_ser(SimpleBinaryJson* v, char const* filePath) {
	TSer(v, filePath);
}
LUISA_EXPORT_API void pydb_dict_ser(SimpleJsonValueDict* v, char const* filePath) {
	TSer(v, filePath);
}
LUISA_EXPORT_API void pydb_arr_ser(SimpleJsonValueArray* v, char const* filePath) {
	TSer(v, filePath);
}
LUISA_EXPORT_API void pydb_deser(SimpleBinaryJson* v, char const* filePath, bool clearLast) {
	TDeser(v, filePath, clearLast);
}
LUISA_EXPORT_API void pydb_dict_deser(SimpleJsonValueDict* v, char const* filePath, bool clearLast) {
	TDeser(v, filePath, clearLast);
}
LUISA_EXPORT_API void pydb_arr_deser(SimpleJsonValueArray* v, char const* filePath, bool clearLast) {
	TDeser(v, filePath, clearLast);
}
LUISA_EXPORT_API void pydb_print(SimpleBinaryJson* v) {
	TPrint(v);
}
LUISA_EXPORT_API void pydb_dict_print(SimpleJsonValueDict* v) {
	TPrint(v);
}
LUISA_EXPORT_API void pydb_arr_print(SimpleJsonValueArray* v) {
	TPrint(v);
}
LUISA_EXPORT_API void pydb_parse(SimpleBinaryJson* v, char const* str, uint64 strLen, bool clearLast) {
	TParse(v, str, strLen, clearLast);
}
LUISA_EXPORT_API void pydb_dict_parse(SimpleJsonValueDict* v, char const* str, uint64 strLen, bool clearLast) {
	TParse(v, str, strLen, clearLast);
}
LUISA_EXPORT_API void pydb_arr_parse(SimpleJsonValueArray* v, char const* str, uint64 strLen, bool clearLast) {
	TParse(v, str, strLen, clearLast);
}

LUISA_EXPORT_API void pydb_dict_set(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType,
	void* valuePtr,
	CSharpValueType valueType) {
	dict->Set(GetCSharpKey(keyPtr, keyType), GetCSharpWriteValue(valuePtr, valueType));
}

LUISA_EXPORT_API CSharpValueType pydb_dict_get_variant(
	SimpleJsonValueDict* dict,
	void* keyPtr,
	CSharpKeyType keyType) {
	auto value = dict->Get(GetCSharpKey(keyPtr, keyType));
	return SetCSharpReadValue(&pyStruct, value);
}
LUISA_EXPORT_API void pydb_dict_remove(SimpleJsonValueDict* dict, void* keyPtr, CSharpKeyType keyType) {
	dict->Remove(GetCSharpKey(keyPtr, keyType));
}

LUISA_EXPORT_API uint pydb_dict_len(SimpleJsonValueDict* dict) { return dict->Length(); }
LUISA_EXPORT_API uint64 pydb_dict_itebegin(SimpleJsonValueDict* dict) {
	auto ite = dict->vars.begin();
	return *reinterpret_cast<uint64 const*>(&ite);
}
LUISA_EXPORT_API bool pydb_dict_iteend(SimpleJsonValueDict* dict, DictIterator end) { return (end == dict->vars.end()); }
LUISA_EXPORT_API uint64 pydb_dict_ite_next(DictIterator end) {
	end++;
	return *reinterpret_cast<uint64 const*>(&end);
}
LUISA_EXPORT_API CSharpValueType pydb_dict_ite_get_variant(DictIterator ite) {
	return SetCSharpReadValue(&pyStruct, ite->second.GetVariant());
}
LUISA_EXPORT_API CSharpKeyType pydb_dict_ite_getkey_variant(DictIterator ite) {
	return SetCSharpKey(&pyStruct, ite->first.GetKey());
}
LUISA_EXPORT_API void pydb_dict_reset(
	SimpleJsonValueDict* dict) {
	dict->Reset();
}
////////////////// Array Area
LUISA_EXPORT_API void pydb_arr_reset(
	SimpleJsonValueArray* arr) {
	arr->Reset();
}
LUISA_EXPORT_API uint pydb_arr_len(SimpleJsonValueArray* arr) {
	return arr->Length();
}
LUISA_EXPORT_API CSharpValueType pydb_arr_get_value_variant(SimpleJsonValueArray* arr, uint index) {
	return SetCSharpReadValue(&pyStruct, arr->Get(index));
}
LUISA_EXPORT_API void pydb_arr_set_value(SimpleJsonValueArray* arr, uint index, void* valuePtr, CSharpValueType valueType) {
	arr->Set(index, GetCSharpWriteValue(valuePtr, valueType));
}
LUISA_EXPORT_API void pydb_arr_add_value(SimpleJsonValueArray* arr, void* valuePtr, CSharpValueType valueType) {
	arr->Add(GetCSharpWriteValue(valuePtr, valueType));
}
LUISA_EXPORT_API void pydb_arr_remove(SimpleJsonValueArray* arr, uint index) {
	arr->Remove(index);
}

LUISA_EXPORT_API uint64 pydb_arr_itebegin(SimpleJsonValueArray* arr) {
	auto a = arr->arr.begin();
	return *reinterpret_cast<uint64 const*>(&a);
}
LUISA_EXPORT_API bool pydb_arr_iteend(SimpleJsonValueArray* arr, ArrayIterator ptr) {
	return (ptr == arr->arr.end());
}
LUISA_EXPORT_API uint64 pydb_arr_ite_next(ArrayIterator ptr) {
	(ptr)++;
	return *reinterpret_cast<uint64 const*>(&ptr);
}
LUISA_EXPORT_API CSharpValueType pydb_arr_ite_get_variant(ArrayIterator ite) {
	return SetCSharpReadValue(&pyStruct, ite->GetVariant());
}
LUISA_EXPORT_API uint64 py_get_str_size() {
	return pyStruct.strv->size();
}
LUISA_EXPORT_API void py_get_chars(char* ptr) {
	memcpy(ptr, pyStruct.strv->begin(), pyStruct.strv->size());
}
LUISA_EXPORT_API void py_bind_64(int64* ptr) {
	*ptr = *pyStruct.intValue;
}
LUISA_EXPORT_API void py_bind_128(int64* ptr) {
	int64* otherPtr = reinterpret_cast<int64*>(&pyStruct);
	ptr[0] = otherPtr[0];
	ptr[1] = otherPtr[1];
}

LUISA_EXPORT_API void py_bind_strview(
	CSharpStringView* strv,
	char* ptr,
	uint64 len) { *strv = CSharpStringView(ptr, len); }
}// namespace toolhub::db