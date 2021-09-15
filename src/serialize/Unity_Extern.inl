#pragma once
#include <serialize/SimpleBinaryJson.h>
#include <serialize/SimpleJsonValue.h>
namespace toolhub::db {

template<typename T>
void DBParse(T *db, StringView strv, vstd::funcPtr_t<void(StringView)> errorCallback, bool clearLast) {
    auto errorMsg = db->Parse(std::string_view(strv.begin(), strv.size()), clearLast);
    if (errorMsg) {
        errorCallback(errorMsg->message);
    } else {
        errorCallback(StringView(nullptr, nullptr));
    }
}
LUISA_EXTERN_C_FUNC void db_get_new(SimpleBinaryJson **pp) {
    *pp = new SimpleBinaryJson();
}
LUISA_EXTERN_C_FUNC void db_dispose(SimpleBinaryJson *p) {
    p->Dispose();
}
LUISA_EXTERN_C_FUNC void db_get_rootnode(SimpleBinaryJson *db, SimpleJsonValueDict **pp) {
    *pp = static_cast<SimpleJsonValueDict *>(db->GetRootNode());
}
LUISA_EXTERN_C_FUNC void db_create_dict(SimpleBinaryJson *db, SimpleJsonValueDict **pp) {
    *pp = db->CreateDict_Nake();
}
LUISA_EXTERN_C_FUNC void db_create_array(SimpleBinaryJson *db, SimpleJsonValueArray **pp) {
    *pp = db->CreateArray_Nake();
}
LUISA_EXTERN_C_FUNC void db_serialize(SimpleBinaryJson *db, vstd::funcPtr_t<void(uint8_t *, uint64)> callback) {
    auto vec = db->Serialize();
    callback(vec.data(), vec.size());
}
LUISA_EXTERN_C_FUNC void db_arr_ser(SimpleJsonValueArray *db, vstd::funcPtr_t<void(uint8_t *, uint64)> callback) {
    auto vec = db->Serialize();
    callback(vec.data(), vec.size());
}
LUISA_EXTERN_C_FUNC void db_dict_ser(SimpleJsonValueDict *db, vstd::funcPtr_t<void(uint8_t *, uint64)> callback) {
    auto vec = db->Serialize();
    callback(vec.data(), vec.size());
}
LUISA_EXTERN_C_FUNC void db_arr_deser(SimpleJsonValueArray *db, uint8_t *ptr, uint64 len, bool *success, bool clearLast) {
    *success = db->Read(std::span<uint8_t const>(ptr, len), clearLast);
}
LUISA_EXTERN_C_FUNC void db_dict_deser(SimpleJsonValueDict *db, uint8_t *ptr, uint64 len, bool *success, bool clearLast) {
    *success = db->Read(std::span<uint8_t const>(ptr, len), clearLast);
}
LUISA_EXTERN_C_FUNC void db_serialize_tofile(SimpleBinaryJson *db, StringView filePath) {
    auto vec = db->Serialize();
    auto file = fopen(filePath.begin(), "wb");
    if (file != nullptr) {
        auto disp = vstd::create_disposer([&]() {
            fclose(file);
        });
        fwrite(vec.data(), vec.size(), 1, file);
    }
}
LUISA_EXTERN_C_FUNC void db_deser(SimpleBinaryJson *db, uint8_t *ptr, uint64 len, bool *success, bool clearLast) {
    *success = db->Read(std::span<uint8_t const>(ptr, len), clearLast);
}
LUISA_EXTERN_C_FUNC void db_dispose_arr(SimpleJsonValueArray *p) {
    p->Dispose();
}
LUISA_EXTERN_C_FUNC void db_print(SimpleBinaryJson *db, vstd::funcPtr_t<void(StringView)> ptr) {
    ptr(db->Print());
}

LUISA_EXTERN_C_FUNC void db_parse(SimpleBinaryJson *db, StringView strv, vstd::funcPtr_t<void(StringView)> errorCallback, bool clearLast) {
    DBParse(db, strv, errorCallback, clearLast);
}
////////////////// Dict Area

LUISA_EXTERN_C_FUNC void db_dispose_dict(SimpleJsonValueDict *ptr) {
    ptr->Dispose();
}
LUISA_EXTERN_C_FUNC void db_dict_set(
    SimpleJsonValueDict *dict,
    void *keyPtr,
    CSharpKeyType keyType,
    void *valuePtr,
    CSharpValueType valueType) {
    dict->Set(GetCSharpKey(keyPtr, keyType), GetCSharpWriteValue(valuePtr, valueType));
}
LUISA_EXTERN_C_FUNC void db_dict_tryset(
    SimpleJsonValueDict *dict,
    void *keyPtr,
    CSharpKeyType keyType,
    void *valuePtr,
    CSharpValueType valueType,
    bool *isTry) {
    Key key;
    WriteJsonVariant value;
    *isTry = dict->TrySet(GetCSharpKey(keyPtr, keyType), GetCSharpWriteValue(valuePtr, valueType));
}
LUISA_EXTERN_C_FUNC void db_dict_get(
    SimpleJsonValueDict *dict,
    void *keyPtr,
    CSharpKeyType keyType,
    CSharpValueType targetValueType,
    void *valuePtr) {
    SetCSharpReadValue(valuePtr, targetValueType, dict->Get(GetCSharpKey(keyPtr, keyType)));
}
LUISA_EXTERN_C_FUNC void db_dict_get_variant(
    SimpleJsonValueDict *dict,
    void *keyPtr,
    CSharpKeyType keyType,
    CSharpValueType *targetValueType,
    void *valuePtr) {
    auto value = dict->Get(GetCSharpKey(keyPtr, keyType));
    *targetValueType = SetCSharpReadValue(valuePtr, value);
}
LUISA_EXTERN_C_FUNC void db_dict_remove(SimpleJsonValueDict *dict, void *keyPtr, CSharpKeyType keyType) {
    dict->Remove(GetCSharpKey(keyPtr, keyType));
}
LUISA_EXTERN_C_FUNC void db_dict_len(SimpleJsonValueDict *dict, int32 *sz) { *sz = dict->Length(); }
LUISA_EXTERN_C_FUNC void db_dict_itebegin(SimpleJsonValueDict *dict, DictIterator *ptr) { *ptr = dict->vars.begin(); }
LUISA_EXTERN_C_FUNC void db_dict_iteend(SimpleJsonValueDict *dict, DictIterator *end, bool *result) { *result = (*end == dict->vars.end()); }
LUISA_EXTERN_C_FUNC void db_dict_ite_next(DictIterator *end) { (*end)++; }
LUISA_EXTERN_C_FUNC void db_dict_ite_get(DictIterator ite, void *valuePtr, CSharpValueType valueType) {
    SetCSharpReadValue(valuePtr, valueType, ite->second.GetVariant());
}
LUISA_EXTERN_C_FUNC void db_dict_ite_get_variant(DictIterator ite, void *valuePtr, CSharpValueType *valueType) {
    *valueType = SetCSharpReadValue(valuePtr, ite->second.GetVariant());
}

LUISA_EXTERN_C_FUNC void db_dict_ite_getkey(DictIterator ite, void *keyPtr, CSharpKeyType keyType) {
    SetCSharpKey(keyPtr, keyType, ite->first.GetKey());
}
LUISA_EXTERN_C_FUNC void db_dict_ite_getkey_variant(DictIterator ite, void *keyPtr, CSharpKeyType *keyType) {
    *keyType = SetCSharpKey(keyPtr, ite->first.GetKey());
}
LUISA_EXTERN_C_FUNC void db_dict_reset(
    SimpleJsonValueDict *dict) {
    dict->Reset();
}
LUISA_EXTERN_C_FUNC void db_dict_parse(SimpleJsonValueDict *db, StringView strv, vstd::funcPtr_t<void(StringView)> errorCallback, bool clearLast) {
    DBParse(db, strv, errorCallback, clearLast);
}
////////////////// Array Area
LUISA_EXTERN_C_FUNC void db_arr_reset(
    SimpleJsonValueArray *arr) {
    arr->Reset();
}
LUISA_EXTERN_C_FUNC void db_arr_len(SimpleJsonValueArray *arr, int32 *sz) {
    *sz = arr->Length();
}
LUISA_EXTERN_C_FUNC void db_arr_get_value(SimpleJsonValueArray *arr, int32 index, void *valuePtr, CSharpValueType valueType) {
    SetCSharpReadValue(valuePtr, valueType, arr->Get(index));
}
LUISA_EXTERN_C_FUNC void db_arr_get_value_variant(SimpleJsonValueArray *arr, int32 index, void *valuePtr, CSharpValueType *valueType) {
    *valueType = SetCSharpReadValue(valuePtr, arr->Get(index));
}
LUISA_EXTERN_C_FUNC void db_arr_set_value(SimpleJsonValueArray *arr, int32 index, void *valuePtr, CSharpValueType valueType) {
    arr->Set(index, GetCSharpWriteValue(valuePtr, valueType));
}
LUISA_EXTERN_C_FUNC void db_arr_add_value(SimpleJsonValueArray *arr, void *valuePtr, CSharpValueType valueType) {
    arr->Add(GetCSharpWriteValue(valuePtr, valueType));
}
LUISA_EXTERN_C_FUNC void db_arr_remove(SimpleJsonValueArray *arr, int32 index) {
    arr->Remove(index);
}
LUISA_EXTERN_C_FUNC void db_arr_itebegin(SimpleJsonValueArray *arr, ArrayIterator *ptr) {
    *ptr = arr->arr.begin();
}
LUISA_EXTERN_C_FUNC void db_arr_iteend(SimpleJsonValueArray *arr, ArrayIterator *ptr, bool *result) {
    *result = (*ptr == arr->arr.end());
}
LUISA_EXTERN_C_FUNC void db_arr_ite_next(SimpleJsonValueArray *arr, ArrayIterator *ptr) {
    (*ptr)++;
}
LUISA_EXTERN_C_FUNC void db_arr_ite_get(ArrayIterator ite, void *valuePtr, CSharpValueType valueType) {
    SetCSharpReadValue(valuePtr, valueType, ite->GetVariant());
}
LUISA_EXTERN_C_FUNC void db_arr_ite_get_variant(ArrayIterator ite, void *valuePtr, CSharpValueType *valueType) {
    *valueType = SetCSharpReadValue(valuePtr, ite->GetVariant());
}
LUISA_EXTERN_C_FUNC void db_arr_parse(SimpleJsonValueArray *db, StringView strv, vstd::funcPtr_t<void(StringView)> errorCallback, bool clearLast) {
    DBParse(db, strv, errorCallback, clearLast);
}
}// namespace toolhub::db