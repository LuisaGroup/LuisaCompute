#pragma vengine_package vengine_database
#include <serialize/Config.h>
#ifdef VENGINE_DB_EXPORT_C
#include <serialize/SimpleBinaryJson.h>
#include <serialize/SimpleJsonValue.h>
#include <string>
namespace toolhub::db {

enum class CSharpKeyType : uint {
    Int64,
    String,
    Guid,
    None
};
enum class CSharpValueType : uint {
    Int64,
    Double,
    String,
    Dict,
    Array,
    Guid,
    None

};

Key GetCSharpKey(void *ptr, CSharpKeyType keyType) {
    switch (keyType) {
        case CSharpKeyType::Int64:
            return {*reinterpret_cast<int64 *>(ptr)};
        case CSharpKeyType::Guid:
            return {*reinterpret_cast<vstd::Guid *>(ptr)};
        case CSharpKeyType::String:
            return {*reinterpret_cast<CSharpStringView *>(ptr)};
        default:
            return {};
    }
}
void SetCSharpKey(void *ptr, CSharpKeyType keyType, Key const &key) {
    switch (keyType) {
        case CSharpKeyType::Int64: {
            *reinterpret_cast<int64 *>(ptr) =
                (key.IsTypeOf<int64>()) ? key.force_get<int64>() : 0;
        } break;
        case CSharpKeyType::Guid: {
            *reinterpret_cast<vstd::Guid *>(ptr) =
                (key.IsTypeOf<vstd::Guid>()) ? key.force_get<vstd::Guid>() : vstd::Guid(false);
        } break;
        case CSharpKeyType::String: {
            *reinterpret_cast<CSharpStringView *>(ptr) =
                (key.IsTypeOf<CSharpStringView>()) ?
                    CSharpStringView{key.force_get<std::string_view>()} :
                    CSharpStringView{};
        } break;
        default:
            VSTL_ABORT();
    }
}
CSharpKeyType SetCSharpKey(void *ptr, Key const &key) {
    CSharpKeyType keyType;
    switch (key.GetType()) {
        case Key::IndexOf<int64>:
            *reinterpret_cast<int64 *>(ptr) = key.force_get<int64>();
            keyType = CSharpKeyType::Int64;
            break;
        case Key::IndexOf<CSharpStringView>:
            *reinterpret_cast<CSharpStringView *>(ptr) = key.force_get<std::string_view>();
            keyType = CSharpKeyType::String;
            break;
        case Key::IndexOf<vstd::Guid>:
            *reinterpret_cast<vstd::Guid *>(ptr) = key.force_get<vstd::Guid>();
            keyType = CSharpKeyType::Guid;
            break;
        default:
            keyType = CSharpKeyType::None;
    }
    return keyType;
}
WriteJsonVariant GetCSharpWriteValue(void *ptr, CSharpValueType valueType) {
    switch (valueType) {
        case CSharpValueType::Array:
            return {UniquePtr<IJsonArray>(*reinterpret_cast<SimpleJsonValueArray **>(ptr))};
        case CSharpValueType::Dict:
            return {UniquePtr<IJsonDict>(*reinterpret_cast<SimpleJsonValueDict **>(ptr))};
        case CSharpValueType::Double:
            return {*reinterpret_cast<double *>(ptr)};
        case CSharpValueType::Guid:
            return {*reinterpret_cast<vstd::Guid *>(ptr)};
        case CSharpValueType::Int64:
            return {*reinterpret_cast<int64 *>(ptr)};
        case CSharpValueType::String:
            return {*reinterpret_cast<CSharpStringView *>(ptr)};
        default:
            return {};
    }
}
void SetCSharpReadValue(void *ptr, CSharpValueType valueType, ReadJsonVariant const &readValue) {
    switch (valueType) {

        case CSharpValueType::Array:
            *reinterpret_cast<SimpleJsonValueArray **>(ptr) =
                (readValue.IsTypeOf<IJsonArray *>()) ? (static_cast<SimpleJsonValueArray *>(readValue.force_get<IJsonArray *>())) : nullptr;
            break;

        case CSharpValueType::Dict:
            *reinterpret_cast<SimpleJsonValueDict **>(ptr) =
                (readValue.IsTypeOf<IJsonDict *>()) ? (static_cast<SimpleJsonValueDict *>(readValue.force_get<IJsonDict *>())) : nullptr;
            break;
        case CSharpValueType::Double:
            if (readValue.IsTypeOf<int64>()) {
                *reinterpret_cast<double *>(ptr) = readValue.force_get<int64>();
            } else if (readValue.IsTypeOf<double>()) {
                *reinterpret_cast<double *>(ptr) = readValue.force_get<double>();
            }
            break;
        case CSharpValueType::Guid:
            *reinterpret_cast<vstd::Guid *>(ptr) =
                (readValue.IsTypeOf<vstd::Guid>()) ? (readValue.force_get<vstd::Guid>()) : vstd::Guid(false);
            break;
        case CSharpValueType::Int64:
            if (readValue.IsTypeOf<int64>()) {
                *reinterpret_cast<int64 *>(ptr) = readValue.force_get<int64>();
            } else if (readValue.IsTypeOf<double>()) {
                *reinterpret_cast<int64 *>(ptr) = readValue.force_get<double>();
            }
            break;
        case CSharpValueType::String:
            *reinterpret_cast<CSharpStringView *>(ptr) =
                (readValue.IsTypeOf<CSharpStringView>()) ?
                    CSharpStringView{readValue.force_get<std::string_view>()} :
                    CSharpStringView{};
            break;
        default:
            VSTL_ABORT();
    }
}

CSharpValueType SetCSharpReadValue(void *ptr, ReadJsonVariant const &readValue) {
    CSharpValueType resultType;
    switch (readValue.GetType()) {
        case ReadJsonVariant::IndexOf<int64>:
            *reinterpret_cast<int64 *>(ptr) = readValue.force_get<int64>();
            resultType = CSharpValueType::Int64;
            break;
        case ReadJsonVariant::IndexOf<double>:
            *reinterpret_cast<double *>(ptr) = readValue.force_get<double>();
            resultType = CSharpValueType::Double;
            break;
        case ReadJsonVariant::IndexOf<CSharpStringView>:
            *reinterpret_cast<CSharpStringView *>(ptr) = readValue.force_get<std::string_view>();
            resultType = CSharpValueType::String;
            break;
        case ReadJsonVariant::IndexOf<vstd::Guid>:
            *reinterpret_cast<vstd::Guid *>(ptr) = readValue.force_get<vstd::Guid>();
            resultType = CSharpValueType::Guid;
            break;
        case ReadJsonVariant::IndexOf<IJsonDict *>:
            *reinterpret_cast<IJsonDict **>(ptr) = readValue.force_get<IJsonDict *>();
            resultType = CSharpValueType::Dict;
            break;
        case ReadJsonVariant::IndexOf<IJsonArray *>:
            *reinterpret_cast<IJsonArray **>(ptr) = readValue.force_get<IJsonArray *>();
            resultType = CSharpValueType::Array;
            break;
        default:
            resultType = CSharpValueType::None;
            break;
    }
    return resultType;
}

using DictIterator = decltype(std::declval<SimpleJsonValueDict>().vars)::Iterator;
using ArrayIterator = decltype(std::declval<SimpleJsonValueArray>().arr.begin());

}// namespace toolhub::db

//#define EXTERN_UNITY
#ifdef VENGINE_CSHARP_SUPPORT
#include <serialize/Unity_Extern.inl>
#endif

#ifdef VENGINE_PYTHON_SUPPORT
#include <serialize/Python_Extern.inl>
#endif

#endif