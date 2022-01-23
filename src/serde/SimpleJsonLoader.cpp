#pragma vengine_package vengine_database

#include <serde/config.h>
#include <vstl/config.h>
#include <serde/SimpleJsonLoader.h>
#include <serde/SimpleBinaryJson.h>
#include <serde/SimpleJsonValue.h>
namespace toolhub::db {
bool SimpleJsonLoader::Check(IJsonDatabase *parent, SimpleJsonVariant const &var) {
    bool res = false;
    switch (var.value.index()) {

        case 3:
            res = (eastl::get<3>(var.value).get() != nullptr);
            break;
        case 4:
            res = (eastl::get<4>(var.value).get() != nullptr);
            break;
        default:
            res = true;
            break;
    }

    return res;
}
SimpleJsonVariant SimpleJsonLoader::DeSerialize(vstd::span<uint8_t const> &arr, SimpleBinaryJson *db) {
    ValueType type = PopValue<ValueType>(arr);
    switch (type) {
        case ValueType::Int: {
            int64 v = PopValue<int64>(arr);
            return SimpleJsonVariant(v);
        }
        case ValueType::Float: {
            double d = PopValue<double>(arr);
            return SimpleJsonVariant(d);
        }

        case ValueType::String: {
            return SimpleJsonVariant(PopValue<vstd::string>(arr));
        }
        case ValueType::ValueDict: {
            auto ptr = db->CreateDict_Nake();
            ptr->LoadFromSer(arr);
            return SimpleJsonVariant(vstd::make_unique<IJsonDict>(ptr));
        }
        case ValueType::ValueArray: {
            auto ptr = db->CreateArray_Nake();
            ptr->LoadFromSer(arr);
            return SimpleJsonVariant(vstd::make_unique<IJsonArray>(ptr));
        }
        case ValueType::GUID: {
            return SimpleJsonVariant(PopValue<vstd::Guid>(arr));
        }
        case ValueType::Bool: {
            return {PopValue<bool>(arr)};
        }
        default:
            return {};
    }
}
SimpleJsonVariant SimpleJsonLoader::DeSerialize_DiffEnding(vstd::span<uint8_t const> &arr, SimpleBinaryJson *db) {
    ValueType type = PopValueReverse<ValueType>(arr);
    switch (type) {
        case ValueType::Int: {
            int64 v = PopValueReverse<int64>(arr);
            return SimpleJsonVariant(v);
        }
        case ValueType::Float: {
            double d = PopValueReverse<double>(arr);
            return SimpleJsonVariant(d);
        }

        case ValueType::String: {
            return SimpleJsonVariant(PopValueReverse<vstd::string>(arr));
        }
        case ValueType::ValueDict: {
            auto ptr = db->CreateDict_Nake();
            ptr->LoadFromSer_DiffEnding(arr);
            return SimpleJsonVariant(vstd::make_unique<IJsonDict>(ptr));
        }
        case ValueType::ValueArray: {
            auto ptr = db->CreateArray_Nake();
            ptr->LoadFromSer_DiffEnding(arr);
            return SimpleJsonVariant(vstd::make_unique<IJsonArray>(ptr));
        }
        case ValueType::GUID: {
            auto guid = PopValueReverse<vstd::Guid::GuidData>(arr);
            return SimpleJsonVariant(guid);
        }
        case ValueType::Bool: {
            return {PopValueReverse<bool>(arr)};
        }
        default:
            return {};
    }
}

void SimpleJsonLoader::Serialize(SimpleJsonVariant const &v, vstd::vector<uint8_t> &data) {
    size_t dataOffset = data.size();
    data.push_back(v.value.index());
    switch (v.value.index()) {
        case WriteVar::IndexOf<int64>:
            PushDataToVector(eastl::get<int64>(v.value), data);
            break;
        case WriteVar::IndexOf<double>:
            PushDataToVector(eastl::get<double>(v.value), data);
            break;
        case WriteVar::IndexOf<vstd::string>:
            PushDataToVector(eastl::get<vstd::string>(v.value), data);
            break;
        case WriteVar::IndexOf<vstd::unique_ptr<IJsonDict>>:
            static_cast<SimpleJsonValueDict *>(eastl::get<vstd::unique_ptr<IJsonDict>>(v.value).get())->M_GetSerData(data);
            break;
        case WriteVar::IndexOf<vstd::unique_ptr<IJsonArray>>:
            static_cast<SimpleJsonValueArray *>(eastl::get<vstd::unique_ptr<IJsonArray>>(v.value).get())->M_GetSerData(data);
            break;
        case WriteVar::IndexOf<vstd::Guid>:
            PushDataToVector(eastl::get<vstd::Guid>(v.value), data);
            break;
        case WriteVar::IndexOf<bool>:
            PushDataToVector(eastl::get<bool>(v.value), data);
            break;
    }
}
ReadJsonVariant SimpleJsonVariant::GetVariant() const {

    using WriteVar = SimpleJsonLoader::WriteVar;
    switch (value.index()) {
        case WriteVar::IndexOf<int64>:
            return eastl::get<int64>(value);
        case WriteVar::IndexOf<double>:
            return eastl::get<double>(value);
        case WriteVar::IndexOf<vstd::string>:
            return eastl::get<vstd::string>(value);
        case WriteVar::IndexOf<vstd::unique_ptr<IJsonDict>>:
            return eastl::get<vstd::unique_ptr<IJsonDict>>(value).get();
        case WriteVar::IndexOf<vstd::unique_ptr<IJsonArray>>:
            return eastl::get<vstd::unique_ptr<IJsonArray>>(value).get();
        case WriteVar::IndexOf<vstd::Guid>:
            return eastl::get<vstd::Guid>(value);
        case WriteVar::IndexOf<bool>:
            return eastl::get<bool>(value);
        default:
            return {nullptr};
    }
}

}// namespace toolhub::db