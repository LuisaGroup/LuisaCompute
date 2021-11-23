#pragma vengine_package vengine_database

#include <serde/config.h>
#include <vstl/config.h>
#include <serde/SimpleJsonLoader.h>
#include <serde/SimpleBinaryJson.h>
#include <serde/SimpleJsonValue.h>
namespace toolhub::db {
bool SimpleJsonLoader::Check(IJsonDatabase* parent, SimpleJsonVariant const& var) {
	bool res = false;
	switch (var.value.index()) {

		case 3:
			res = (var.value.get<3>().get() != nullptr);
			break;
		case 4:
			res = (var.value.get<4>().get() != nullptr);
			break;
		default:
			res = true;
			break;
	}

	return res;
}
SimpleJsonVariant SimpleJsonLoader::DeSerialize(std::span<uint8_t const>& arr, SimpleBinaryJson* db) {
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
SimpleJsonVariant SimpleJsonLoader::DeSerialize_DiffEnding(std::span<uint8_t const>& arr, SimpleBinaryJson* db) {
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

void SimpleJsonLoader::Serialize(SimpleJsonVariant const& v, vstd::vector<uint8_t>& data) {
	size_t dataOffset = data.size();
	data.push_back(v.value.index());
	switch (v.value.index()) {
		case WriteJsonVariant::IndexOf<int64>:
			PushDataToVector(v.value.template force_get<int64>(), data);
			break;
		case WriteJsonVariant::IndexOf<double>:
			PushDataToVector(v.value.template force_get<double>(), data);
			break;
		case WriteJsonVariant::IndexOf<vstd::string>:
			PushDataToVector(v.value.template force_get<vstd::string>(), data);
			break;
		case WriteJsonVariant::IndexOf<vstd::unique_ptr<IJsonDict>>:
			static_cast<SimpleJsonValueDict*>(v.value.template force_get<vstd::unique_ptr<IJsonDict>>().get())->M_GetSerData(data);
			break;
		case WriteJsonVariant::IndexOf<vstd::unique_ptr<IJsonArray>>:
			static_cast<SimpleJsonValueArray*>(v.value.template force_get<vstd::unique_ptr<IJsonArray>>().get())->M_GetSerData(data);
			break;
		case WriteJsonVariant::IndexOf<vstd::Guid>:
			PushDataToVector(v.value.template force_get<vstd::Guid>(), data);
			break;
		case WriteJsonVariant::IndexOf<bool>:
			PushDataToVector(v.value.template force_get<bool>(), data);
			break;
	}
}
ReadJsonVariant SimpleJsonVariant::GetVariant() const {

	switch (value.index()) {
		case WriteJsonVariant::IndexOf<int64>:
			return value.template force_get<int64>();
		case WriteJsonVariant::IndexOf<double>:
			return value.template force_get<double>();
		case WriteJsonVariant::IndexOf<vstd::string>:
			return value.template force_get<vstd::string>();
		case WriteJsonVariant::IndexOf<vstd::unique_ptr<IJsonDict>>:
			return value.template force_get<vstd::unique_ptr<IJsonDict>>().get();
		case WriteJsonVariant::IndexOf<vstd::unique_ptr<IJsonArray>>:
			return value.template force_get<vstd::unique_ptr<IJsonArray>>().get();
		case WriteJsonVariant::IndexOf<vstd::Guid>:
			return value.template force_get<vstd::Guid>();
		case WriteJsonVariant::IndexOf<bool>:
			return value.template force_get<bool>();
		default:
			return {nullptr};
	}
}

}// namespace toolhub::db