#pragma vengine_package vengine_database

#include <serialize/SimpleJsonLoader.h>
#include <serialize/SimpleBinaryJson.h>
#include <serialize/SimpleJsonValue.h>
namespace toolhub::db {
bool SimpleJsonLoader::Check(IJsonDatabase* parent, SimpleJsonVariant const& var) {
	bool res = false;
	switch (var.value.index()) {
		case 0:
		case 1:
		case 2:
		case 5:
			res = true;
			break;
		case 3:
			res = (var.value.get<3>().get() != nullptr);
			break;
		case 4:
			res = (var.value.get<4>().get() != nullptr);
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
			return SimpleJsonVariant(PopValue<std::string>(arr));
		}
		case ValueType::ValueDict: {
			auto ptr = db->dictValuePool.New(db);
			ptr->LoadFromSer(arr);
			return SimpleJsonVariant(ptr);
		}
		case ValueType::ValueArray: {
			auto ptr = db->arrValuePool.New(db);
			ptr->LoadFromSer(arr);
			return SimpleJsonVariant(ptr);
		}
		case ValueType::GUID: {
			return SimpleJsonVariant(PopValue<vstd::Guid>(arr));
		}
		default:
			return SimpleJsonVariant();
	}
}
SimpleJsonVariant SimpleJsonLoader::DeSerialize_Concurrent(std::span<uint8_t const>& arr, ConcurrentBinaryJson* db) {
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
			return SimpleJsonVariant(PopValue<std::string>(arr));
		}
		case ValueType::ValueDict: {
			auto ptr = db->dictValuePool.New_Lock(db->dictPoolMtx, db);
			ptr->LoadFromSer(arr);
			return SimpleJsonVariant(ptr);
		}
		case ValueType::ValueArray: {
			auto ptr = db->arrValuePool.New_Lock(db->arrPoolMtx, db);
			ptr->LoadFromSer(arr);
			return SimpleJsonVariant(ptr);
		}
		case ValueType::GUID: {
			return SimpleJsonVariant(PopValue<vstd::Guid>(arr));
		}
		default:
			return SimpleJsonVariant();
	}
}
void SimpleJsonLoader::Serialize(SimpleJsonVariant const& v, std::vector<uint8_t>& data) {
	size_t dataOffset = data.size();
	data.push_back(v.value.index());

	switch (v.value.index()) {
		case 0:
			PushDataToVector(v.value.get<0>(), data);
			break;
		case 1:
			PushDataToVector(v.value.get<1>(), data);
			break;
		case 2:
			PushDataToVector(v.value.get<2>(), data);
			break;
		case 5:
			PushDataToVector(v.value.get<5>(), data);
			break;
		case 3:
			static_cast<SimpleJsonValueDict*>(v.value.get<3>().get())->M_GetSerData(data);
			break;
		case 4:
			static_cast<SimpleJsonValueArray*>(v.value.get<4>().get())->M_GetSerData(data);
			break;
	}
}
void SimpleJsonLoader::Serialize_Concurrent(SimpleJsonVariant const& v, std::vector<uint8_t>& data) {
	size_t dataOffset = data.size();
	data.push_back(v.value.index());

	switch (v.value.index()) {
		case 0:
			PushDataToVector(v.value.get<0>(), data);
			break;
		case 1:
			PushDataToVector(v.value.get<1>(), data);
			break;
		case 2:
			PushDataToVector(v.value.get<2>(), data);
			break;
		case 5:
			PushDataToVector(v.value.get<5>(), data);
			break;
		case 3:
			static_cast<ConcurrentJsonValueDict*>(v.value.get<3>().get())->M_GetSerData(data);
			break;
		case 4:
			static_cast<ConcurrentJsonValueArray*>(v.value.get<4>().get())->M_GetSerData(data);
			break;
	}
}
ReadJsonVariant SimpleJsonVariant::GetVariant() const {

	switch (value.index()) {
		case 0:
			return value.get<0>();
		case 1:
			return value.get<1>();
		case 2:
			return value.get<2>();
		case 5:
			return value.get<5>();
		case 3:
			return static_cast<SimpleJsonValueDict*>(value.get<3>().get());
		case 4:
			return static_cast<SimpleJsonValueArray*>(value.get<4>().get());
		default:
			return ReadJsonVariant();
	}
}
ReadJsonVariant SimpleJsonVariant::GetVariant_Concurrent() const {
	switch (value.index()) {
		case 0:
			return value.get<0>();
		case 1:
			return value.get<1>();
		case 2:
			return value.get<2>();
		case 5:
			return value.get<5>();
		case 3:
			return static_cast<ConcurrentJsonValueDict*>(value.get<3>().get());
		case 4:
			return static_cast<ConcurrentJsonValueArray*>(value.get<4>().get());
		default:
			return ReadJsonVariant();
	}
}

}// namespace toolhub::db