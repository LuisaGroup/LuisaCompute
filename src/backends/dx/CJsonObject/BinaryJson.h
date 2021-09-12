#pragma once
#include <Common/Common.h>
#include <vstl/VObject.h>
#include <CJsonObject/SerializeStruct.h>
class BinaryReader;
namespace SerializeStruct {
class SerializedData;
}
class LUISA_DLL BinaryJson final : public VObject {
private:
	bool initialized = false;
	bool isArray;
	struct ArrayData {
		void* allocatedPtr;
		vstd::vector<SerializeStruct::SerializedData*> datas;
	};
	union {
		StackObject<HashMap<vstd::string, std::unique_ptr<SerializeStruct::SerializedData>>> keyValueDatas;
		StackObject<ArrayData> arrayDatas;
	};
	void DisposeSelf();
	void Parse(char const*& ptr, bool isArray);
	template<typename T>
	static T* Read(char const*& ptr) {
		T* pp = (T*)ptr;
		ptr += sizeof(T);
		return pp;
	}

public:
	template<typename IteFunc>
	void IterateKeyValue(
		IteFunc const& func) const;
	template<typename IteFunc>
	void IterateArray(
		IteFunc const& func) const;

	BinaryJson(vstd::vector<char> const& data);
	BinaryJson(vstd::string const& path);
	BinaryJson(BinaryReader& reader);
	BinaryJson(vstd::vector<vstd::string> const& paths);
	BinaryJson(char const*& ptr, bool isArray) {
		Parse(ptr, isArray);
	}
	uint64 GetArraySize() const;
	bool IsArray() const {
		return isArray;
	}
	bool Get(vstd::string const& name, vstd::string& str) const;
	bool Get(vstd::string const& name, int64& intValue) const;

	bool Get(vstd::string const& name, double& floatValue) const;
	bool Get(vstd::string const& name, bool& boolValue) const;
	bool Get(vstd::string const& name, BinaryJson*& objValue) const;
	bool ContainedKey(vstd::string const& name);
	inline bool Get(vstd::string const& name, uint& intValue) const {
		int64 v;
		if (!Get(name, v)) return false;
		intValue = v;
		return true;
	}
	inline bool Get(vstd::string const& name, int32_t& intValue) const {
		int64 v;
		if (!Get(name, v)) return false;
		intValue = v;
		return true;
	}
	inline bool Get(vstd::string const& name, float& floatValue) const {
		double v;
		if (!Get(name, v)) return false;
		floatValue = v;
		return true;
	}
	bool Get(uint64 iWitch, vstd::string& str) const;
	bool Get(uint64 iWitch, int64& intValue) const;
	bool Get(uint64 iWitch, double& floatValue) const;
	bool Get(uint64 iWitch, bool& boolValue) const;
	bool Get(uint64 iWitch, BinaryJson*& objValue) const;
	inline bool Get(uint64 iWitch, uint& intValue) const {
		int64 v;
		if (!Get(iWitch, v)) return false;
		intValue = v;
		return true;
	}
	inline bool Get(uint64 iWitch, int32_t& intValue) const {
		int64 v;
		if (!Get(iWitch, v)) return false;
		intValue = v;
		return true;
	}
	inline bool Get(uint64 iWitch, float& floatValue) const {
		double v;
		if (!Get(iWitch, v)) return false;
		floatValue = v;
		return true;
	}
	inline ~BinaryJson() {
		DisposeSelf();
	}
	VSTL_OVERRIDE_OPERATOR_NEW
	VSTL_DELETE_COPY_CONSTRUCT(BinaryJson)
};
namespace SerializeStruct {
class LUISA_DLL SerializedData {
public:
	SerializeStruct::ObjectType type;
	union {
		StackObject<vstd::string> str;
		StackObject<BinaryJson> obj;
		int64 intValue;
		double floatValue;
	};
	bool initialized = true;
	//TODO
	SerializedData() : initialized(false) {}
	SerializedData(std::nullptr_t);
	SerializedData(int64 intValue);
	SerializedData(double floatValue);
	SerializedData(bool value);
	SerializedData(char const* start, char const* end);
	SerializedData(char const*& ptr, bool isArray);
	~SerializedData();
	void Dispose();
	VSTL_OVERRIDE_OPERATOR_NEW
	VSTL_DELETE_COPY_CONSTRUCT(SerializedData)
};
}// namespace SerializeStruct

template<typename IteFunc>
inline void BinaryJson::IterateKeyValue(
	IteFunc const& func) const {
	if (!isArray) {
		keyValueDatas->IterateAll([&](vstd::string const& key, std::unique_ptr<SerializeStruct::SerializedData>& value) -> void {
			func(key, *value.get());
		});
	}
}
template<typename IteFunc>
inline void BinaryJson::IterateArray(
	IteFunc const& func) const {
	if (isArray) {
		for (uint64 key = 0; key < arrayDatas->datas.size(); ++key) {
			auto i = arrayDatas->datas[key];
			func(*i);
		}
	}
}
