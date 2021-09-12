#pragma once
#include <Common/Common.h>
#include <CJsonObject/CJsonObject.hpp>
#include <CJsonObject/SerializeStruct.h>
class  JsonCompiler
{
	JsonCompiler() = delete;
	VSTL_DELETE_COPY_CONSTRUCT(JsonCompiler)
	template <typename T>
	static void PushBackData(vstd::vector<char>& vec, T const& data)
	{
		auto sz = vec.size();
		vec.resize(sz + sizeof(T));
		memcpy(vec.data() + sz	, &data, sizeof(T));
	}

	template <typename T>
	static void PushBackAll(vstd::vector<char>& vec, T const* data, size_t size)
	{
		auto sz = vec.size();
		auto dataSize = sizeof(T) * size;
		vec.resize(sz + dataSize);
		memcpy(vec.data() + sz, data, dataSize);
	}

	template <>
	static void PushBackData<vstd::string>(vstd::vector<char>& vec, vstd::string const& data)
	{
		PushBackData<uint64>(vec, data.size());
		PushBackAll<char>(vec, data.data(), data.size());
	}
public:
	static void Serialize(neb::CJsonObject& jsonObj, vstd::vector<char>& data);
};