#pragma once
#include "../Common/Common.h"
#include "CJsonObject.hpp"
#include "SerializeStruct.h"
class  JsonCompiler
{
	JsonCompiler() = delete;
	KILL_COPY_CONSTRUCT(JsonCompiler)
	template <typename T>
	static void PushBackData(vengine::vector<char>& vec, T const& data)
	{
		auto sz = vec.size();
		vec.resize(sz + sizeof(T));
		memcpy(vec.data() + sz	, &data, sizeof(T));
	}

	template <typename T>
	static void PushBackAll(vengine::vector<char>& vec, T const* data, size_t size)
	{
		auto sz = vec.size();
		auto dataSize = sizeof(T) * size;
		vec.resize(sz + dataSize);
		memcpy(vec.data() + sz, data, dataSize);
	}

	template <>
	static void PushBackData<vengine::string>(vengine::vector<char>& vec, vengine::string const& data)
	{
		PushBackData<uint64>(vec, data.size());
		PushBackAll<char>(vec, data.data(), data.size());
	}
public:
	static void Serialize(neb::CJsonObject& jsonObj, vengine::vector<char>& data);
};