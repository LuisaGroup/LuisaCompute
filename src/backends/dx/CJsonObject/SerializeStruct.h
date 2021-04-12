#pragma once
#include <Common/Common.h>
namespace SerializeStruct
{
	enum class ObjectType : uint8_t
	{
		JsonObject = 0,
		JsonArray = 1,
		String = 2,
		False = 3,
		True = 4,
		Null= 5,
		Int = 6,
		Float = 7
	};
	struct JsonObjectHeader
	{
		uint64 keyValueCount;
	};
	struct JsonArrayHeader
	{
		uint64 elementCount;
	};
	struct StringHeader
	{
		uint64 stringLength;
	};
}