#pragma once
#include <Common/Common.h>
#include <objbase.h>
class MGuid
{
public:
	static vengine::string GetFormattedGUIDString();
	static vengine::string GetGUIDString();
private:
	static vengine::string GuidToFormattedString(const GUID& guid);
	static vengine::string GuidToString(const GUID& guid);
public:
	char c[32];
	MGuid();
	bool operator==(MGuid const& guid)
	{
		return BinaryEqualTo<MGuid>(this, &guid);
	}

	bool operator!=(MGuid const& guid)
	{
		return !operator==(guid);
	}
};

namespace vengine
{
	template <>
	struct hash<MGuid>
	{
		void operator()(MGuid const& guid) const
		{
			Hash::CharArrayHash((char const*)&guid, sizeof(MGuid));
		}
	};
}