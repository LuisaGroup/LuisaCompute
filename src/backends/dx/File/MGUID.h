#pragma once
#include <Common/Common.h>
#include <objbase.h>
class MGuid
{
public:
	static vstd::string GetFormattedGUIDString();
	static vstd::string GetGUIDString();
private:
	static vstd::string GuidToFormattedString(const GUID& guid);
	static vstd::string GuidToString(const GUID& guid);
public:
	char c[32];
	MGuid();
	bool operator==(MGuid const& guid) const
	{
		return memcmp((char const*)this, (char const*)&guid, sizeof(MGuid)) == 0;
	}

	bool operator!=(MGuid const& guid)
	{
		return !operator==(guid);
	}
};

namespace vstd
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