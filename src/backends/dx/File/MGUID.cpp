#include <File/MGuid.h>

vstd::string MGuid::GuidToFormattedString(const GUID& guid)
{
	char buf[64] = { 0 };

	sprintf_s(buf, sizeof(buf),

		"{%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",

		guid.Data1, guid.Data2, guid.Data3,

		guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],

		guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);

	return vstd::string(buf);
}

vstd::string MGuid::GuidToString(const GUID& guid)
{
	char buf[64] = { 0 };

	sprintf_s(buf, sizeof(buf),

		"%08X%04X%04X%02X%02X%02X%02X%02X%02X%02X%02X",

		guid.Data1, guid.Data2, guid.Data3,

		guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],

		guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);

	return vstd::string(buf);
}

vstd::string MGuid::GetGUIDString()
{
	GUID guid;

	HRESULT h = CoCreateGuid(&guid);
	if (h == S_OK)
		return GuidToString(guid);
	return "";
}

vstd::string MGuid::GetFormattedGUIDString()
{
	GUID guid;

	HRESULT h = CoCreateGuid(&guid);
	if (h == S_OK)
		return GuidToFormattedString(guid);
	return "";
}

MGuid::MGuid()
{
	GUID guid;
	CoCreateGuid(&guid);
	char buf[64];

	sprintf_s(buf, sizeof(buf),

		"%08X%04X%04X%02X%02X%02X%02X%02X%02X%02X%02X",

		guid.Data1, guid.Data2, guid.Data3,

		guid.Data4[0], guid.Data4[1], guid.Data4[2], guid.Data4[3],

		guid.Data4[4], guid.Data4[5], guid.Data4[6], guid.Data4[7]);
	memcpy(c, buf, 32);
}