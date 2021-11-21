#pragma once
#include <serde/IJsonObject.h>
namespace toolhub {
class JsonNetServerRpc {
public:
	// Object vstd::Type: true for dict, false for array
	static db::WriteJsonVariant JsonNetServer_Init(db::IJsonDatabase* db, db::WriteJsonVariant&& objType);
	static db::WriteJsonVariant JsonNetServer_Finalize(db::IJsonDatabase* db, db::WriteJsonVariant&& objType);
	static db::WriteJsonVariant JsonNetServer_Create(db::IJsonDatabase* db, db::WriteJsonVariant&& objType);
	static db::WriteJsonVariant JsonNetServer_Dispose(db::IJsonDatabase* db, db::WriteJsonVariant&& guid);
	static db::WriteJsonVariant JsonNetServer_Reset(db::IJsonDatabase* db, db::WriteJsonVariant&& guid);
	static db::WriteJsonVariant JsonNetServer_Reserve(db::IJsonDatabase* db, db::WriteJsonVariant&& guidSize);
	static db::WriteJsonVariant JsonNetServer_Dict_Set(db::IJsonDatabase* db, db::WriteJsonVariant&& guidKV);
	static db::WriteJsonVariant JsonNetServer_Dict_Remove(db::IJsonDatabase* db, db::WriteJsonVariant&& guidKey);
	static db::WriteJsonVariant JsonNetServer_Array_Set(db::IJsonDatabase* db, db::WriteJsonVariant&& guidIndexValue);
	static db::WriteJsonVariant JsonNetServer_Array_Remove(db::IJsonDatabase* db, db::WriteJsonVariant&& guidIndex);
	static db::WriteJsonVariant JsonNetServer_Array_Add(db::IJsonDatabase* db, db::WriteJsonVariant&& guidValue);
#ifdef VENGINE_SERVER_PROJECT
	static void Init();
#endif
};
}// namespace toolhub::db