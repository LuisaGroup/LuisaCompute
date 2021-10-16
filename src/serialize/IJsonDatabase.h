#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
#include <vstl/VGuid.h>
namespace toolhub::db {
class IJsonDict;
class IJsonArray;
class IJsonDatabase : public vstd::IDisposable {
protected:
	~IJsonDatabase() = default;

public:
	virtual IJsonDict* GetRootNode() = 0;
	virtual vstd::unique_ptr<IJsonDict> CreateDict() = 0;
	virtual vstd::unique_ptr<IJsonArray> CreateArray() = 0;
	virtual IJsonDict* CreateDict_RawPtr() = 0;
	virtual IJsonArray* CreateArray_RawPtr() = 0;
	////////// Extension
	virtual bool CompileFromPython(char const* code) {
		//Not Implemented
		return false;
	}
};

}// namespace toolhub::db