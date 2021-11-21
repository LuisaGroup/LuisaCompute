#pragma once
#include <serde/SimpleJsonValue.h>
#include <vstl/VGuid.h>
namespace toolhub::db {
class Database;
class SimpleBinaryJson final : public IJsonDatabase {
public:
	vstd::StackObject<SimpleJsonValueDict> root;
	//SimpleJsonValueDict root: error
	SimpleBinaryJson();
	~SimpleBinaryJson();
	VSTD_SELF_PTR;
	vstd::Pool<SimpleJsonValueArray, VEngine_AllocType::VEngine, true> arrValuePool;
	vstd::Pool<SimpleJsonValueDict, VEngine_AllocType::VEngine, true> dictValuePool;
	vstd::spin_mutex arrMtx;
	vstd::spin_mutex dictMtx;
	IJsonDict* GetRootNode() override;
	vstd::unique_ptr<IJsonDict> CreateDict() override;
	vstd::unique_ptr<IJsonArray> CreateArray() override;
	SimpleJsonValueDict* CreateDict_Nake();
	SimpleJsonValueArray* CreateArray_Nake();
	IJsonDict* CreateDict_RawPtr() override { return CreateDict_Nake(); }
	IJsonArray* CreateArray_RawPtr() override { return CreateArray_Nake(); }
	vstd::vector<vstd::unique_ptr<IJsonDict>> CreateDicts(size_t count) override;
	vstd::vector<vstd::unique_ptr<IJsonArray>> CreateArrays(size_t count) override;
	vstd::vector<IJsonDict*> CreateDicts_RawPtr(size_t count) override;
	vstd::vector<IJsonArray*> CreateArrays_RawPtr(size_t count) override;

#ifdef VENGINE_PYTHON_SUPPORT
	bool CompileFromPython(char const* code) override;
#endif
	KILL_COPY_CONSTRUCT(SimpleBinaryJson)
	KILL_MOVE_CONSTRUCT(SimpleBinaryJson)
};
}// namespace toolhub::db