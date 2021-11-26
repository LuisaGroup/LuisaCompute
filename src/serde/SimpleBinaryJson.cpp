#pragma vengine_package vengine_database
#include <serde/config.h>
#include <serde/SimpleBinaryJson.h>
#include <serde/DatabaseInclude.h>
#include <serde/PythonLib.h>
namespace toolhub::db {
#ifdef VENGINE_PYTHON_SUPPORT
static std::mutex pyMtx;
static SimpleBinaryJson* cur_Obj = nullptr;

 SimpleBinaryJson* db_get_curobj() {
	return cur_Obj;
}
bool SimpleBinaryJson::CompileFromPython(char const* code) {
	std::lock_guard lck(pyMtx);
	auto pyLibData = py::PythonLibImpl::Current();
	cur_Obj = this;
	pyLibData->Initialize();
	auto disp = vstd::create_disposer([&]() {
		cur_Obj = nullptr;
		pyLibData->Finalize();
	});
	return pyLibData->ExecutePythonString(code);
}
#endif
////////////////// Single Thread DB
SimpleBinaryJson::SimpleBinaryJson()
	: arrValuePool(32, false), dictValuePool(32, false) {
	root.New(this);
}

IJsonDict* SimpleBinaryJson::GetRootNode() {
	return root;
}
SimpleBinaryJson ::~SimpleBinaryJson() {
	root.Delete();
}
vstd::unique_ptr<IJsonDict> SimpleBinaryJson::CreateDict() {
	return vstd::make_unique<IJsonDict>(dictValuePool.New_Lock(dictMtx, this));
}
vstd::unique_ptr<IJsonArray> SimpleBinaryJson::CreateArray() {
	return vstd::make_unique<IJsonArray>(arrValuePool.New_Lock(arrMtx, this));
}
SimpleJsonValueDict* SimpleBinaryJson::CreateDict_Nake() {
	auto ptr = dictValuePool.New_Lock(dictMtx, this);
	return ptr;
}
SimpleJsonValueArray* SimpleBinaryJson::CreateArray_Nake() {
	auto ptr = arrValuePool.New_Lock(arrMtx, this);
	return ptr;
}

vstd::vector<vstd::unique_ptr<IJsonDict>> SimpleBinaryJson::CreateDicts(size_t count) {
	vstd::vector<vstd::unique_ptr<IJsonDict>> vec;
	vec.reserve(count);
	for (auto i : vstd::range(count)) {
		vec.emplace_back(CreateDict_Nake());
	}
	return vec;
}
vstd::vector<vstd::unique_ptr<IJsonArray>> SimpleBinaryJson::CreateArrays(size_t count) {
	vstd::vector<vstd::unique_ptr<IJsonArray>> vec;
	vec.reserve(count);
	for (auto i : vstd::range(count)) {
		vec.emplace_back(CreateArray_Nake());
	}
	return vec;
}
vstd::vector<IJsonDict*> SimpleBinaryJson::CreateDicts_RawPtr(size_t count) {
	vstd::vector<IJsonDict*> vec;
	vec.reserve(count);
	for (auto i : vstd::range(count)) {
		vec.emplace_back(CreateDict_Nake());
	}
	return vec;
}
vstd::vector<IJsonArray*> SimpleBinaryJson::CreateArrays_RawPtr(size_t count) {
	vstd::vector<IJsonArray*> vec;
	vec.reserve(count);
	for (auto i : vstd::range(count)) {
		vec.emplace_back(CreateArray_Nake());
	}
	return vec;
}

IJsonDatabase* Database_Impl::CreateDatabase() const {
	return new SimpleBinaryJson();
}
IJsonDatabase* Database_Impl::CreateDatabase(
	vstd::function<void* (size_t)> const& allocFunc) const {
	return ((vstd::StackObject<SimpleBinaryJson>*)allocFunc(sizeof(SimpleBinaryJson)))->New();
}



static vstd::optional<Database_Impl> database_Impl;
VSTL_EXPORT_C toolhub::db::Database const *Database_GetFactory() {
	database_Impl.New();
	return database_Impl;
}


}// namespace toolhub::db