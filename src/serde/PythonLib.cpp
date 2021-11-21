#pragma vengine_package vengine_database
#include <serde/config.h>
#include <vstl/config.h>

#ifdef VENGINE_PYTHON_SUPPORT
#include <serde/PythonLib.h>
#include <vstl/DynamicDLL.h>
//#include <serde/PythonInclude/Python.h>
namespace toolhub::py {

bool pyInitialized = false;
vstd::optional<DynamicDLL> pyDll;
struct PyFuncs {
	vstd::funcPtr_t<void()> Py_Initialize;
	vstd::funcPtr_t<void()> Py_Finalize;
	vstd::funcPtr_t<void*(char const*)> PyUnicode_FromString;
	vstd::funcPtr_t<void*(void*)> PyImport_Import;
	vstd::funcPtr_t<int(char const*, void*)> PyRun_SimpleStringFlags;
	vstd::funcPtr_t<void*(void*, char const*)> PyObject_GetAttrString;
	vstd::funcPtr_t<void*(void*, void*)> PyObject_CallObject;
	void Init(DynamicDLL* dll) {
		dll->GetDLLFunc(Py_Initialize, "Py_Initialize");
		dll->GetDLLFunc(Py_Finalize, "Py_Finalize");
		dll->GetDLLFunc(PyUnicode_FromString, "PyUnicode_FromString");
		dll->GetDLLFunc(PyImport_Import, "PyImport_Import");
		dll->GetDLLFunc(PyRun_SimpleStringFlags, "PyRun_SimpleStringFlags");
		dll->GetDLLFunc(PyObject_CallObject, "PyObject_CallObject");
		dll->GetDLLFunc(PyObject_GetAttrString, "PyObject_GetAttrString");
	}
};
static PyFuncs pyFuncs;
PythonLibImpl::PythonLibImpl() {
	pyDll.New("Python39.dll");
	pyFuncs.Init(pyDll);
}
PythonLibImpl::~PythonLibImpl() {
	if (pyInitialized) {
		pyFuncs.Py_Finalize();
		pyDll.Delete();
	}
}
void PythonLibImpl::Initialize() {
	if (pyInitialized) return;
	pyFuncs.Py_Initialize();
	pyInitialized = true;
}
void PythonLibImpl::Finalize() {
	if (!pyInitialized) return;
	pyFuncs.Py_Finalize();
	pyInitialized = false;
}
bool PythonLibImpl::ExecutePythonString(char const* c_str) {
	if (!pyInitialized) return false;
	pyFuncs.PyRun_SimpleStringFlags(c_str, nullptr);
	return true;
}
bool PythonLibImpl::ExecuteFunc(
	char const* moduleName,
	char const* funcName) {
	if (!pyInitialized) return false;
	auto pName = pyFuncs.PyUnicode_FromString(moduleName);
	auto pModule = pyFuncs.PyImport_Import(pName);
	auto pFunc = pyFuncs.PyObject_GetAttrString(pModule, funcName);
	pyFuncs.PyObject_CallObject(pFunc, nullptr);
	return true;
}

vstd::optional<PythonLibImpl> lib;
PythonLibImpl* PythonLibImpl::Current() {
	lib.New();
	return lib;
}
}// namespace toolhub::py
#endif