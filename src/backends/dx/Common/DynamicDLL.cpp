#include <Common/DynamicDLL.h>
#include <Windows.h>
#include <core/logging.h>
DynamicDLL::DynamicDLL(char const* name) {
	inst = reinterpret_cast<size_t>(LoadLibraryA(name));
	if (inst == 0) {
		LUISA_ERROR_WITH_LOCATION(
			"Can not find DLL ");
		VSTL_ABORT();
	}
}
DynamicDLL::~DynamicDLL() {
	if (inst != 0)
		FreeLibrary(reinterpret_cast<HINSTANCE>(inst));
}

DynamicDLL::DynamicDLL(DynamicDLL&& d) {
	inst = d.inst;
	d.inst = 0;
}

size_t DynamicDLL::GetFuncPtr(char const* name) {
	return reinterpret_cast<size_t>(GetProcAddress(reinterpret_cast<HINSTANCE>(inst), name));
}