#pragma once
#include <config.h>
#include "vstring.h"
#include "DLL.h"
#include "Memory.h"
#include <Windows.h>
#include "Log.h"
class DLL_COMMON DynamicDLL final {
	HINSTANCE inst;
	template<typename T>
	struct IsFuncPtr {
		static constexpr bool value = false;
	};

	template<typename _Ret, typename... Args>
	struct IsFuncPtr<_Ret (*)(Args...)> {
		static constexpr bool value = true;
	};

public:
	DynamicDLL(char const* name);
	~DynamicDLL();
	template<typename T>
	void GetDLLFunc(T& funcPtr, char const* name) {
		static_assert(IsFuncPtr<std::remove_cvref_t<T>>::value, "DLL Only Support Function Pointer!");
		auto ptr = GetProcAddress(inst, name);
		if (ptr == nullptr) {
			VEngine_Log(
				{"Can not find function ",
				 name});
			throw 0;
		}
		funcPtr = reinterpret_cast<T>(ptr);
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(DynamicDLL)
};