#pragma once
#include <VEngineConfig.h>
#include <Common/vstring.h>
#include <Common/Memory.h>
#include <Common/Log.h>
class VENGINE_DLL_COMMON DynamicDLL final {
	size_t inst;
	template<typename T>
	struct IsFuncPtr {
		static constexpr bool value = false;
	};

	template<typename _Ret, typename... Args>
	struct IsFuncPtr<_Ret (*)(Args...)> {
		static constexpr bool value = true;
	};
	size_t GetFuncPtr(char const* name);

public:
	DynamicDLL(char const* fileName);
	~DynamicDLL();
	template<typename T>
	void GetDLLFunc(T& funcPtr, char const* name) {
		static_assert(IsFuncPtr<std::remove_cvref_t<T>>::value, "DLL Only Support Function Pointer!"_sv);
		auto ptr = GetFuncPtr(name);
		if (ptr == 0) {
			VEngine_Log(
				{"Can not find function ",
				 name});
			VENGINE_EXIT;
		}
		funcPtr = reinterpret_cast<T>(ptr);
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(DynamicDLL)
};