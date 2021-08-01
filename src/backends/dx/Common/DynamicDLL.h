#pragma once
#include <VEngineConfig.h>
#include <Common/vstring.h>
#include <Common/Memory.h>
#include <Common/Log.h>
class VENGINE_DLL_COMMON DynamicDLL final : public vstd::IOperatorNewBase {
	size_t inst;
	template<typename T>
	struct IsFuncPtr {
		static constexpr bool value = false;
	};

	template<typename _Ret, typename... Args>
	struct IsFuncPtr<_Ret (*)(Args...)> {
		static constexpr bool value = true;
	};
	template<typename _Ret, typename... Args>
	struct IsFuncPtr<_Ret(Args...)> {
		static constexpr bool value = true;
	};
	size_t GetFuncPtr(char const* name);

public:
	DynamicDLL(char const* fileName);
	~DynamicDLL();
	DynamicDLL(DynamicDLL const&) = delete;
	DynamicDLL(DynamicDLL&&);
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
	template<typename T>
	funcPtr_t<T> GetDLLFunc(char const* name) {
		static_assert(IsFuncPtr<std::remove_cvref_t<T>>::value, "DLL Only Support Function Pointer!"_sv);
		auto ptr = GetFuncPtr(name);
		if (ptr == 0) {
			VEngine_Log(
				{"Can not find function ",
				 name});
			VENGINE_EXIT;
		}
		return reinterpret_cast<funcPtr_t<T>>(ptr);
	}
};