#pragma once
#include <vstl/vstlconfig.h>
#include <vstl/vstring.h>
#include <vstl/Memory.h>
#include <Common/Log.h>
class LUISA_DLL DynamicDLL final {
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
			vstl_log(
				{"Can not find function ",
				 name});
			VSTL_ABORT();
		}
		funcPtr = reinterpret_cast<T>(ptr);
	}
	template<typename T>
	funcPtr_t<T> GetDLLFunc(char const* name) {
		static_assert(IsFuncPtr<std::remove_cvref_t<T>>::value, "DLL Only Support Function Pointer!"_sv);
		auto ptr = GetFuncPtr(name);
		if (ptr == 0) {
			vstl_log(
				{"Can not find function ",
				 name});
			VSTL_ABORT();
		}
		return reinterpret_cast<funcPtr_t<T>>(ptr);
	}
};
