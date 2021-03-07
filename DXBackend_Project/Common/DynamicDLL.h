#pragma once
#include "vstring.h"
#include "DLL.h"
#include "Memory.h"
#include <Windows.h>
#include "Log.h"
class DynamicDLL final
{
	HINSTANCE inst;
	template <typename T>
	struct GetFuncPtrFromDll;
	template <typename Ret, typename ... Args>
	struct GetFuncPtrFromDll <Ret(Args...)>
	{
		using FuncType = typename Ret(* _cdecl)(Args...);
		static FuncType Run(HINSTANCE h, LPCSTR str) noexcept
		{
			auto ptr = GetProcAddress(h, str);
			if (ptr == nullptr) {
				VEngine_Log(
					{"Can not find function ",
					 str});
				throw 0;
			}
			return (FuncType)(ptr);
		}
	};
public:
	DynamicDLL(char const* name);
	~DynamicDLL();
	template <typename T>
	typename GetFuncPtrFromDll<T>::FuncType
		GetDLLFunc(char const* str)
	{
		return GetFuncPtrFromDll<T>::Run(inst, str);
	}
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(DynamicDLL)
};