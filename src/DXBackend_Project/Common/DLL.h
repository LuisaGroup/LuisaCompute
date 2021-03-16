#pragma once

#define DLL_EXPORT extern "C" _declspec(dllexport)

#define VENGINE_CDECL _cdecl
#define VENGINE_STD_CALL _stdcall
#define VENGINE_VECTOR_CALL _vectorcall
#define VENGINE_FAST_CALL _fastcall