#pragma once
#define VENGINE_DB_EXPORT_C

#ifdef VENGINE_DB_EXPORT_C
#define VENGINE_PYTHON_SUPPORT
#define VENGINE_CSHARP_SUPPORT

#define LUISA_EXTERN_C_FUNC extern "C" _declspec(dllexport)

#endif