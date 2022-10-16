#pragma once

#if defined(_MSC_VER)

#ifdef LC_CORE_EXPORT_DLL
#define LC_CORE_API __declspec(dllexport)
#else
#define LC_CORE_API __declspec(dllimport)
#endif

#ifdef LC_VSTL_EXPORT_DLL
#define LC_VSTL_API __declspec(dllexport)
#else
#define LC_VSTL_API __declspec(dllimport)
#endif

#ifdef LC_AST_EXPORT_DLL
#define LC_AST_API __declspec(dllexport)
#else
#define LC_AST_API __declspec(dllimport)
#endif

#ifdef LC_RUNTIME_EXPORT_DLL
#define LC_RUNTIME_API __declspec(dllexport)
#else
#define LC_RUNTIME_API __declspec(dllimport)
#endif

#ifdef LC_DSL_EXPORT_DLL
#define LC_DSL_API __declspec(dllexport)
#else
#define LC_DSL_API __declspec(dllimport)
#endif

#ifdef LC_IR_EXPORT_DLL
#define LC_IR_API __declspec(dllexport)
#else
#define LC_IR_API __declspec(dllimport)
#endif

#ifdef LC_RTX_EXPORT_DLL
#define LC_RTX_API __declspec(dllexport)
#else
#define LC_RTX_API __declspec(dllimport)
#endif

#else
#define LC_VSTL_API
#define LC_AST_API
#define LC_RUNTIME_API
#define LC_CORE_API
#define LC_DSL_API
#define LC_RTX_API
#endif
