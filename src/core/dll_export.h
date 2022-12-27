#pragma once

#ifdef __cplusplus
#define LUISA_EXTERN_C extern "C"
#define LUISA_NOEXCEPT noexcept
#else
#define LUISA_EXTERN_C
#define LUISA_NOEXCEPT
#endif

#ifdef _MSC_VER
#define LUISA_FORCE_INLINE inline
#define LUISA_NEVER_INLINE __declspec(noinline)
#define LUISA_DLL
#define LUISA_EXPORT_API LUISA_EXTERN_C __declspec(dllexport)
#define LUISA_IMPORT_API LUISA_EXTERN_C __declspec(dllimport)
#else
#define LUISA_FORCE_INLINE __attribute__((always_inline, hot)) inline
#define LUISA_NEVER_INLINE __attribute__((noinline))
#define LUISA_DLL
#define LUISA_EXPORT_API LUISA_EXTERN_C __attribute__((visibility("default")))
#define LUISA_IMPORT_API LUISA_EXTERN_C
#endif

#ifdef _MSC_VER

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

#ifdef LC_COMPILE_EXPORT_DLL
#define LC_COMPILE_API __declspec(dllexport)
#else
#define LC_COMPILE_API __declspec(dllimport)
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

#ifdef LC_SERDE_LIB_EXPORT_DLL
#define LC_SERDE_LIB_API __declspec(dllexport)
#else
#define LC_SERDE_LIB_API __declspec(dllimport)
#endif

#ifdef LC_SHADERGRAPH_LIB_EXPORT_DLL
#define LC_SHADER_GRAPH_LIB_API extern "C" __declspec(dllexport)
#else
#define LC_SHADER_GRAPH_LIB_API extern "C" __declspec(dllimport)
#endif

#ifdef LC_REMOTE_EXPORT_DLL
#define LC_REMOTE_API __declspec(dllexport)
#else
#define LC_REMOTE_API __declspec(dllimport)
#endif

#else
#define LC_CORE_API
#define LC_VSTL_API
#define LC_AST_API
#define LC_RUNTIME_API
#define LC_DSL_API
#define LC_IR_API
#define LC_SERDE_LIB_API
#define LC_SHADER_GRAPH_LIB_API
#define LC_REMOTE_API
#endif
