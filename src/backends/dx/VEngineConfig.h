#pragma once
//#define CLANG_COMPILER
#ifdef CLANG_COMPILER
#define _XM_NO_INTRINSICS_
#define m128_f32 vector4_f32
#define m128_u32 vector4_u32
#endif
#define _CONSOLE
#define _CRT_SECURE_NO_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define NOMINMAX
#define UNICODE

#ifdef NDEBUG
#define DEBUG
#endif

#ifdef _DEBUG
#define DEBUG
#endif

#define VENGINE_DLL_COMMON
#define VENGINE_DLL_RENDERER
#include <cstdlib>
#define VENGINE_EXIT throw 0