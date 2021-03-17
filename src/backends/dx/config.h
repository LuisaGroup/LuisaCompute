#pragma once
#ifdef CLANG_COMPILER
#define _XM_NO_INTRINSICS_
#define m128_f32 vector4_f32
#define m128_u32 vector4_u32
#endif
#define DLL_COMMON