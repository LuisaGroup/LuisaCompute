//
// Created by Mike Smith on 2021/3/15.
//

#pragma once

#if defined(__x86_64__)
#include <immintrin.h>
#define LUISA_INTRIN_PAUSE() _mm_pause()
#elif defined(_M_X64)
#include <windows.h>
#define LUISA_INTRIN_PAUSE() YieldProcessor()
#elif defined(__aarch64__)
#define LUISA_INTRIN_PAUSE() asm volatile("isb")
#else
#define LUISA_INTRIN_PAUSE() []{}()
#endif
