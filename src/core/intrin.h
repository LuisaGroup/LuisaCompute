//
// Created by Mike Smith on 2021/3/15.
//

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#define LUISA_ARCH_X86_64
#elif defined(__aarch64__)
#define LUISA_ARCH_ARM64
#else
#error Unsupported architecture
#endif

#if defined(LUISA_ARCH_X86_64)
#include <immintrin.h>
#define LUISA_INTRIN_PAUSE() _mm_pause()
namespace luisa {
using float16_t = int16_t;
using float32x4_t = __m128;
}// namespace luisa
#elif defined(LUISA_ARCH_ARM64)
#include <arm_neon.h>
namespace luisa {
using float16_t = ::float16_t;
using float32x4_t = ::float32x4_t;
}// namespace luisa
#define LUISA_INTRIN_PAUSE() asm volatile("isb")
#else
#define LUISA_INTRIN_PAUSE() [] {}()
#endif
