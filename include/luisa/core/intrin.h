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
#include <cstdint>
#include <cassert>

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

#include <thread>
#define LUISA_INTRIN_PAUSE() std::this_thread::yield()

#endif

////////////// assume
#ifdef NDEBUG // assume only enabled in non-debug mode.
#if defined(__clang__)// Clang
#define LUISA_ASSUME(x) (__builtin_assume(x))
#elif defined(_MSC_VER)// MSVC
#define LUISA_ASSUME(x) (__assume(x))
#else// GCC
#define LUISA_ASSUME(x) \
    if (!(x)) __builtin_unreachable()
#endif
#else
#define LUISA_ASSUME(expression) assert(expression)
#endif