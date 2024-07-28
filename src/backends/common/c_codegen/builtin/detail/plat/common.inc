#pragma once
#include <stdint.h>

#ifdef _WIN32
#include <intrin0.h>
#endif

#ifndef __cplusplus
#include <stdbool.h>
#else
#include <type_traits>
extern "C" {
#endif

inline static bool atomic_is_lock_free() { return true; }

typedef enum skr_memory_order {
  skr_memory_order_relaxed,
  skr_memory_order_acquire,
  skr_memory_order_consume = skr_memory_order_acquire,
  skr_memory_order_release,
  skr_memory_order_acq_rel,
  skr_memory_order_seq_cst = skr_memory_order_acq_rel
} skr_memory_order;

#define _SAtomic(tp)         volatile tp

#ifndef SATOMIC_VAR_INIT
  #define SATOMIC_VAR_INIT(x)  x
#endif

#ifdef _WIN32
inline static void skr_memorybarrier_acquire() { _ReadWriteBarrier(); }
inline static void skr_memorybarrier_release() { _ReadWriteBarrier(); }
#else
inline static void skr_memorybarrier_acquire() { __asm__ __volatile__("": : :"memory"); }
inline static void skr_memorybarrier_release() { __asm__ __volatile__("": : :"memory"); }
#endif

#ifdef __cplusplus
}
#endif