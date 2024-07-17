#pragma once
#ifdef _WIN32
  #include "plat/msvc32.inc"
  #include "plat/msvc64.inc"
#else
  #include "plat/gcc32.inc"
  #include "plat/gcc64.inc"
#endif

// C11 Standard Macros

#define skr_atomic_store_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_store_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_store_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_store_explicit((volatile _plat_atom64*) px, v, mo), \
  volatile uint64_t*: _plat64_atomic_store_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_load_explicit(px, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_load_explicit((volatile _plat_atom32*)px, mo), \
  volatile uint32_t*: _plat32_atomic_load_explicit((volatile _plat_atom32*)px, mo), \
  volatile int64_t*: _plat64_atomic_load_explicit((volatile _plat_atom64*)px, mo), \
  volatile uint64_t*: _plat64_atomic_load_explicit((volatile _plat_atom64*)px, mo))

#define skr_atomic_exchange_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_exchange_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_exchange_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_exchange_explicit((volatile _plat_atom64*)px, v, mo), \
  volatile uint64_t*: _plat64_atomic_exchange_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_fetch_add_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_fetch_add_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_fetch_add_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_fetch_add_explicit((volatile _plat_atom64*)px, v, mo), \
  volatile uint64_t*: _plat64_atomic_fetch_add_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_fetch_sub_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_fetch_sub_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_fetch_sub_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_fetch_sub_explicit((volatile _plat_atom64*)px, v, mo), \
  volatile uint64_t*: _plat64_atomic_fetch_sub_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_fetch_or_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_fetch_or_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_fetch_or_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_fetch_or_explicit((volatile _plat_atom64*)px, v, mo), \
  volatile uint64_t*: _plat64_atomic_fetch_or_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_fetch_xor_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_fetch_xor_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_fetch_xor_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_fetch_xor_explicit((volatile _plat_atom64*)px, v, mo), \
  volatile uint64_t*: _plat64_atomic_fetch_xor_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_fetch_and_explicit(px, v, mo) _Generic((px), \
  volatile int32_t*: _plat32_atomic_fetch_and_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile uint32_t*: _plat32_atomic_fetch_and_explicit((volatile _plat_atom32*)px, v, mo), \
  volatile int64_t*: _plat64_atomic_fetch_and_explicit((volatile _plat_atom64*)px, v, mo), \
  volatile uint64_t*: _plat64_atomic_fetch_and_explicit((volatile _plat_atom64*)px, v, mo))

#define skr_atomic_compare_exchange_weak_explicit(px, e, v, mo1, mo2) _Generic((px), \
  volatile int32_t*: _plat32_atomic_compare_exchange_weak_explicit((volatile _plat_atom32*)px, (_plat_atom32*)e, v, mo1, mo2), \
  volatile uint32_t*: _plat32_atomic_compare_exchange_weak_explicit((volatile _plat_atom32*)px, (_plat_atom32*)e, v, mo1, mo2), \
  volatile int64_t*: _plat64_atomic_compare_exchange_weak_explicit((volatile _plat_atom64*)px, (_plat_atom64*)e, v, mo1, mo2), \
  volatile uint64_t*: _plat64_atomic_compare_exchange_weak_explicit((volatile _plat_atom64*)px, (_plat_atom64*)e, v, mo1, mo2))

#define skr_atomic_compare_exchange_strong_explicit(px, e, v, mo1, mo2) _Generic((px), \
  volatile int32_t*: _plat32_atomic_compare_exchange_strong_explicit((volatile _plat_atom32*)px, (_plat_atom32*)e, v, mo1, mo2), \
  volatile uint32_t*: _plat32_atomic_compare_exchange_strong_explicit((volatile _plat_atom32*)px, (_plat_atom32*)e, v, mo1, mo2), \
  volatile int64_t*: _plat64_atomic_compare_exchange_strong_explicit((volatile _plat_atom64*)px, (_plat_atom64*)e, v, mo1, mo2), \
  volatile uint64_t*: _plat64_atomic_compare_exchange_strong_explicit((volatile _plat_atom64*)px, (_plat_atom64*)e, v, mo1, mo2))

#define skr_atomic_store(px, v) skr_atomic_store_explicit(px, v, skr_memory_order_seq_cst)
#define skr_atomic_load(px, v) skr_atomic_load_explicit(px, skr_memory_order_seq_cst)
#define skr_atomic_exchange(px, v) skr_atomic_exchange_explicit(px, v, skr_memory_order_seq_cst)

#define skr_atomic_fetch_add(px, v) skr_atomic_fetch_add_explicit(px, v, skr_memory_order_seq_cst)
#define skr_atomic_fetch_sub(px, v) skr_atomic_fetch_sub_explicit(px, v, skr_memory_order_seq_cst)
#define skr_atomic_fetch_or(px, v) skr_atomic_fetch_or_explicit(px, v, skr_memory_order_seq_cst)
#define skr_atomic_fetch_xor(px, v) skr_atomic_fetch_xor_explicit(px, v, skr_memory_order_seq_cst)
#define skr_atomic_fetch_and(px, v) skr_atomic_fetch_and_explicit(px, v, skr_memory_order_seq_cst)

#define skr_atomic_compare_exchange_weak(px, e, v) skr_atomic_compare_exchange_weak_explicit(px, e, v, skr_memory_order_seq_cst, skr_memory_order_seq_cst)
#define skr_atomic_compare_exchange_strong(px, e, v) skr_atomic_compare_exchange_strong_explicit(px, e, v, skr_memory_order_seq_cst, skr_memory_order_seq_cst)