#pragma once
#include "./common.inc"

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t _plat_atom32;

#define REL(mo) if((mo) == skr_memory_order_acquire || (mo) == skr_memory_order_acq_rel) skr_memorybarrier_release()
#define ACQ(mo) if((mo) == skr_memory_order_release || (mo) == skr_memory_order_acq_rel) skr_memorybarrier_acquire()

inline static _plat_atom32 _plat32_atomic_fetch_add_explicit(_SAtomic(_plat_atom32)* p, _plat_atom32 x, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)__sync_fetch_and_add(p, x);
  ACQ(mo);
  return v;
}

inline static _plat_atom32 _plat32_atomic_fetch_sub_explicit(_SAtomic(_plat_atom32)* p, _plat_atom32 x, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)__sync_fetch_and_sub(p, x);
  ACQ(mo);
  return v;
}

inline static _plat_atom32 _plat32_atomic_fetch_and_explicit(_SAtomic(_plat_atom32)* p, _plat_atom32 x, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)__sync_fetch_and_and(p, x);
  ACQ(mo);
  return v;
}

inline static _plat_atom32 _plat32_atomic_fetch_or_explicit(_SAtomic(_plat_atom32)* p, _plat_atom32 x, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)__sync_fetch_and_or(p, x);
  ACQ(mo);
  return v;
}

inline static _plat_atom32 _plat32_atomic_fetch_xor_explicit(_SAtomic(_plat_atom32)* p, _plat_atom32 x, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)__sync_fetch_and_xor(p, x);
  ACQ(mo);
  return v;
}

inline static bool _plat32_atomic_compare_exchange_strong_explicit(_SAtomic(_plat_atom32)*p, _plat_atom32* expected, _plat_atom32 desired, skr_memory_order succ, skr_memory_order fail) {
  REL(succ);
  _plat_atom32 read = (_plat_atom32)__sync_val_compare_and_swap((volatile _plat_atom32*)p, (_plat_atom32)(*expected), (_plat_atom32)desired);
  if (read == *expected) {
    ACQ(succ);
    return true;
  }
  else {
    *expected = read;
    ACQ(fail);
    return false;
  }
}

inline static bool _plat32_atomic_compare_exchange_weak_explicit(_SAtomic(_plat_atom32)*p, _plat_atom32* expected, _plat_atom32 desired, skr_memory_order succ, skr_memory_order fail) {
  return _plat32_atomic_compare_exchange_strong_explicit(p, expected, desired, succ, fail);
}

inline static _plat_atom32 _plat32_atomic_exchange_explicit(_SAtomic(_plat_atom32)*p, _plat_atom32 exchange, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)__sync_lock_test_and_set((volatile _plat_atom32*)p, (_plat_atom32)exchange);
  ACQ(mo);
  return v;
}

inline static void _plat32_atomic_thread_fence(skr_memory_order mo) {
  REL(mo);
  _SAtomic(_plat_atom32) x = 0;
  _plat32_atomic_exchange_explicit(&x, 1, mo);
  ACQ(mo);
}

inline static _plat_atom32 _plat32_atomic_load_explicit(_SAtomic(_plat_atom32)* p, skr_memory_order mo) {
  REL(mo);
  _plat_atom32 v = (_plat_atom32)(*p);
  ACQ(mo);
  return v;
}

inline static void _plat32_atomic_store_explicit(_SAtomic(_plat_atom32)* p, _plat_atom32 x, skr_memory_order mo) {
  REL(mo);
  __sync_lock_test_and_set(p, x);
  ACQ(mo);
}

#undef REL
#undef ACQ

#ifdef __cplusplus
}
#endif