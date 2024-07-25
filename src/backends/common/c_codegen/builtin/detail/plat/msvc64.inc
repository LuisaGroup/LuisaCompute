#pragma once
#include "./common.inc"

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t _plat_atom64;

inline static _plat_atom64 _plat64_atomic_fetch_add_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 add, skr_memory_order mo) {
  (void)(mo);
  return (_plat_atom64)_InterlockedExchangeAdd64((volatile _plat_atom64*)p, (_plat_atom64)add);
}

inline static _plat_atom64 _plat64_atomic_fetch_sub_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 sub, skr_memory_order mo) {
  (void)(mo);
  return (_plat_atom64)_InterlockedExchangeAdd64((volatile _plat_atom64*)p, -((_plat_atom64)sub));
}

inline static _plat_atom64 _plat64_atomic_fetch_and_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 x, skr_memory_order mo) {
  (void)(mo);
  return (_plat_atom64)_InterlockedAnd64((volatile _plat_atom64*)p, (_plat_atom64)x);
}

inline static _plat_atom64 _plat64_atomic_fetch_or_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 x, skr_memory_order mo) {
  (void)(mo);
  return (_plat_atom64)_InterlockedOr64((volatile _plat_atom64*)p, (_plat_atom64)x);
}

inline static _plat_atom64 _plat64_atomic_fetch_xor_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 x, skr_memory_order mo) {
  (void)(mo);
  return (_plat_atom64)_InterlockedXor64((volatile _plat_atom64*)p, (_plat_atom64)x);
}

inline static bool _plat64_atomic_compare_exchange_strong_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64* expected, _plat_atom64 desired, skr_memory_order mo1, skr_memory_order mo2) {
  (void)(mo1); (void)(mo2);
  _plat_atom64 read = (_plat_atom64)_InterlockedCompareExchange64((volatile _plat_atom64*)p, (_plat_atom64)desired, (_plat_atom64)(*expected));
  if (read == *expected) {
    return true;
  }
  else {
    *expected = read;
    return false;
  }
}

inline static bool _plat64_atomic_compare_exchange_weak_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64* expected, _plat_atom64 desired, skr_memory_order mo1, skr_memory_order mo2) {
  return _plat64_atomic_compare_exchange_strong_explicit(p, expected, desired, mo1, mo2);
}

inline static _plat_atom64 _plat64_atomic_exchange_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 exchange, skr_memory_order mo) {
  (void)(mo);
  return (_plat_atom64)_InterlockedExchange64((volatile _plat_atom64*)p, (_plat_atom64)exchange);
}

inline static void _plat64_atomic_thread_fence(skr_memory_order mo) {
  (void)(mo);
  _SAtomic(_plat_atom64) x = 0;
  _plat64_atomic_exchange_explicit(&x, 1, mo);
}

inline static _plat_atom64 _plat64_atomic_load_explicit(_SAtomic(_plat_atom64) const* p, skr_memory_order mo) {
  (void)(mo);
#if defined(_M_IX86) || defined(_M_X64)
  return *p;
#else
  _plat_atom64 x = *p;
  if (mo > skr_memory_order_relaxed) {
    while (!_plat64_atomic_compare_exchange_weak_explicit((_SAtomic(_plat_atom64)*)p, &x, x, mo, skr_memory_order_relaxed)) { /* nothing */ };
  }
  return x;
#endif
}

inline static void _plat64_atomic_store_explicit(_SAtomic(_plat_atom64)*p, _plat_atom64 x, skr_memory_order mo) {
  (void)(mo);
#if defined(_M_IX86) || defined(_M_X64)
  *p = x;
#else
  _plat64_atomic_exchange_explicit(p, x, mo);
#endif
}

inline static int64_t _plat64_atomic_loadi64_explicit(_SAtomic(int64_t)*p, skr_memory_order mo) {
  (void)(mo);
#if defined(_M_X64)
  return *p;
#else
  int64_t old = *p;
  int64_t x = old;
  while ((old = InterlockedCompareExchange64(p, x, old)) != x) {
    x = old;
  }
  return x;
#endif
}

inline static void _plat64_atomic_storei64_explicit(_SAtomic(int64_t)*p, int64_t x, skr_memory_order mo) {
  (void)(mo);
#if defined(x_M_IX86) || defined(_M_X64)
  *p = x;
#else
  InterlockedExchange64(p, x);
#endif
}

#ifdef __cplusplus
}
#endif