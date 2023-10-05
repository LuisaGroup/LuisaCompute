extern "C" [[noreturn]] void lc_abort(const void *, int msg) noexcept;
extern "C" [[noreturn]] void lc_abort_and_print_sll(const void *, const char *, unsigned int, unsigned int) noexcept;
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float exp10f(float x) { return powf(10.0f, x); }
inline int __clz (unsigned int x) {
    return __builtin_clz(x);
}
inline int __ctz (unsigned int x) {
    return __builtin_ctz(x);
}
inline int __clz (unsigned long long x) {
    return __builtin_clzll(x);
}
inline int __ctz (unsigned long long x) {
    return __builtin_ctzll(x);
}
inline int __ffs(unsigned int x) {
    return __builtin_ffs(x);
}
inline int __ffs(unsigned long long x) {
    return __builtin_ffsll(x);
}
inline int __popc(unsigned int x) {
    return __builtin_popcount(x);
}
inline int __brev(unsigned int x) {
    return __builtin_bswap32(x);
}
inline int __popc(unsigned long long x) {
    return __builtin_popcountll(x);
}
inline int __brev(unsigned long long x) {
    return __builtin_bswap64(x);
}

inline int __float_as_int(float x) {
    union { float f; int i; } u = { x };
    return u.i;
}

#define inf __builtin_inf()

#define __device__
#define lc_assert(cond, msg)  do { if (!(cond)) { lc_abort(k_args->internal_data, msg); } } while (false)
#define lc_unreachable(msg) { lc_abort(k_args->internal_data, msg); }
#define lc_assume(cond)
#define lc_dispatch_id() lc_make_uint3(k_args->dispatch_id[0], k_args->dispatch_id[1], k_args->dispatch_id[2])
#define lc_dispatch_size() lc_make_uint3(k_args->dispatch_size[0], k_args->dispatch_size[1], k_args->dispatch_size[2])
#define lc_thread_id() lc_make_uint3(k_args->thread_id[0], k_args->thread_id[1], k_args->thread_id[2])
#define lc_block_id() lc_make_uint3(k_args->block_id[0], k_args->block_id[1], k_args->block_id[2])
#ifdef _WIN32
#define lc_kernel extern "C" __declspec(dllexport)
#else
#define lc_kernel extern "C"
#endif
