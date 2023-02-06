inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float exp10f(float x) { return powf(10.0f, x); }
inline int __clz (unsigned int x) {
    return __builtin_clz(x);
}
inline int __popc(unsigned int x) {
    return __builtin_popcount(x);
}
inline int __brev(unsigned int x) {
    return __builtin_bswap32(x);
}
inline int __clz (unsigned long long x) {
    return __builtin_clzll(x);
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

#define print_backtrace_hint()   fprintf(stderr, "set LUISA_BACKTRACE=1 for more information\n")

#define __device__
#define k_args_params    const KernelFnArgs* k_args
#define k_buffer_arg(i)     lc_buffer_arg(k_args, i)
#define k_buffer_capture(i)     lc_buffer_capture(k_args, i)
#define k_bindless_arg(i)   lc_bindless_arg(k_args, i)
#define k_bindless_capture(i)   lc_bindless_capture(k_args, i)
#define lc_assert(cond)  do { if (!(cond)) { fprintf(stderr, "Assertion failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); print_backtrace_hint(); abort(); } } while (false)
#define lc_unreachable() { fprintf(stderr, "Unreachable code at %s:%d\n", __FILE__, __LINE__); print_backtrace_hint(); abort(); }
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
