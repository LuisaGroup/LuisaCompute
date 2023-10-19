//
// Created by Mike on 10/19/2023.
//

#ifndef __CUDA_FP16_H__// avoid including this header if a future NVRTC version brings it by default.

struct alignas(2) __half_raw {
    unsigned short x;
};

struct alignas(4) __half2_raw {
    unsigned short x;
    unsigned short y;
};

struct __half;
struct __half2;

__device__ inline __half __double2half(const double a);
__device__ inline __half __float2half(const float a);
__device__ inline __half __float2half_rn(const float a);
__device__ inline __half __float2half_rz(const float a);
__device__ inline __half __float2half_rd(const float a);
__device__ inline __half __float2half_ru(const float a);
__device__ inline float __half2float(const __half a);
__device__ inline __half2 __float2half2_rn(const float a);
__device__ inline __half2 __floats2half2_rn(const float a, const float b);
__device__ inline float __low2float(const __half2 a);
__device__ inline float __high2float(const __half2 a);
__device__ inline signed char __half2char_rz(const __half h);
__device__ inline unsigned char __half2uchar_rz(const __half h);
__device__ inline short int __half2short_rz(const __half h);
__device__ inline unsigned short int __half2ushort_rz(const __half h);
__device__ inline int __half2int_rz(const __half h);
__device__ inline unsigned int __half2uint_rz(const __half h);
__device__ inline long long int __half2ll_rz(const __half h);
__device__ inline unsigned long long int __half2ull_rz(const __half h);
__device__ inline __half2 make_half2(const __half x, const __half y);
__device__ inline __half2 __float22half2_rn(const float2 a);
__device__ inline float2 __half22float2(const __half2 a);
__device__ inline int __half2int_rn(const __half h);
__device__ inline int __half2int_rd(const __half h);
__device__ inline int __half2int_ru(const __half h);
__device__ inline __half __int2half_rn(const int i);
__device__ inline __half __int2half_rz(const int i);
__device__ inline __half __int2half_rd(const int i);
__device__ inline __half __int2half_ru(const int i);
__device__ inline short int __half2short_rn(const __half h);
__device__ inline short int __half2short_rd(const __half h);
__device__ inline short int __half2short_ru(const __half h);
__device__ inline __half __short2half_rn(const short int i);
__device__ inline __half __short2half_rz(const short int i);
__device__ inline __half __short2half_rd(const short int i);
__device__ inline __half __short2half_ru(const short int i);
__device__ inline unsigned int __half2uint_rn(const __half h);
__device__ inline unsigned int __half2uint_rd(const __half h);
__device__ inline unsigned int __half2uint_ru(const __half h);
__device__ inline __half __uint2half_rn(const unsigned int i);
__device__ inline __half __uint2half_rz(const unsigned int i);
__device__ inline __half __uint2half_rd(const unsigned int i);
__device__ inline __half __uint2half_ru(const unsigned int i);
__device__ inline unsigned short int __half2ushort_rn(const __half h);
__device__ inline unsigned short int __half2ushort_rd(const __half h);
__device__ inline unsigned short int __half2ushort_ru(const __half h);
__device__ inline __half __ushort2half_rn(const unsigned short int i);
__device__ inline __half __ushort2half_rz(const unsigned short int i);
__device__ inline __half __ushort2half_rd(const unsigned short int i);
__device__ inline __half __ushort2half_ru(const unsigned short int i);
__device__ inline unsigned long long int __half2ull_rn(const __half h);
__device__ inline unsigned long long int __half2ull_rd(const __half h);
__device__ inline unsigned long long int __half2ull_ru(const __half h);
__device__ inline __half __ull2half_rn(const unsigned long long int i);
__device__ inline __half __ull2half_rz(const unsigned long long int i);
__device__ inline __half __ull2half_rd(const unsigned long long int i);
__device__ inline __half __ull2half_ru(const unsigned long long int i);
__device__ inline long long int __half2ll_rn(const __half h);
__device__ inline long long int __half2ll_rd(const __half h);
__device__ inline long long int __half2ll_ru(const __half h);
__device__ inline __half __ll2half_rn(const long long int i);
__device__ inline __half __ll2half_rz(const long long int i);
__device__ inline __half __ll2half_rd(const long long int i);
__device__ inline __half __ll2half_ru(const long long int i);
__device__ inline __half htrunc(const __half h);
__device__ inline __half hceil(const __half h);
__device__ inline __half hfloor(const __half h);
__device__ inline __half hrint(const __half h);
__device__ inline __half2 h2trunc(const __half2 h);
__device__ inline __half2 h2ceil(const __half2 h);
__device__ inline __half2 h2floor(const __half2 h);
__device__ inline __half2 h2rint(const __half2 h);
__device__ inline __half2 __half2half2(const __half a);
__device__ inline __half2 __lowhigh2highlow(const __half2 a);
__device__ inline __half2 __lows2half2(const __half2 a, const __half2 b);
__device__ inline __half2 __highs2half2(const __half2 a, const __half2 b);
__device__ inline __half __high2half(const __half2 a);
__device__ inline __half __low2half(const __half2 a);
__device__ inline int __hisinf(const __half a);
__device__ inline __half2 __halves2half2(const __half a, const __half b);
__device__ inline __half2 __low2half2(const __half2 a);
__device__ inline __half2 __high2half2(const __half2 a);
__device__ inline short int __half_as_short(const __half h);
__device__ inline unsigned short int __half_as_ushort(const __half h);
__device__ inline __half __short_as_half(const short int i);
__device__ inline __half __ushort_as_half(const unsigned short int i);
__device__ inline __half __hmax(const __half a, const __half b);
__device__ inline __half __hmin(const __half a, const __half b);
__device__ inline __half2 __hmax2(const __half2 a, const __half2 b);
__device__ inline __half2 __hmin2(const __half2 a, const __half2 b);
__device__ inline __half2 __shfl_sync(const unsigned mask, const __half2 var, const int delta, const int width = warpSize);
__device__ inline __half2 __shfl_up_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width = warpSize);
__device__ inline __half2 __shfl_down_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width = warpSize);
__device__ inline __half2 __shfl_xor_sync(const unsigned mask, const __half2 var, const int delta, const int width = warpSize);
__device__ inline __half __shfl_sync(const unsigned mask, const __half var, const int delta, const int width = warpSize);
__device__ inline __half __shfl_up_sync(const unsigned mask, const __half var, const unsigned int delta, const int width = warpSize);
__device__ inline __half __shfl_down_sync(const unsigned mask, const __half var, const unsigned int delta, const int width = warpSize);
__device__ inline __half __shfl_xor_sync(const unsigned mask, const __half var, const int delta, const int width = warpSize);
__device__ inline __half2 __ldg(const __half2 *const ptr);
__device__ inline __half __ldg(const __half *const ptr);
__device__ inline __half2 __ldcg(const __half2 *const ptr);
__device__ inline __half __ldcg(const __half *const ptr);
__device__ inline __half2 __ldca(const __half2 *const ptr);
__device__ inline __half __ldca(const __half *const ptr);
__device__ inline __half2 __ldcs(const __half2 *const ptr);
__device__ inline __half __ldcs(const __half *const ptr);
__device__ inline __half2 __ldlu(const __half2 *const ptr);
__device__ inline __half __ldlu(const __half *const ptr);
__device__ inline __half2 __ldcv(const __half2 *const ptr);
__device__ inline __half __ldcv(const __half *const ptr);
__device__ inline void __stwb(__half2 *const ptr, const __half2 value);
__device__ inline void __stwb(__half *const ptr, const __half value);
__device__ inline void __stcg(__half2 *const ptr, const __half2 value);
__device__ inline void __stcg(__half *const ptr, const __half value);
__device__ inline void __stcs(__half2 *const ptr, const __half2 value);
__device__ inline void __stcs(__half *const ptr, const __half value);
__device__ inline void __stwt(__half2 *const ptr, const __half2 value);
__device__ inline void __stwt(__half *const ptr, const __half value);
__device__ inline __half2 __heq2(const __half2 a, const __half2 b);
__device__ inline __half2 __hne2(const __half2 a, const __half2 b);
__device__ inline __half2 __hle2(const __half2 a, const __half2 b);
__device__ inline __half2 __hge2(const __half2 a, const __half2 b);
__device__ inline __half2 __hlt2(const __half2 a, const __half2 b);
__device__ inline __half2 __hgt2(const __half2 a, const __half2 b);
__device__ inline __half2 __hequ2(const __half2 a, const __half2 b);
__device__ inline __half2 __hneu2(const __half2 a, const __half2 b);
__device__ inline __half2 __hleu2(const __half2 a, const __half2 b);
__device__ inline __half2 __hgeu2(const __half2 a, const __half2 b);
__device__ inline __half2 __hltu2(const __half2 a, const __half2 b);
__device__ inline __half2 __hgtu2(const __half2 a, const __half2 b);
__device__ inline unsigned __heq2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hne2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hle2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hge2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hlt2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hgt2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hequ2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hneu2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hleu2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hgeu2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hltu2_mask(const __half2 a, const __half2 b);
__device__ inline unsigned __hgtu2_mask(const __half2 a, const __half2 b);
__device__ inline __half2 __hisnan2(const __half2 a);
__device__ inline __half2 __hadd2(const __half2 a, const __half2 b);
__device__ inline __half2 __hsub2(const __half2 a, const __half2 b);
__device__ inline __half2 __hmul2(const __half2 a, const __half2 b);
__device__ inline __half2 __hadd2_rn(const __half2 a, const __half2 b);
__device__ inline __half2 __hsub2_rn(const __half2 a, const __half2 b);
__device__ inline __half2 __hmul2_rn(const __half2 a, const __half2 b);
__device__ inline __half2 __h2div(const __half2 a, const __half2 b);
__device__ inline __half2 __habs2(const __half2 a);
__device__ inline __half2 __hadd2_sat(const __half2 a, const __half2 b);
__device__ inline __half2 __hsub2_sat(const __half2 a, const __half2 b);
__device__ inline __half2 __hmul2_sat(const __half2 a, const __half2 b);
__device__ inline __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
__device__ inline __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
__device__ inline __half2 __hneg2(const __half2 a);
__device__ inline __half __habs(const __half a);
__device__ inline __half __hadd(const __half a, const __half b);
__device__ inline __half __hsub(const __half a, const __half b);
__device__ inline __half __hmul(const __half a, const __half b);
__device__ inline __half __hadd_rn(const __half a, const __half b);
__device__ inline __half __hsub_rn(const __half a, const __half b);
__device__ inline __half __hmul_rn(const __half a, const __half b);
__device__ inline __half __hdiv(const __half a, const __half b);
__device__ inline __half __hadd_sat(const __half a, const __half b);
__device__ inline __half __hsub_sat(const __half a, const __half b);
__device__ inline __half __hmul_sat(const __half a, const __half b);
__device__ inline __half __hfma(const __half a, const __half b, const __half c);
__device__ inline __half __hfma_sat(const __half a, const __half b, const __half c);
__device__ inline __half __hneg(const __half a);
__device__ inline bool __hbeq2(const __half2 a, const __half2 b);
__device__ inline bool __hbne2(const __half2 a, const __half2 b);
__device__ inline bool __hble2(const __half2 a, const __half2 b);
__device__ inline bool __hbge2(const __half2 a, const __half2 b);
__device__ inline bool __hblt2(const __half2 a, const __half2 b);
__device__ inline bool __hbgt2(const __half2 a, const __half2 b);
__device__ inline bool __hbequ2(const __half2 a, const __half2 b);
__device__ inline bool __hbneu2(const __half2 a, const __half2 b);
__device__ inline bool __hbleu2(const __half2 a, const __half2 b);
__device__ inline bool __hbgeu2(const __half2 a, const __half2 b);
__device__ inline bool __hbltu2(const __half2 a, const __half2 b);
__device__ inline bool __hbgtu2(const __half2 a, const __half2 b);
__device__ inline bool __heq(const __half a, const __half b);
__device__ inline bool __hne(const __half a, const __half b);
__device__ inline bool __hle(const __half a, const __half b);
__device__ inline bool __hge(const __half a, const __half b);
__device__ inline bool __hlt(const __half a, const __half b);
__device__ inline bool __hgt(const __half a, const __half b);
__device__ inline bool __hequ(const __half a, const __half b);
__device__ inline bool __hneu(const __half a, const __half b);
__device__ inline bool __hleu(const __half a, const __half b);
__device__ inline bool __hgeu(const __half a, const __half b);
__device__ inline bool __hltu(const __half a, const __half b);
__device__ inline bool __hgtu(const __half a, const __half b);
__device__ inline bool __hisnan(const __half a);
__device__ inline __half __hmax_nan(const __half a, const __half b);
__device__ inline __half __hmin_nan(const __half a, const __half b);
__device__ inline __half __hfma_relu(const __half a, const __half b, const __half c);
__device__ inline __half2 __hmax2_nan(const __half2 a, const __half2 b);
__device__ inline __half2 __hmin2_nan(const __half2 a, const __half2 b);
__device__ inline __half2 __hfma2_relu(const __half2 a, const __half2 b, const __half2 c);
__device__ inline __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c);
__device__ inline __half hsqrt(const __half a);
__device__ inline __half hrsqrt(const __half a);
__device__ inline __half hrcp(const __half a);
__device__ inline __half hlog(const __half a);
__device__ inline __half hlog2(const __half a);
__device__ inline __half hlog10(const __half a);
__device__ inline __half hexp(const __half a);
__device__ inline __half hexp2(const __half a);
__device__ inline __half hexp10(const __half a);
__device__ inline __half hcos(const __half a);
__device__ inline __half hsin(const __half a);
__device__ inline __half2 h2sqrt(const __half2 a);
__device__ inline __half2 h2rsqrt(const __half2 a);
__device__ inline __half2 h2rcp(const __half2 a);
__device__ inline __half2 h2log(const __half2 a);
__device__ inline __half2 h2log2(const __half2 a);
__device__ inline __half2 h2log10(const __half2 a);
__device__ inline __half2 h2exp(const __half2 a);
__device__ inline __half2 h2exp2(const __half2 a);
__device__ inline __half2 h2exp10(const __half2 a);
__device__ inline __half2 h2cos(const __half2 a);
__device__ inline __half2 h2sin(const __half2 a);
__device__ inline __half2 atomicAdd(__half2 *const address, const __half2 val);
__device__ inline __half atomicAdd(__half *const address, const __half val);

struct alignas(2) __half {

protected:
    unsigned short __x{};

public:
    __half() = default;
    __device__ constexpr __half(const __half_raw &hr) : __x(hr.x) {}
    __device__ __half &operator=(const __half_raw &hr) {
        __x = hr.x;
        return *this;
    }
    __device__ volatile __half &operator=(const __half_raw &hr) volatile {
        __x = hr.x;
        return *this;
    }
    __device__ volatile __half &operator=(const volatile __half_raw &hr) volatile {
        __x = hr.x;
        return *this;
    }
    __device__ operator __half_raw() const {
        __half_raw ret;
        ret.x = __x;
        return ret;
    }
    __device__ operator __half_raw() const volatile {
        __half_raw ret;
        ret.x = __x;
        return ret;
    }
    __device__ __half(const float f) { __x = __float2half(f).__x; }
    __device__ __half(const double f) { __x = __double2half(f).__x; }
    __device__ operator float() const { return __half2float(*this); }
    __device__ __half &operator=(const float f) {
        __x = __float2half(f).__x;
        return *this;
    }
    __device__ __half &operator=(const double f) {
        __x = __double2half(f).__x;
        return *this;
    }
    __device__ __half(const short val) { __x = __short2half_rn(val).__x; }
    __device__ __half(const unsigned short val) { __x = __ushort2half_rn(val).__x; }
    __device__ __half(const int val) { __x = __int2half_rn(val).__x; }
    __device__ __half(const unsigned int val) { __x = __uint2half_rn(val).__x; }
    __device__ __half(const long val) {
        if (sizeof(long) == sizeof(long long)) {
            __x = __ll2half_rn(static_cast<long long>(val)).__x;
        } else {
            __x = __int2half_rn(static_cast<int>(val)).__x;
        }
    }
    __device__ __half(const unsigned long val) {
        if (sizeof(unsigned long) == sizeof(unsigned long long)) {
            __x = __ull2half_rn(static_cast<unsigned long long>(val)).__x;
        } else {
            __x = __uint2half_rn(static_cast<unsigned int>(val)).__x;
        }
    }
    __device__ __half(const long long val) { __x = __ll2half_rn(val).__x; }
    __device__ __half(const unsigned long long val) { __x = __ull2half_rn(val).__x; }
    __device__ operator signed char() const { return __half2char_rz(*this); }
    __device__ operator unsigned char() const { return __half2uchar_rz(*this); }
    __device__ operator char() const {
        char value;
        if (((char)-1) < (char)0) {
            value = static_cast<char>(__half2char_rz(*this));
        } else {
            value = static_cast<char>(__half2uchar_rz(*this));
        }
        return value;
    }
    __device__ operator short() const { return __half2short_rz(*this); }
    __device__ operator unsigned short() const { return __half2ushort_rz(*this); }
    __device__ operator int() const { return __half2int_rz(*this); }
    __device__ operator unsigned int() const { return __half2uint_rz(*this); }
    __device__ operator long() const {
        long retval;
        if (sizeof(long) == sizeof(long long)) {
            retval = static_cast<long>(__half2ll_rz(*this));
        } else {
            retval = static_cast<long>(__half2int_rz(*this));
        }
        return retval;
    }
    __device__ operator unsigned long() const {
        unsigned long retval;
        if (sizeof(unsigned long) == sizeof(unsigned long long)) {
            retval = static_cast<unsigned long>(__half2ull_rz(*this));
        } else {
            retval = static_cast<unsigned long>(__half2uint_rz(*this));
        }
        return retval;
    }
    __device__ operator long long() const { return __half2ll_rz(*this); }
    __device__ operator unsigned long long() const { return __half2ull_rz(*this); }
    __device__ __half &operator=(const short val) {
        __x = __short2half_rn(val).__x;
        return *this;
    }
    __device__ __half &operator=(const unsigned short val) {
        __x = __ushort2half_rn(val).__x;
        return *this;
    }
    __device__ __half &operator=(const int val) {
        __x = __int2half_rn(val).__x;
        return *this;
    }
    __device__ __half &operator=(const unsigned int val) {
        __x = __uint2half_rn(val).__x;
        return *this;
    }
    __device__ __half &operator=(const long long val) {
        __x = __ll2half_rn(val).__x;
        return *this;
    }
    __device__ __half &operator=(const unsigned long long val) {
        __x = __ull2half_rn(val).__x;
        return *this;
    }
    __device__ constexpr operator bool() const { return (__x & 0x7FFFU) != 0U; }
};
__device__ inline __half operator+(const __half &lh, const __half &rh) { return __hadd(lh, rh); }
__device__ inline __half operator-(const __half &lh, const __half &rh) { return __hsub(lh, rh); }
__device__ inline __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }
__device__ inline __half operator/(const __half &lh, const __half &rh) { return __hdiv(lh, rh); }
__device__ inline __half &operator+=(__half &lh, const __half &rh) {
    lh = __hadd(lh, rh);
    return lh;
}
__device__ inline __half &operator-=(__half &lh, const __half &rh) {
    lh = __hsub(lh, rh);
    return lh;
}
__device__ inline __half &operator*=(__half &lh, const __half &rh) {
    lh = __hmul(lh, rh);
    return lh;
}
__device__ inline __half &operator/=(__half &lh, const __half &rh) {
    lh = __hdiv(lh, rh);
    return lh;
}
__device__ inline __half &operator++(__half &h) {
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return h;
}
__device__ inline __half &operator--(__half &h) {
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return h;
}
__device__ inline __half operator++(__half &h, const int ignored) {
    static_cast<void>(ignored);
    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return ret;
}
__device__ inline __half operator--(__half &h, const int ignored) {
    static_cast<void>(ignored);
    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return ret;
}
__device__ inline __half operator+(const __half &h) { return h; }
__device__ inline __half operator-(const __half &h) { return __hneg(h); }
__device__ inline bool operator==(const __half &lh, const __half &rh) { return __heq(lh, rh); }
__device__ inline bool operator!=(const __half &lh, const __half &rh) { return __hneu(lh, rh); }
__device__ inline bool operator>(const __half &lh, const __half &rh) { return __hgt(lh, rh); }
__device__ inline bool operator<(const __half &lh, const __half &rh) { return __hlt(lh, rh); }
__device__ inline bool operator>=(const __half &lh, const __half &rh) { return __hge(lh, rh); }
__device__ inline bool operator<=(const __half &lh, const __half &rh) { return __hle(lh, rh); }

struct alignas(4) __half2 {
    __half x{};
    __half y{};

public:
    __half2() = default;
    __device__ __half2(const __half2 &&src) { *(reinterpret_cast<unsigned int *>(&(*this))) = std::move(*(reinterpret_cast<const unsigned int *>(&(src)))); }
    __device__ __half2 &operator=(const __half2 &&src) {
        *(reinterpret_cast<unsigned int *>(&(*this))) = std::move(*(reinterpret_cast<const unsigned int *>(&(src))));
        return *this;
    }
    __device__ constexpr __half2(const __half &a, const __half &b) : x(a), y(b) {}
    __device__ __half2(const __half2 &src) { *(reinterpret_cast<unsigned int *>(&(*this))) = *(reinterpret_cast<const unsigned int *>(&(src))); }
    __device__ __half2 &operator=(const __half2 &src) {
        *(reinterpret_cast<unsigned int *>(&(*this))) = *(reinterpret_cast<const unsigned int *>(&(src)));
        return *this;
    }
    __device__ __half2(const __half2_raw &h2r) { *(reinterpret_cast<unsigned int *>(&(*this))) = *(reinterpret_cast<const unsigned int *>(&(h2r))); }
    __device__ __half2 &operator=(const __half2_raw &h2r) {
        *(reinterpret_cast<unsigned int *>(&(*this))) = *(reinterpret_cast<const unsigned int *>(&(h2r)));
        return *this;
    }
    __device__ operator __half2_raw() const {
        __half2_raw ret;
        ret.x = 0U;
        ret.y = 0U;
        *(reinterpret_cast<unsigned int *>(&(ret))) = *(reinterpret_cast<const unsigned int *>(&(*this)));
        return ret;
    }
};
__device__ inline __half2 operator+(const __half2 &lh, const __half2 &rh) { return __hadd2(lh, rh); }
__device__ inline __half2 operator-(const __half2 &lh, const __half2 &rh) { return __hsub2(lh, rh); }
__device__ inline __half2 operator*(const __half2 &lh, const __half2 &rh) { return __hmul2(lh, rh); }
__device__ inline __half2 operator/(const __half2 &lh, const __half2 &rh) { return __h2div(lh, rh); }
__device__ inline __half2 &operator+=(__half2 &lh, const __half2 &rh) {
    lh = __hadd2(lh, rh);
    return lh;
}
__device__ inline __half2 &operator-=(__half2 &lh, const __half2 &rh) {
    lh = __hsub2(lh, rh);
    return lh;
}
__device__ inline __half2 &operator*=(__half2 &lh, const __half2 &rh) {
    lh = __hmul2(lh, rh);
    return lh;
}
__device__ inline __half2 &operator/=(__half2 &lh, const __half2 &rh) {
    lh = __h2div(lh, rh);
    return lh;
}
__device__ inline __half2 &operator++(__half2 &h) {
    __half2_raw one;
    one.x = 0x3C00U;
    one.y = 0x3C00U;
    h = __hadd2(h, one);
    return h;
}
__device__ inline __half2 &operator--(__half2 &h) {
    __half2_raw one;
    one.x = 0x3C00U;
    one.y = 0x3C00U;
    h = __hsub2(h, one);
    return h;
}
__device__ inline __half2 operator++(__half2 &h, const int ignored) {
    static_cast<void>(ignored);
    const __half2 ret = h;
    __half2_raw one;
    one.x = 0x3C00U;
    one.y = 0x3C00U;
    h = __hadd2(h, one);
    return ret;
}
__device__ inline __half2 operator--(__half2 &h, const int ignored) {
    static_cast<void>(ignored);
    const __half2 ret = h;
    __half2_raw one;
    one.x = 0x3C00U;
    one.y = 0x3C00U;
    h = __hsub2(h, one);
    return ret;
}
__device__ inline __half2 operator+(const __half2 &h) { return h; }
__device__ inline __half2 operator-(const __half2 &h) { return __hneg2(h); }
__device__ inline bool operator==(const __half2 &lh, const __half2 &rh) { return __hbeq2(lh, rh); }
__device__ inline bool operator!=(const __half2 &lh, const __half2 &rh) { return __hbneu2(lh, rh); }
__device__ inline bool operator>(const __half2 &lh, const __half2 &rh) { return __hbgt2(lh, rh); }
__device__ inline bool operator<(const __half2 &lh, const __half2 &rh) { return __hblt2(lh, rh); }
__device__ inline bool operator>=(const __half2 &lh, const __half2 &rh) { return __hbge2(lh, rh); }
__device__ inline bool operator<=(const __half2 &lh, const __half2 &rh) { return __hble2(lh, rh); }

__device__ inline __half __double2half(const double a) {
    {
        __half val;
        asm("{  cvt.rn.f16.f64 %0, %1;}\n"
            : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
            : "d"(a));
        return val;
    }
}
__device__ inline __half __float2half(const float a) {
    __half val;
    { asm("{  cvt.rn.f16.f32 %0, %1;}\n"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
          : "f"(a)); }
    return val;
}
__device__ inline __half __float2half_rn(const float a) {
    __half val;
    { asm("{  cvt.rn.f16.f32 %0, %1;}\n"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
          : "f"(a)); }
    return val;
}
__device__ inline __half __float2half_rz(const float a) {
    __half val;
    { asm("{  cvt.rz.f16.f32 %0, %1;}\n"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
          : "f"(a)); }
    return val;
}
__device__ inline __half __float2half_rd(const float a) {
    __half val;
    { asm("{  cvt.rm.f16.f32 %0, %1;}\n"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
          : "f"(a)); }
    return val;
}
__device__ inline __half __float2half_ru(const float a) {
    __half val;
    { asm("{  cvt.rp.f16.f32 %0, %1;}\n"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
          : "f"(a)); }
    return val;
}
__device__ inline __half2 __float2half2_rn(const float a) {
    __half2 val;
    { asm("{.reg .f16 low;\n"
          "  cvt.rn.f16.f32 low, %1;\n"
          "  mov.b32 %0, {low,low};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "f"(a)); }
    return val;
}
__device__ inline __half2 __internal_device_float2_to_half2_rn(const float a, const float b) {
    __half2 val;
    { asm("{.reg .f16 low,high;\n"
          "  cvt.rn.f16.f32 low, %1;\n"
          "  cvt.rn.f16.f32 high, %2;\n"
          "  mov.b32 %0, {low,high};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "f"(a), "f"(b)); }
    return val;
}
__device__ inline __half2 __floats2half2_rn(const float a, const float b) {
    __half2 val;
    { val = __internal_device_float2_to_half2_rn(a, b); }
    return val;
}
__device__ inline float __half2float(const __half a) {
    float val;
    { asm("{  cvt.f32.f16 %0, %1;}\n"
          : "=f"(val)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(a))))); }
    return val;
}
__device__ inline float __low2float(const __half2 a) {
    float val;
    { asm("{.reg .f16 low,high;\n"
          "  mov.b32 {low,high},%1;\n"
          "  cvt.f32.f16 %0, low;}\n"
          : "=f"(val)
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return val;
}
__device__ inline float __high2float(const __half2 a) {
    float val;
    { asm("{.reg .f16 low,high;\n"
          "  mov.b32 {low,high},%1;\n"
          "  cvt.f32.f16 %0, high;}\n"
          : "=f"(val)
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return val;
}
__device__ inline signed char __half2char_rz(const __half h) {
    signed char i;
    {
        const float f = __half2float(h);
        const signed char max_val = (signed char)0x7fU;
        const signed char min_val = (signed char)0x80U;
        const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
        if (bits > (unsigned short)0xF800U) {
            i = 0;
        } else if (f > static_cast<float>(max_val)) {
            i = max_val;
        } else if (f < static_cast<float>(min_val)) {
            i = min_val;
        } else {
            i = static_cast<signed char>(f);
        }
    }
    return i;
}
__device__ inline unsigned char __half2uchar_rz(const __half h) {
    unsigned char i;
    {
        const float f = __half2float(h);
        const unsigned char max_val = 0xffU;
        const unsigned char min_val = 0U;
        const unsigned short bits = static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
        if (bits > (unsigned short)0xF800U) {
            i = 0U;
        } else if (f > static_cast<float>(max_val)) {
            i = max_val;
        } else if (f < static_cast<float>(min_val)) {
            i = min_val;
        } else {
            i = static_cast<unsigned char>(f);
        }
    }
    return i;
}
__device__ inline short int __half2short_rz(const __half h) {
    short int i;
    { asm("cvt.rzi.s16.f16 %0, %1;"
          : "=h"(i)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(h))))); }
    return i;
}
__device__ inline unsigned short int __half2ushort_rz(const __half h) {
    unsigned short int i;
    { asm("cvt.rzi.u16.f16 %0, %1;"
          : "=h"(i)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(h))))); }
    return i;
}
__device__ inline int __half2int_rz(const __half h) {
    int i;
    { asm("cvt.rzi.s32.f16 %0, %1;"
          : "=r"(i)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(h))))); }
    return i;
}
__device__ inline unsigned int __half2uint_rz(const __half h) {
    unsigned int i;
    { asm("cvt.rzi.u32.f16 %0, %1;"
          : "=r"(i)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(h))))); }
    return i;
}
__device__ inline long long int __half2ll_rz(const __half h) {
    long long int i;
    { asm("cvt.rzi.s64.f16 %0, %1;"
          : "=l"(i)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(h))))); }
    return i;
}
__device__ inline unsigned long long int __half2ull_rz(const __half h) {
    unsigned long long int i;
    { asm("cvt.rzi.u64.f16 %0, %1;"
          : "=l"(i)
          : "h"(*(reinterpret_cast<const unsigned short *>(&(h))))); }
    return i;
}
__device__ inline __half2 make_half2(const __half x, const __half y) {
    __half2 t;
    t.x = x;
    t.y = y;
    return t;
}
__device__ inline __half2 __float22half2_rn(const float2 a) {
    const __half2 val = __floats2half2_rn(a.x, a.y);
    return val;
}
__device__ inline float2 __half22float2(const __half2 a) {
    float hi_float;
    float lo_float;
    {
        asm("{.reg .f16 low,high;\n"
            "  mov.b32 {low,high},%1;\n"
            "  cvt.f32.f16 %0, low;}\n"
            : "=f"(lo_float)
            : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))));
        asm("{.reg .f16 low,high;\n"
            "  mov.b32 {low,high},%1;\n"
            "  cvt.f32.f16 %0, high;}\n"
            : "=f"(hi_float)
            : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))));
    }
    return make_float2(lo_float, hi_float);
}
__device__ inline int __half2int_rn(const __half h) {
    int i;
    asm("cvt.rni.s32.f16 %0, %1;"
        : "=r"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline int __half2int_rd(const __half h) {
    int i;
    asm("cvt.rmi.s32.f16 %0, %1;"
        : "=r"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline int __half2int_ru(const __half h) {
    int i;
    asm("cvt.rpi.s32.f16 %0, %1;"
        : "=r"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline __half __int2half_rn(const int i) {
    __half h;
    { asm("cvt.rn.f16.s32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline __half __int2half_rz(const int i) {
    __half h;
    { asm("cvt.rz.f16.s32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline __half __int2half_rd(const int i) {
    __half h;
    { asm("cvt.rm.f16.s32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline __half __int2half_ru(const int i) {
    __half h;
    { asm("cvt.rp.f16.s32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline short int __half2short_rn(const __half h) {
    short int i;
    asm("cvt.rni.s16.f16 %0, %1;"
        : "=h"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline short int __half2short_rd(const __half h) {
    short int i;
    asm("cvt.rmi.s16.f16 %0, %1;"
        : "=h"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline short int __half2short_ru(const __half h) {
    short int i;
    asm("cvt.rpi.s16.f16 %0, %1;"
        : "=h"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline __half __short2half_rn(const short int i) {
    __half h;
    { asm("cvt.rn.f16.s16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline __half __short2half_rz(const short int i) {
    __half h;
    { asm("cvt.rz.f16.s16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline __half __short2half_rd(const short int i) {
    __half h;
    { asm("cvt.rm.f16.s16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline __half __short2half_ru(const short int i) {
    __half h;
    { asm("cvt.rp.f16.s16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline unsigned int __half2uint_rn(const __half h) {
    unsigned int i;
    asm("cvt.rni.u32.f16 %0, %1;"
        : "=r"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline unsigned int __half2uint_rd(const __half h) {
    unsigned int i;
    asm("cvt.rmi.u32.f16 %0, %1;"
        : "=r"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline unsigned int __half2uint_ru(const __half h) {
    unsigned int i;
    asm("cvt.rpi.u32.f16 %0, %1;"
        : "=r"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline __half __uint2half_rn(const unsigned int i) {
    __half h;
    { asm("cvt.rn.f16.u32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline __half __uint2half_rz(const unsigned int i) {
    __half h;
    { asm("cvt.rz.f16.u32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline __half __uint2half_rd(const unsigned int i) {
    __half h;
    { asm("cvt.rm.f16.u32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline __half __uint2half_ru(const unsigned int i) {
    __half h;
    { asm("cvt.rp.f16.u32 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "r"(i)); }
    return h;
}
__device__ inline unsigned short int __half2ushort_rn(const __half h) {
    unsigned short int i;
    asm("cvt.rni.u16.f16 %0, %1;"
        : "=h"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline unsigned short int __half2ushort_rd(const __half h) {
    unsigned short int i;
    asm("cvt.rmi.u16.f16 %0, %1;"
        : "=h"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline unsigned short int __half2ushort_ru(const __half h) {
    unsigned short int i;
    asm("cvt.rpi.u16.f16 %0, %1;"
        : "=h"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline __half __ushort2half_rn(const unsigned short int i) {
    __half h;
    { asm("cvt.rn.f16.u16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline __half __ushort2half_rz(const unsigned short int i) {
    __half h;
    { asm("cvt.rz.f16.u16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline __half __ushort2half_rd(const unsigned short int i) {
    __half h;
    { asm("cvt.rm.f16.u16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline __half __ushort2half_ru(const unsigned short int i) {
    __half h;
    { asm("cvt.rp.f16.u16 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "h"(i)); }
    return h;
}
__device__ inline unsigned long long int __half2ull_rn(const __half h) {
    unsigned long long int i;
    asm("cvt.rni.u64.f16 %0, %1;"
        : "=l"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline unsigned long long int __half2ull_rd(const __half h) {
    unsigned long long int i;
    asm("cvt.rmi.u64.f16 %0, %1;"
        : "=l"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline unsigned long long int __half2ull_ru(const __half h) {
    unsigned long long int i;
    asm("cvt.rpi.u64.f16 %0, %1;"
        : "=l"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline __half __ull2half_rn(const unsigned long long int i) {
    __half h;
    { asm("cvt.rn.f16.u64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half __ull2half_rz(const unsigned long long int i) {
    __half h;
    { asm("cvt.rz.f16.u64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half __ull2half_rd(const unsigned long long int i) {
    __half h;
    { asm("cvt.rm.f16.u64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half __ull2half_ru(const unsigned long long int i) {
    __half h;
    { asm("cvt.rp.f16.u64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline long long int __half2ll_rn(const __half h) {
    long long int i;
    asm("cvt.rni.s64.f16 %0, %1;"
        : "=l"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline long long int __half2ll_rd(const __half h) {
    long long int i;
    asm("cvt.rmi.s64.f16 %0, %1;"
        : "=l"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline long long int __half2ll_ru(const __half h) {
    long long int i;
    asm("cvt.rpi.s64.f16 %0, %1;"
        : "=l"(i)
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return i;
}
__device__ inline __half __ll2half_rn(const long long int i) {
    __half h;
    { asm("cvt.rn.f16.s64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half __ll2half_rz(const long long int i) {
    __half h;
    { asm("cvt.rz.f16.s64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half __ll2half_rd(const long long int i) {
    __half h;
    { asm("cvt.rm.f16.s64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half __ll2half_ru(const long long int i) {
    __half h;
    { asm("cvt.rp.f16.s64 %0, %1;"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(h))))
          : "l"(i)); }
    return h;
}
__device__ inline __half htrunc(const __half h) {
    __half r;
    asm("cvt.rzi.f16.f16 %0, %1;"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(r))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return r;
}
__device__ inline __half hceil(const __half h) {
    __half r;
    asm("cvt.rpi.f16.f16 %0, %1;"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(r))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return r;
}
__device__ inline __half hfloor(const __half h) {
    __half r;
    asm("cvt.rmi.f16.f16 %0, %1;"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(r))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return r;
}
__device__ inline __half hrint(const __half h) {
    __half r;
    asm("cvt.rni.f16.f16 %0, %1;"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(r))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(h)))));
    return r;
}
__device__ inline __half2 h2trunc(const __half2 h) {
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rzi.f16.f16 low, low;\n"
        "  cvt.rzi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
        : "r"(*(reinterpret_cast<const unsigned int *>(&(h)))));
    return val;
}
__device__ inline __half2 h2ceil(const __half2 h) {
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rpi.f16.f16 low, low;\n"
        "  cvt.rpi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
        : "r"(*(reinterpret_cast<const unsigned int *>(&(h)))));
    return val;
}
__device__ inline __half2 h2floor(const __half2 h) {
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rmi.f16.f16 low, low;\n"
        "  cvt.rmi.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
        : "r"(*(reinterpret_cast<const unsigned int *>(&(h)))));
    return val;
}
__device__ inline __half2 h2rint(const __half2 h) {
    __half2 val;
    asm("{.reg .f16 low,high;\n"
        "  mov.b32 {low,high}, %1;\n"
        "  cvt.rni.f16.f16 low, low;\n"
        "  cvt.rni.f16.f16 high, high;\n"
        "  mov.b32 %0, {low,high};}\n"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
        : "r"(*(reinterpret_cast<const unsigned int *>(&(h)))));
    return val;
}
__device__ inline __half2 __lows2half2(const __half2 a, const __half2 b) {
    __half2 val;
    { asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
          "  mov.b32 {alow,ahigh}, %1;\n"
          "  mov.b32 {blow,bhigh}, %2;\n"
          "  mov.b32 %0, {alow,blow};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))), "r"(*(reinterpret_cast<const unsigned int *>(&(b))))); }
    return val;
}
__device__ inline __half2 __highs2half2(const __half2 a, const __half2 b) {
    __half2 val;
    { asm("{.reg .f16 alow,ahigh,blow,bhigh;\n"
          "  mov.b32 {alow,ahigh}, %1;\n"
          "  mov.b32 {blow,bhigh}, %2;\n"
          "  mov.b32 %0, {ahigh,bhigh};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))), "r"(*(reinterpret_cast<const unsigned int *>(&(b))))); }
    return val;
}
__device__ inline __half __low2half(const __half2 a) {
    __half ret;
    { asm("{.reg .f16 low,high;\n"
          " mov.b32 {low,high}, %1;\n"
          " mov.b16 %0, low;}"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return ret;
}
__device__ inline int __hisinf(const __half a) {
    int retval;
    const __half_raw araw = __half_raw(a);
    if (araw.x == 0xFC00U) {
        retval = -1;
    } else if (araw.x == 0x7C00U) {
        retval = 1;
    } else {
        retval = 0;
    }
    return retval;
}
__device__ inline __half2 __low2half2(const __half2 a) {
    __half2 val;
    { asm("{.reg .f16 low,high;\n"
          "  mov.b32 {low,high}, %1;\n"
          "  mov.b32 %0, {low,low};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return val;
}
__device__ inline __half2 __high2half2(const __half2 a) {
    __half2 val;
    { asm("{.reg .f16 low,high;\n"
          "  mov.b32 {low,high}, %1;\n"
          "  mov.b32 %0, {high,high};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return val;
}
__device__ inline __half __high2half(const __half2 a) {
    __half ret;
    { asm("{.reg .f16 low,high;\n"
          " mov.b32 {low,high}, %1;\n"
          " mov.b16 %0, high;}"
          : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return ret;
}
__device__ inline __half2 __halves2half2(const __half a, const __half b) {
    __half2 val;
    { asm("{  mov.b32 %0, {%1,%2};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))), "h"(*(reinterpret_cast<const unsigned short *>(&(b))))); }
    return val;
}
__device__ inline __half2 __half2half2(const __half a) {
    __half2 val;
    { asm("{  mov.b32 %0, {%1,%1};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "h"(*(reinterpret_cast<const unsigned short *>(&(a))))); }
    return val;
}
__device__ inline __half2 __lowhigh2highlow(const __half2 a) {
    __half2 val;
    { asm("{.reg .f16 low,high;\n"
          "  mov.b32 {low,high}, %1;\n"
          "  mov.b32 %0, {high,low};}\n"
          : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
          : "r"(*(reinterpret_cast<const unsigned int *>(&(a))))); }
    return val;
}
__device__ inline short int __half_as_short(const __half h) {
    { return static_cast<short int>(*(reinterpret_cast<const unsigned short *>(&(h)))); }
}
__device__ inline unsigned short int __half_as_ushort(const __half h) {
    { return *(reinterpret_cast<const unsigned short *>(&(h))); }
}
__device__ inline __half __short_as_half(const short int i) {
    {
        __half h;
        *(reinterpret_cast<unsigned short *>(&(h))) = static_cast<unsigned short int>(i);
        return h;
    }
}
__device__ inline __half __ushort_as_half(const unsigned short int i) {
    {
        __half h;
        *(reinterpret_cast<unsigned short *>(&(h))) = i;
        return h;
    }
}
__device__ inline __half __internal_device_hmax(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        float fr;
        asm("{max.f32 %0,%1,%2;\n}"
            : "=f"(fr)
            : "f"(fa), "f"(fb));
        const __half hr = __float2half(fr);
        return hr;
    }
}
__device__ inline __half __internal_device_hmin(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        float fr;
        asm("{min.f32 %0,%1,%2;\n}"
            : "=f"(fr)
            : "f"(fa), "f"(fb));
        const __half hr = __float2half(fr);
        return hr;
    }
}
__device__ inline __half __hmax(const __half a, const __half b) {
    { return __internal_device_hmax(a, b); }
}
__device__ inline __half __hmin(const __half a, const __half b) {
    { return __internal_device_hmin(a, b); }
}
__device__ inline __half2 __hmax2(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hmax(a.x, b.x);
        val.y = __hmax(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hmin2(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hmin(a.x, b.x);
        val.y = __hmin(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __shfl(const __half2 var, const int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.idx.b32"
                     " %0,%1,%2,%3;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c));
        return r;
    }
}
__device__ inline __half2 __shfl_up(const __half2 var, const unsigned int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = (warp_size - static_cast<unsigned>(width)) << 8U;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.up.b32"
                     " %0,%1,%2,%3;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c));
        return r;
    }
}
__device__ inline __half2 __shfl_down(const __half2 var, const unsigned int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.down.b32"
                     " %0,%1,%2,%3;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c));
        return r;
    }
}
__device__ inline __half2 __shfl_xor(const __half2 var, const int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.bfly.b32"
                     " %0,%1,%2,%3;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c));
        return r;
    }
}
__device__ inline __half2 __shfl_sync(const unsigned mask, const __half2 var, const int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.sync.idx.b32"
                     " %0,%1,%2,%3,%4;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c), "r"(mask));
        return r;
    }
}
__device__ inline __half2 __shfl_up_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = (warp_size - static_cast<unsigned>(width)) << 8U;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.sync.up.b32"
                     " %0,%1,%2,%3,%4;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c), "r"(mask));
        return r;
    }
}
__device__ inline __half2 __shfl_down_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.sync.down.b32"
                     " %0,%1,%2,%3,%4;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c), "r"(mask));
        return r;
    }
}
__device__ inline __half2 __shfl_xor_sync(const unsigned mask, const __half2 var, const int delta, const int width) {
    unsigned int warp_size;
    asm("{mov.u32 %0, WARP_SZ;\n}"
        : "=r"(warp_size));
    const unsigned int c = ((warp_size - static_cast<unsigned>(width)) << 8U) | 0x1fU;
    {
        __half2 r;
        asm volatile("{"
                     "shfl.sync.bfly.b32"
                     " %0,%1,%2,%3,%4;\n}"
                     : "=r"(*(reinterpret_cast<unsigned int *>(&(r))))
                     : "r"(*(reinterpret_cast<const unsigned int *>(&(var)))), "r"(delta), "r"(c), "r"(mask));
        return r;
    }
}
__device__ inline __half __shfl(const __half var, const int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl(temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_up(const __half var, const unsigned int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_up(temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_down(const __half var, const unsigned int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_down(temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_xor(const __half var, const int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_xor(temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_sync(const unsigned mask, const __half var, const int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_up_sync(const unsigned mask, const __half var, const unsigned int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_up_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_down_sync(const unsigned mask, const __half var, const unsigned int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_down_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half __shfl_xor_sync(const unsigned mask, const __half var, const int delta, const int width) {
    const __half2 temp1 = __halves2half2(var, var);
    const __half2 temp2 = __shfl_xor_sync(mask, temp1, delta, width);
    return __low2half(temp2);
}
__device__ inline __half2 __ldg(const __half2 *const ptr) {
    __half2 ret;
    asm("ld.global.nc.b32 %0, [%1];"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half __ldg(const __half *const ptr) {
    __half ret;
    asm("ld.global.nc.b16 %0, [%1];"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half2 __ldcg(const __half2 *const ptr) {
    __half2 ret;
    asm("ld.global.cg.b32 %0, [%1];"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half __ldcg(const __half *const ptr) {
    __half ret;
    asm("ld.global.cg.b16 %0, [%1];"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half2 __ldca(const __half2 *const ptr) {
    __half2 ret;
    asm("ld.global.ca.b32 %0, [%1];"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half __ldca(const __half *const ptr) {
    __half ret;
    asm("ld.global.ca.b16 %0, [%1];"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half2 __ldcs(const __half2 *const ptr) {
    __half2 ret;
    asm("ld.global.cs.b32 %0, [%1];"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half __ldcs(const __half *const ptr) {
    __half ret;
    asm("ld.global.cs.b16 %0, [%1];"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
        : "l"(ptr));
    return ret;
}
__device__ inline __half2 __ldlu(const __half2 *const ptr) {
    __half2 ret;
    asm("ld.global.lu.b32 %0, [%1];"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(ret))))
        : "l"(ptr)
        : "memory");
    return ret;
}
__device__ inline __half __ldlu(const __half *const ptr) {
    __half ret;
    asm("ld.global.lu.b16 %0, [%1];"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
        : "l"(ptr)
        : "memory");
    return ret;
}
__device__ inline __half2 __ldcv(const __half2 *const ptr) {
    __half2 ret;
    asm("ld.global.cv.b32 %0, [%1];"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(ret))))
        : "l"(ptr)
        : "memory");
    return ret;
}
__device__ inline __half __ldcv(const __half *const ptr) {
    __half ret;
    asm("ld.global.cv.b16 %0, [%1];"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(ret))))
        : "l"(ptr)
        : "memory");
    return ret;
}
__device__ inline void __stwb(__half2 *const ptr, const __half2 value) {
    asm("st.global.wb.b32 [%0], %1;" ::"l"(ptr), "r"(*(reinterpret_cast<const unsigned int *>(&(value))))
        : "memory");
}
__device__ inline void __stwb(__half *const ptr, const __half value) {
    asm("st.global.wb.b16 [%0], %1;" ::"l"(ptr), "h"(*(reinterpret_cast<const unsigned short *>(&(value))))
        : "memory");
}
__device__ inline void __stcg(__half2 *const ptr, const __half2 value) {
    asm("st.global.cg.b32 [%0], %1;" ::"l"(ptr), "r"(*(reinterpret_cast<const unsigned int *>(&(value))))
        : "memory");
}
__device__ inline void __stcg(__half *const ptr, const __half value) {
    asm("st.global.cg.b16 [%0], %1;" ::"l"(ptr), "h"(*(reinterpret_cast<const unsigned short *>(&(value))))
        : "memory");
}
__device__ inline void __stcs(__half2 *const ptr, const __half2 value) {
    asm("st.global.cs.b32 [%0], %1;" ::"l"(ptr), "r"(*(reinterpret_cast<const unsigned int *>(&(value))))
        : "memory");
}
__device__ inline void __stcs(__half *const ptr, const __half value) {
    asm("st.global.cs.b16 [%0], %1;" ::"l"(ptr), "h"(*(reinterpret_cast<const unsigned short *>(&(value))))
        : "memory");
}
__device__ inline void __stwt(__half2 *const ptr, const __half2 value) {
    asm("st.global.wt.b32 [%0], %1;" ::"l"(ptr), "r"(*(reinterpret_cast<const unsigned int *>(&(value))))
        : "memory");
}
__device__ inline void __stwt(__half *const ptr, const __half value) {
    asm("st.global.wt.b16 [%0], %1;" ::"l"(ptr), "h"(*(reinterpret_cast<const unsigned short *>(&(value))))
        : "memory");
}
__device__ inline __half2 __heq2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __heq(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __heq(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hne2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hne(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hne(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hle2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hle(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hle(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hge2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hge(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hge(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hlt2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hlt(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hlt(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hgt2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hgt(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hgt(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hequ2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hequ(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hequ(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hneu2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hneu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hneu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hleu2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hleu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hleu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hgeu2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hgeu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hgeu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hltu2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hltu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hltu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline __half2 __hgtu2(const __half2 a, const __half2 b) {
    {
        __half2_raw val;
        val.x = __hgtu(a.x, b.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hgtu(a.y, b.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        return __half2(val);
    }
}
__device__ inline unsigned __heq2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __heq(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __heq(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hne2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hne(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hne(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hle2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hle(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hle(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hge2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hge(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hge(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hlt2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hlt(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hlt(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hgt2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hgt(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hgt(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hequ2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hequ(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hequ(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hneu2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hneu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hneu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hleu2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hleu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hleu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hgeu2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hgeu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hgeu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hltu2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hltu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hltu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline unsigned __hgtu2_mask(const __half2 a, const __half2 b) {
    {
        const unsigned short px = __hgtu(a.x, b.x) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        const unsigned short py = __hgtu(a.y, b.y) ? (unsigned short)0xFFFFU : (unsigned short)0U;
        unsigned ur = (unsigned)py;
        ur <<= (unsigned)16U;
        ur |= (unsigned)px;
        return ur;
    }
}
__device__ inline bool __hbeq2(const __half2 a, const __half2 b) {
    const unsigned mask = __heq2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbne2(const __half2 a, const __half2 b) {
    const unsigned mask = __hne2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hble2(const __half2 a, const __half2 b) {
    const unsigned mask = __hle2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbge2(const __half2 a, const __half2 b) {
    const unsigned mask = __hge2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hblt2(const __half2 a, const __half2 b) {
    const unsigned mask = __hlt2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbgt2(const __half2 a, const __half2 b) {
    const unsigned mask = __hgt2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbequ2(const __half2 a, const __half2 b) {
    const unsigned mask = __hequ2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbneu2(const __half2 a, const __half2 b) {
    const unsigned mask = __hneu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbleu2(const __half2 a, const __half2 b) {
    const unsigned mask = __hleu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbgeu2(const __half2 a, const __half2 b) {
    const unsigned mask = __hgeu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbltu2(const __half2 a, const __half2 b) {
    const unsigned mask = __hltu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __hbgtu2(const __half2 a, const __half2 b) {
    const unsigned mask = __hgtu2_mask(a, b);
    return (mask == 0xFFFFFFFFU);
}
__device__ inline bool __heq(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa == fb);
    }
}
__device__ inline bool __hne(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa != fb) && (!__hisnan(a)) && (!__hisnan(b));
    }
}
__device__ inline bool __hle(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa <= fb);
    }
}
__device__ inline bool __hge(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa >= fb);
    }
}
__device__ inline bool __hlt(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa < fb);
    }
}
__device__ inline bool __hgt(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa > fb);
    }
}
__device__ inline bool __hequ(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa == fb) || (__hisnan(a)) || (__hisnan(b));
    }
}
__device__ inline bool __hneu(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa != fb);
    }
}
__device__ inline bool __hleu(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa <= fb) || (__hisnan(a)) || (__hisnan(b));
    }
}
__device__ inline bool __hgeu(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa >= fb) || (__hisnan(a)) || (__hisnan(b));
    }
}
__device__ inline bool __hltu(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa < fb) || (__hisnan(a)) || (__hisnan(b));
    }
}
__device__ inline bool __hgtu(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return (fa > fb) || (__hisnan(a)) || (__hisnan(b));
    }
}
__device__ inline __half2 __hadd2(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hadd(a.x, b.x);
        val.y = __hadd(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hsub2(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hsub(a.x, b.x);
        val.y = __hsub(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hmul2(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hmul(a.x, b.x);
        val.y = __hmul(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hadd2_sat(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hadd_sat(a.x, b.x);
        val.y = __hadd_sat(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hsub2_sat(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hsub_sat(a.x, b.x);
        val.y = __hsub_sat(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hmul2_sat(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hmul_sat(a.x, b.x);
        val.y = __hmul_sat(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hadd2_rn(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hadd_rn(a.x, b.x);
        val.y = __hadd_rn(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hsub2_rn(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hsub_rn(a.x, b.x);
        val.y = __hsub_rn(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __hmul2_rn(const __half2 a, const __half2 b) {
    {
        __half2 val;
        val.x = __hmul_rn(a.x, b.x);
        val.y = __hmul_rn(a.y, b.y);
        return val;
    }
}
__device__ inline __half2 __h2div(const __half2 a, const __half2 b) {
    __half ha = __low2half(a);
    __half hb = __low2half(b);
    const __half v1 = __hdiv(ha, hb);
    ha = __high2half(a);
    hb = __high2half(b);
    const __half v2 = __hdiv(ha, hb);
    return __halves2half2(v1, v2);
}
__device__ inline __half __hadd(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return __float2half(fa + fb);
    }
}
__device__ inline __half __hsub(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return __float2half(fa - fb);
    }
}
__device__ inline __half __hmul(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return __float2half(fa * fb);
    }
}
__device__ inline __half __hadd_sat(const __half a, const __half b) {
    { return __hmin(__hmax(__hadd(a, b), __ushort_as_half((unsigned short)0x0000U)), __ushort_as_half((unsigned short)0x3C00U)); }
}
__device__ inline __half __hsub_sat(const __half a, const __half b) {
    { return __hmin(__hmax(__hsub(a, b), __ushort_as_half((unsigned short)0x0000U)), __ushort_as_half((unsigned short)0x3C00U)); }
}
__device__ inline __half __hmul_sat(const __half a, const __half b) {
    { return __hmin(__hmax(__hmul(a, b), __ushort_as_half((unsigned short)0x0000U)), __ushort_as_half((unsigned short)0x3C00U)); }
}
__device__ inline __half __hadd_rn(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return __float2half(fa + fb);
    }
}
__device__ inline __half __hsub_rn(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return __float2half(fa - fb);
    }
}
__device__ inline __half __hmul_rn(const __half a, const __half b) {
    {
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        return __float2half(fa * fb);
    }
}
__device__ inline __half __hdiv(const __half a, const __half b) {
    {
        __half v;
        __half abs;
        __half den;
        *(reinterpret_cast<unsigned short *>(&(den))) = 0x008FU;
        float rcp;
        const float fa = __half2float(a);
        const float fb = __half2float(b);
        asm("{rcp.approx.ftz.f32 %0, %1;\n}"
            : "=f"(rcp)
            : "f"(fb));
        float fv = rcp * fa;
        v = __float2half(fv);
        abs = __habs(v);
        if (__hlt(abs, den) && __hlt(__float2half(0.0f), abs)) {
            const float err = __fmaf_rn(-fb, fv, fa);
            fv = __fmaf_rn(rcp, err, fv);
            v = __float2half(fv);
        }
        return v;
    }
}
__device__ inline __half hexp2(const __half a) {
    __half val;
    asm("{.reg.b32         f, ULP;         \n"
        " .reg.b16         r;              \n"
        "  mov.b16         r,%1;           \n"
        "  cvt.f32.f16     f,r;            \n"
        "  ex2.approx.ftz.f32      f,f;    \n"
        "  mov.b32         ULP, 0x33800000U;\n"
        "  fma.rn.f32      f,f,ULP,f;      \n"
        "  cvt.rn.f16.f32      r,f;        \n"
        "  mov.b16         %0,r;           \n"
        "}"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))));
    return val;
}
__device__ inline __half2 h2exp2(const __half2 a) {
    __half2 val;
    asm("{.reg.b16         hl, hu;         \n"
        " .reg.b32         fl, fu, ULP;    \n"
        "  mov.b32         {hl, hu}, %1;   \n"
        "  cvt.f32.f16     fl, hl;         \n"
        "  cvt.f32.f16     fu, hu;         \n"
        "  ex2.approx.ftz.f32  fl, fl;     \n"
        "  ex2.approx.ftz.f32  fu, fu;     \n"
        "  mov.b32         ULP, 0x33800000U;\n"
        "  fma.rn.f32      fl,fl,ULP,fl;   \n"
        "  fma.rn.f32      fu,fu,ULP,fu;   \n"
        "  cvt.rn.f16.f32      hl, fl;     \n"
        "  cvt.rn.f16.f32      hu, fu;     \n"
        "  mov.b32         %0, {hl, hu};   \n"
        "}"
        : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
        : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))));
    return val;
}
__device__ inline __half2 h2rcp(const __half2 a) {
    {
        __half2 val;
        asm("{.reg.b16         hl, hu;         \n"
            " .reg.b32         fl, fu;         \n"
            "  mov.b32         {hl, hu}, %1;   \n"
            "  cvt.f32.f16     fl, hl;         \n"
            "  cvt.f32.f16     fu, hu;         \n"
            "  "
            "rcp"
            ".approx.ftz.f32   fl, fl;     \n"
            "  "
            "rcp"
            ".approx.ftz.f32   fu, fu;     \n"
            "  cvt.rn.f16.f32      hl, fl;     \n"
            "  cvt.rn.f16.f32      hu, fu;     \n"
            "  mov.b32         %0, {hl, hu};   \n"
            "}"
            : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
            : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))));
        return val;
    }
}
__device__ inline __half hrcp(const __half a) {
    {
        __half val;
        asm("{.reg.b32         f;        \n"
            " .reg.b16         r;        \n"
            "  mov.b16         r,%1;     \n"
            "  cvt.f32.f16     f,r;      \n"
            "  "
            "rcp"
            ".approx.ftz.f32   f,f;  \n"
            "  cvt.rn.f16.f32      r,f;  \n"
            "  mov.b16         %0,r;     \n"
            "}"
            : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
            : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))));
        return val;
    }
}
__device__ inline __half2 h2rsqrt(const __half2 a) {
    {
        __half2 val;
        asm("{.reg.b16         hl, hu;         \n"
            " .reg.b32         fl, fu;         \n"
            "  mov.b32         {hl, hu}, %1;   \n"
            "  cvt.f32.f16     fl, hl;         \n"
            "  cvt.f32.f16     fu, hu;         \n"
            "  "
            "rsqrt"
            ".approx.ftz.f32   fl, fl;     \n"
            "  "
            "rsqrt"
            ".approx.ftz.f32   fu, fu;     \n"
            "  cvt.rn.f16.f32      hl, fl;     \n"
            "  cvt.rn.f16.f32      hu, fu;     \n"
            "  mov.b32         %0, {hl, hu};   \n"
            "}"
            : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
            : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))));
        return val;
    }
}
__device__ inline __half hrsqrt(const __half a) {
    {
        __half val;
        asm("{.reg.b32         f;        \n"
            " .reg.b16         r;        \n"
            "  mov.b16         r,%1;     \n"
            "  cvt.f32.f16     f,r;      \n"
            "  "
            "rsqrt"
            ".approx.ftz.f32   f,f;  \n"
            "  cvt.rn.f16.f32      r,f;  \n"
            "  mov.b16         %0,r;     \n"
            "}"
            : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
            : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))));
        return val;
    }
}
__device__ inline __half2 h2sqrt(const __half2 a) {
    {
        __half2 val;
        asm("{.reg.b16         hl, hu;         \n"
            " .reg.b32         fl, fu;         \n"
            "  mov.b32         {hl, hu}, %1;   \n"
            "  cvt.f32.f16     fl, hl;         \n"
            "  cvt.f32.f16     fu, hu;         \n"
            "  "
            "sqrt"
            ".approx.ftz.f32   fl, fl;     \n"
            "  "
            "sqrt"
            ".approx.ftz.f32   fu, fu;     \n"
            "  cvt.rn.f16.f32      hl, fl;     \n"
            "  cvt.rn.f16.f32      hu, fu;     \n"
            "  mov.b32         %0, {hl, hu};   \n"
            "}"
            : "=r"(*(reinterpret_cast<unsigned int *>(&(val))))
            : "r"(*(reinterpret_cast<const unsigned int *>(&(a)))));
        return val;
    }
}
__device__ inline __half hsqrt(const __half a) {
    {
        __half val;
        asm("{.reg.b32         f;        \n"
            " .reg.b16         r;        \n"
            "  mov.b16         r,%1;     \n"
            "  cvt.f32.f16     f,r;      \n"
            "  "
            "sqrt"
            ".approx.ftz.f32   f,f;  \n"
            "  cvt.rn.f16.f32      r,f;  \n"
            "  mov.b16         %0,r;     \n"
            "}"
            : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
            : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))));
        return val;
    }
}
__device__ inline __half2 __hisnan2(const __half2 a) {
    __half2 r;
    {
        __half2_raw val;
        val.x = __hisnan(a.x) ? (unsigned short)0x3C00U : (unsigned short)0U;
        val.y = __hisnan(a.y) ? (unsigned short)0x3C00U : (unsigned short)0U;
        r = __half2(val);
    }
    return r;
}
__device__ inline bool __hisnan(const __half a) {
    {
        const __half_raw hr = static_cast<__half_raw>(a);
        return ((hr.x & (unsigned short)0x7FFFU) > (unsigned short)0x7C00U);
    }
}
__device__ inline __half2 __hneg2(const __half2 a) {
    __half2 r;
    {
        r.x = __hneg(a.x);
        r.y = __hneg(a.y);
    }
    return r;
}
__device__ inline __half __hneg(const __half a) {
    {
        const float fa = __half2float(a);
        return __float2half(-fa);
    }
}
__device__ inline __half2 __habs2(const __half2 a) {
    __half2 r;
    {
        r.x = __habs(a.x);
        r.y = __habs(a.y);
    }
    return r;
}
__device__ inline __half __habs(const __half a) {
    {
        __half_raw abs_a_raw = static_cast<__half_raw>(a);
        abs_a_raw.x &= (unsigned short)0x7FFFU;
        if (abs_a_raw.x > (unsigned short)0x7C00U) { abs_a_raw.x = (unsigned short)0x7FFFU; }
        return static_cast<__half>(abs_a_raw);
    }
}
__device__ inline __half __hmax_nan(const __half a, const __half b) {
    {
        __half maxval;
        if (__hisnan(a) || __hisnan(b)) {
            maxval = __ushort_as_half((unsigned short)0x7FFFU);
        } else {
            maxval = __hmax(a, b);
        }
        return maxval;
    }
}
__device__ inline __half __hmin_nan(const __half a, const __half b) {
    {
        __half minval;
        if (__hisnan(a) || __hisnan(b)) {
            minval = __ushort_as_half((unsigned short)0x7FFFU);
        } else {
            minval = __hmin(a, b);
        }
        return minval;
    }
}
__device__ inline __half2 __hmax2_nan(const __half2 a, const __half2 b) {
    {
        __half2 result = __hmax2(a, b);
        if (__hisnan(a.x) || __hisnan(b.x)) { result.x = __ushort_as_half((unsigned short)0x7FFFU); }
        if (__hisnan(a.y) || __hisnan(b.y)) { result.y = __ushort_as_half((unsigned short)0x7FFFU); }
        return result;
    }
}
__device__ inline __half2 __hmin2_nan(const __half2 a, const __half2 b) {
    {
        __half2 result = __hmin2(a, b);
        if (__hisnan(a.x) || __hisnan(b.x)) { result.x = __ushort_as_half((unsigned short)0x7FFFU); }
        if (__hisnan(a.y) || __hisnan(b.y)) { result.y = __ushort_as_half((unsigned short)0x7FFFU); }
        return result;
    }
}
__device__ inline __half2 atomicAdd(__half2 *const address, const __half2 val) {
    {
        unsigned int *address_as_uint = (unsigned int *)address;
        unsigned int old = *address_as_uint;
        unsigned int assumed;
        do {
            assumed = old;
            __half2 new_val = __hadd2(val, *(__half2 *)&assumed);
            old = atomicCAS(address_as_uint, assumed, *(unsigned int *)&new_val);
        } while (assumed != old);
        return *(__half2 *)&old;
    }
}
__device__ inline __half __hfma(const __half a, const __half b, const __half c) {
    __half val;
    asm("{fma.rn.f16 %0,%1,%2,%3;\n}"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))),
          "h"(*(reinterpret_cast<const unsigned short *>(&(b)))),
          "h"(*(reinterpret_cast<const unsigned short *>(&(c)))));
    return val;
}
__device__ inline __half _hfma_sat(const __half a, const __half b, const __half c) {
    __half val;
    asm("{fma.rn.sat.f16 %0,%1,%2,%3;\n}"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(val))))
        : "h"(*(reinterpret_cast<const unsigned short *>(&(a)))),
          "h"(*(reinterpret_cast<const unsigned short *>(&(b)))),
          "h"(*(reinterpret_cast<const unsigned short *>(&(c)))));
    return val;
}

using half = __half;
using half2 = __half2;

#endif
