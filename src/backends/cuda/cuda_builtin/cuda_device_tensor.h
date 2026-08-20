//
// Device-side implementations of the runtime tensor `CallOp`s (plan.md §3.3).
//
// Every `lc_tensor_*` device function is *cooperative*: the enclosing kernel
// dispatches enough threads to cover the op's iteration domain and every
// thread executes the same call, partitioning the domain grid-stride (or
// one-warp-per-tile for the tensor-core GEMM). The descriptor arguments are
// the flat encoding produced by `include/luisa/dsl/tensor_ops.h`:
//
//   descriptor := (dtype:uint, rank:uint, extents:uint4, strides:uint4,
//                  offset:uint, addr:uint64)
//
// dtypes: 0 = f16 (half), 1 = f32 (float), 2 = i32 (int).
//
// The kernels mirror the target behavior of
// `src/tests/cuda/test_cuda_tensor_dispatch.cpp` (plan.md Appendix A):
//   - element-wise ops use a grid-stride loop over `count` with a contiguous
//     fast path and a generic strided path;
//   - the F32 GEMM is a flat one-thread-per-output kernel;
//   - the F16 GEMM uses the classic inline-PTX `mma.sync.aligned.m16n8k16`
//     tensor-core instruction (two MMAs per 16x16x16 WMMA op) with an FP32
//     accumulator, and falls back to a scalar F16 kernel for shapes that are
//     not multiples of 16 (the WMMA fragment loads would read out of bounds);
//   - reductions and scans are single-pass per-output-thread (correct v1;
//     block/warp-cooperative reductions are a later optimization milestone).
//

#pragma once

// ---------------------------------------------------------------------------
// Scalar type traits
// ---------------------------------------------------------------------------

template<int> struct LCTensorScalar;
template<> struct LCTensorScalar<0> { using type = half; };
template<> struct LCTensorScalar<1> { using type = float; };
template<> struct LCTensorScalar<2> { using type = int; };

__device__ __forceinline__ float lc_tensor_to_float(half x) noexcept { return __half2float(x); }
__device__ __forceinline__ float lc_tensor_to_float(float x) noexcept { return x; }
__device__ __forceinline__ float lc_tensor_to_float(int x) noexcept { return static_cast<float>(x); }

template<typename T>
__device__ __forceinline__ T lc_tensor_from_float(int, float x) noexcept;
template<>
__device__ __forceinline__ half lc_tensor_from_float<half>(int, float x) noexcept { return __float2half(x); }
template<>
__device__ __forceinline__ float lc_tensor_from_float<float>(int, float x) noexcept { return x; }
template<>
__device__ __forceinline__ int lc_tensor_from_float<int>(int, float x) noexcept { return static_cast<int>(x); }

template<typename T>
__device__ __forceinline__ T lc_tensor_abs_generic(T x) noexcept {
    return x < static_cast<T>(0) ? -x : x;
}
template<typename T>
__device__ __forceinline__ T lc_tensor_min_generic(T a, T b) noexcept {
    return b < a ? b : a;
}
template<typename T>
__device__ __forceinline__ T lc_tensor_max_generic(T a, T b) noexcept {
    return a < b ? b : a;
}

// Math helpers evaluated in float (well-defined for all three dtypes).
#define LC_TENSOR_MATH1(NAME, FN)                              \
    template<typename T>                                       \
    __device__ __forceinline__ T lc_tensor_##NAME(T x) noexcept { \
        return lc_tensor_from_float<T>(0, FN(lc_tensor_to_float(x))); \
    }
LC_TENSOR_MATH1(exp, expf)
LC_TENSOR_MATH1(log, logf)
LC_TENSOR_MATH1(sqrt, sqrtf)
LC_TENSOR_MATH1(rsqrt, rsqrtf)
LC_TENSOR_MATH1(sin, sinf)
LC_TENSOR_MATH1(cos, cosf)
LC_TENSOR_MATH1(tan, tanf)
LC_TENSOR_MATH1(tanh, tanhf)
LC_TENSOR_MATH1(erf, erff)
LC_TENSOR_MATH1(ceil, ceilf)
LC_TENSOR_MATH1(floor, floorf)
LC_TENSOR_MATH1(round, roundf)
#undef LC_TENSOR_MATH1

template<typename T>
__device__ __forceinline__ T lc_tensor_pow(T x, T y) noexcept {
    return lc_tensor_from_float<T>(0, powf(lc_tensor_to_float(x), lc_tensor_to_float(y)));
}
template<typename T>
__device__ __forceinline__ T lc_tensor_sigmoid(T x) noexcept {
    auto v = lc_tensor_to_float(x);
    return lc_tensor_from_float<T>(0, 1.0f / (1.0f + expf(-v)));
}
template<typename T>
__device__ __forceinline__ T lc_tensor_gelu(T x) noexcept {
    auto v = lc_tensor_to_float(x);
    return lc_tensor_from_float<T>(0, 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)));
}
template<typename T>
__device__ __forceinline__ T lc_tensor_relu(T x) noexcept {
    return lc_tensor_max_generic(x, static_cast<T>(0));
}
template<typename T>
__device__ __forceinline__ T lc_tensor_leaky_relu(T x) noexcept {
    return x < static_cast<T>(0) ? static_cast<T>(x * static_cast<T>(0.01f)) : x;
}

__device__ __forceinline__ bool lc_tensor_isnan_impl(half x) noexcept { return isnan(__half2float(x)); }
__device__ __forceinline__ bool lc_tensor_isnan_impl(float x) noexcept { return isnan(x); }
__device__ __forceinline__ bool lc_tensor_isnan_impl(int) noexcept { return false; }
__device__ __forceinline__ bool lc_tensor_isinf_impl(half x) noexcept { return isinf(__half2float(x)); }
__device__ __forceinline__ bool lc_tensor_isinf_impl(float x) noexcept { return isinf(x); }
__device__ __forceinline__ bool lc_tensor_isinf_impl(int) noexcept { return false; }

template<typename T>
__device__ __forceinline__ T lc_tensor_bits_to(lc_uint bits) noexcept;
template<>
__device__ __forceinline__ half lc_tensor_bits_to<half>(lc_uint bits) noexcept {
    return __ushort_as_half(static_cast<unsigned short>(bits));
}
template<>
__device__ __forceinline__ float lc_tensor_bits_to<float>(lc_uint bits) noexcept {
    return __int_as_float(static_cast<int>(bits));
}
template<>
__device__ __forceinline__ int lc_tensor_bits_to<int>(lc_uint bits) noexcept {
    return static_cast<int>(bits);
}

template<typename T>
__device__ __forceinline__ T lc_tensor_fma_impl(T a, T b, T c) noexcept {
    return static_cast<T>(a * b + c);
}
template<>
__device__ __forceinline__ half lc_tensor_fma_impl<half>(half a, half b, half c) noexcept {
    return __hfma(a, b, c);
}
template<>
__device__ __forceinline__ float lc_tensor_fma_impl<float>(float a, float b, float c) noexcept {
    return fmaf(a, b, c);
}

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------

struct LCTensorDesc {
    lc_uint dtype;
    lc_uint rank;
    lc_uint4 extents;
    lc_uint4 strides;
    lc_uint offset;
    lc_ulong addr;
};

__device__ __forceinline__ LCTensorDesc lc_tensor_desc(
    lc_uint dtype, lc_uint rank, lc_uint4 extents, lc_uint4 strides,
    lc_uint offset, lc_ulong addr) noexcept {
    return LCTensorDesc{dtype, rank, extents, strides, offset, addr};
}

__device__ __forceinline__ lc_uint lc_tensor_extent(const LCTensorDesc &d, int i) noexcept {
    return (&d.extents.x)[i];
}

__device__ __forceinline__ lc_uint lc_tensor_stride(const LCTensorDesc &d, int i) noexcept {
    return (&d.strides.x)[i];
}

__device__ __forceinline__ lc_uint lc_tensor_numel(const LCTensorDesc &d) noexcept {
    lc_uint n = 1u;
    for (int i = 0; i < static_cast<int>(d.rank); ++i) {
        n *= lc_tensor_extent(d, i);
    }
    return n;
}

__device__ __forceinline__ bool lc_tensor_contiguous(const LCTensorDesc &d) noexcept {
    lc_uint expected = 1u;
    for (int i = static_cast<int>(d.rank) - 1; i >= 0; --i) {
        if (lc_tensor_stride(d, i) != expected) { return false; }
        expected *= lc_tensor_extent(d, i);
    }
    return true;
}

// Row-major logical flat index -> storage offset (in elements), honoring the
// descriptor strides and storage offset. Only the first `rank` entries of
// `coords` are used.
__device__ __forceinline__ lc_uint lc_tensor_decompose_offset(
    const LCTensorDesc &d, lc_uint flat, lc_uint *coords) noexcept {
    lc_uint idx = flat;
    lc_uint off = d.offset;
    for (int i = static_cast<int>(d.rank) - 1; i >= 0; --i) {
        lc_uint e = lc_tensor_extent(d, i);
        lc_uint c = e > 0u ? idx % e : 0u;
        idx /= e;
        coords[i] = c;
        off += c * lc_tensor_stride(d, i);
    }
    return off;
}

__device__ __forceinline__ lc_uint lc_tensor_offset_from_coords(
    const LCTensorDesc &d, const lc_uint *coords) noexcept {
    lc_uint off = d.offset;
    for (int i = 0; i < static_cast<int>(d.rank); ++i) {
        off += coords[i] * lc_tensor_stride(d, i);
    }
    return off;
}

__device__ __forceinline__ void lc_tensor_grid_stride_begin(lc_uint &tid, lc_uint &total) noexcept {
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    total = blockDim.x * gridDim.x;
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ void lc_tensor_copy_typed(
    const LCTensorDesc &dst, const LCTensorDesc &src, lc_uint count) noexcept {
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *dptr = reinterpret_cast<T *>(dst.addr);
    auto *sptr = reinterpret_cast<const T *>(src.addr);
    if (lc_tensor_contiguous(dst) && lc_tensor_contiguous(src)) {
        for (lc_uint i = tid; i < count; i += total) {
            dptr[dst.offset + i] = sptr[src.offset + i];
        }
    } else {
        lc_uint cd[4], cs[4];
        for (lc_uint i = tid; i < count; i += total) {
            auto dof = lc_tensor_decompose_offset(dst, i, cd);
            auto sof = lc_tensor_decompose_offset(src, i, cs);
            dptr[dof] = sptr[sof];
        }
    }
}

__device__ __forceinline__ void lc_tensor_copy(
    lc_uint ddt, lc_uint drk, lc_uint4 dex, lc_uint4 dst_, lc_uint dof, lc_ulong dad,
    lc_uint sdt, lc_uint srk, lc_uint4 sex, lc_uint4 sst, lc_uint sof, lc_ulong sad,
    lc_uint count) noexcept {
    LCTensorDesc dst{ddt, drk, dex, dst_, dof, dad};
    LCTensorDesc src{sdt, srk, sex, sst, sof, sad};
    switch (ddt) {
        case 0: lc_tensor_copy_typed<half>(dst, src, count); break;
        case 1: lc_tensor_copy_typed<float>(dst, src, count); break;
        case 2: lc_tensor_copy_typed<int>(dst, src, count); break;
        default: lc_trap(); break;
    }
}

template<typename T>
__device__ __forceinline__ void lc_tensor_fill_typed(
    const LCTensorDesc &dst, lc_uint value_bits, lc_uint count) noexcept {
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *dptr = reinterpret_cast<T *>(dst.addr);
    auto value = lc_tensor_bits_to<T>(value_bits);
    if (lc_tensor_contiguous(dst)) {
        for (lc_uint i = tid; i < count; i += total) {
            dptr[dst.offset + i] = value;
        }
    } else {
        lc_uint cd[4];
        for (lc_uint i = tid; i < count; i += total) {
            dptr[lc_tensor_decompose_offset(dst, i, cd)] = value;
        }
    }
}

__device__ __forceinline__ void lc_tensor_fill(
    lc_uint ddt, lc_uint drk, lc_uint4 dex, lc_uint4 dst_, lc_uint dof, lc_ulong dad,
    lc_uint value_bits, lc_uint count) noexcept {
    LCTensorDesc dst{ddt, drk, dex, dst_, dof, dad};
    switch (ddt) {
        case 0: lc_tensor_fill_typed<half>(dst, value_bits, count); break;
        case 1: lc_tensor_fill_typed<float>(dst, value_bits, count); break;
        case 2: lc_tensor_fill_typed<int>(dst, value_bits, count); break;
        default: lc_trap(); break;
    }
}

// dtype pair dispatch for cast
template<int D, int S>
struct LCTensorCast {
    __device__ __forceinline__ static typename LCTensorScalar<D>::type apply(
        typename LCTensorScalar<S>::type v) noexcept;
};
#define LC_TENSOR_CAST_SPEC(D, S, EXPR)                                                 \
    template<>                                                                          \
    struct LCTensorCast<D, S> {                                                         \
        __device__ __forceinline__ static typename LCTensorScalar<D>::type apply(       \
            typename LCTensorScalar<S>::type v) noexcept {                              \
            return (EXPR);                                                              \
        }                                                                               \
    };
LC_TENSOR_CAST_SPEC(0, 1, __float2half(v))
LC_TENSOR_CAST_SPEC(1, 0, __half2float(v))
LC_TENSOR_CAST_SPEC(0, 2, __float2half(static_cast<float>(v)))
LC_TENSOR_CAST_SPEC(2, 0, static_cast<int>(__half2float(v)))
LC_TENSOR_CAST_SPEC(1, 2, static_cast<float>(v))
LC_TENSOR_CAST_SPEC(2, 1, static_cast<int>(v))
LC_TENSOR_CAST_SPEC(0, 0, v)
LC_TENSOR_CAST_SPEC(1, 1, v)
LC_TENSOR_CAST_SPEC(2, 2, v)
#undef LC_TENSOR_CAST_SPEC

template<int D, int S>
__device__ __forceinline__ void lc_tensor_cast_typed(
    const LCTensorDesc &dst, const LCTensorDesc &src, lc_uint count) noexcept {
    using DT = typename LCTensorScalar<D>::type;
    using ST = typename LCTensorScalar<S>::type;
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *dptr = reinterpret_cast<DT *>(dst.addr);
    auto *sptr = reinterpret_cast<const ST *>(src.addr);
    if (lc_tensor_contiguous(dst) && lc_tensor_contiguous(src)) {
        for (lc_uint i = tid; i < count; i += total) {
            dptr[dst.offset + i] = LCTensorCast<D, S>::apply(sptr[src.offset + i]);
        }
    } else {
        lc_uint cd[4], cs[4];
        for (lc_uint i = tid; i < count; i += total) {
            dptr[lc_tensor_decompose_offset(dst, i, cd)] =
                LCTensorCast<D, S>::apply(sptr[lc_tensor_decompose_offset(src, i, cs)]);
        }
    }
}

__device__ __forceinline__ void lc_tensor_cast(
    lc_uint ddt, lc_uint drk, lc_uint4 dex, lc_uint4 dst_, lc_uint dof, lc_ulong dad,
    lc_uint sdt, lc_uint srk, lc_uint4 sex, lc_uint4 sst, lc_uint sof, lc_ulong sad,
    lc_uint count) noexcept {
    LCTensorDesc dst{ddt, drk, dex, dst_, dof, dad};
    LCTensorDesc src{sdt, srk, sex, sst, sof, sad};
    switch (ddt * 10u + sdt) {
        case 0u: lc_tensor_cast_typed<0, 0>(dst, src, count); break;
        case 1u: lc_tensor_cast_typed<0, 1>(dst, src, count); break;
        case 2u: lc_tensor_cast_typed<0, 2>(dst, src, count); break;
        case 10u: lc_tensor_cast_typed<1, 0>(dst, src, count); break;
        case 11u: lc_tensor_cast_typed<1, 1>(dst, src, count); break;
        case 12u: lc_tensor_cast_typed<1, 2>(dst, src, count); break;
        case 20u: lc_tensor_cast_typed<2, 0>(dst, src, count); break;
        case 21u: lc_tensor_cast_typed<2, 1>(dst, src, count); break;
        case 22u: lc_tensor_cast_typed<2, 2>(dst, src, count); break;
        default: lc_trap(); break;
    }
}

template<typename T>
__device__ __forceinline__ void lc_tensor_permute_typed(
    const LCTensorDesc &dst, const LCTensorDesc &src, lc_uint4 perm) noexcept {
    auto count = lc_tensor_numel(dst);
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *dptr = reinterpret_cast<T *>(dst.addr);
    auto *sptr = reinterpret_cast<const T *>(src.addr);
    lc_uint cs[4], cd[4];
    for (lc_uint i = tid; i < count; i += total) {
        auto sof = lc_tensor_decompose_offset(src, i, cs);
        for (int d = 0; d < static_cast<int>(dst.rank); ++d) {
            cd[d] = cs[(&perm.x)[d]];
        }
        dptr[lc_tensor_offset_from_coords(dst, cd)] = sptr[sof];
    }
}

__device__ __forceinline__ void lc_tensor_permute(
    lc_uint ddt, lc_uint drk, lc_uint4 dex, lc_uint4 dst_, lc_uint dof, lc_ulong dad,
    lc_uint sdt, lc_uint srk, lc_uint4 sex, lc_uint4 sst, lc_uint sof, lc_ulong sad,
    lc_uint4 perm) noexcept {
    LCTensorDesc dst{ddt, drk, dex, dst_, dof, dad};
    LCTensorDesc src{sdt, srk, sex, sst, sof, sad};
    switch (ddt) {
        case 0: lc_tensor_permute_typed<half>(dst, src, perm); break;
        case 1: lc_tensor_permute_typed<float>(dst, src, perm); break;
        case 2: lc_tensor_permute_typed<int>(dst, src, perm); break;
        default: lc_trap(); break;
    }
}

template<typename T>
__device__ __forceinline__ void lc_tensor_concat_typed(
    const LCTensorDesc &dst, lc_uint dim, lc_uint num_src,
    const LCTensorDesc *srcs, lc_uint count) noexcept {
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *dptr = reinterpret_cast<T *>(dst.addr);
    lc_uint cd[4];
    for (lc_uint i = tid; i < count; i += total) {
        auto dof = lc_tensor_decompose_offset(dst, i, cd);
        lc_uint cursor = 0u;
        for (lc_uint s = 0u; s < num_src; ++s) {
            auto &sd = srcs[s];
            auto se = lc_tensor_extent(sd, static_cast<int>(dim));
            if (cd[dim] < cursor + se) {
                lc_uint cs[4];
                for (int d = 0; d < static_cast<int>(dst.rank); ++d) {
                    cs[d] = cd[d];
                }
                cs[dim] = cd[dim] - cursor;
                auto *sptr = reinterpret_cast<const T *>(sd.addr);
                dptr[dof] = sptr[lc_tensor_offset_from_coords(sd, cs)];
                break;
            }
            cursor += se;
        }
    }
}

__device__ __forceinline__ void lc_tensor_concat(
    lc_uint ddt, lc_uint drk, lc_uint4 dex, lc_uint4 dst_, lc_uint dof, lc_ulong dad,
    lc_uint dim, lc_uint num_src,
    lc_uint s0dt, lc_uint s0rk, lc_uint4 s0ex, lc_uint4 s0st, lc_uint s0of, lc_ulong s0ad,
    lc_uint s1dt, lc_uint s1rk, lc_uint4 s1ex, lc_uint4 s1st, lc_uint s1of, lc_ulong s1ad,
    lc_uint s2dt, lc_uint s2rk, lc_uint4 s2ex, lc_uint4 s2st, lc_uint s2of, lc_ulong s2ad,
    lc_uint s3dt, lc_uint s3rk, lc_uint4 s3ex, lc_uint4 s3st, lc_uint s3of, lc_ulong s3ad,
    lc_uint s4dt, lc_uint s4rk, lc_uint4 s4ex, lc_uint4 s4st, lc_uint s4of, lc_ulong s4ad,
    lc_uint s5dt, lc_uint s5rk, lc_uint4 s5ex, lc_uint4 s5st, lc_uint s5of, lc_ulong s5ad,
    lc_uint s6dt, lc_uint s6rk, lc_uint4 s6ex, lc_uint4 s6st, lc_uint s6of, lc_ulong s6ad,
    lc_uint s7dt, lc_uint s7rk, lc_uint4 s7ex, lc_uint4 s7st, lc_uint s7of, lc_ulong s7ad) noexcept {
    LCTensorDesc dst{ddt, drk, dex, dst_, dof, dad};
    LCTensorDesc srcs[8] = {
        {s0dt, s0rk, s0ex, s0st, s0of, s0ad},
        {s1dt, s1rk, s1ex, s1st, s1of, s1ad},
        {s2dt, s2rk, s2ex, s2st, s2of, s2ad},
        {s3dt, s3rk, s3ex, s3st, s3of, s3ad},
        {s4dt, s4rk, s4ex, s4st, s4of, s4ad},
        {s5dt, s5rk, s5ex, s5st, s5of, s5ad},
        {s6dt, s6rk, s6ex, s6st, s6of, s6ad},
        {s7dt, s7rk, s7ex, s7st, s7of, s7ad},
    };
    auto count = lc_tensor_numel(dst);
    switch (ddt) {
        case 0: lc_tensor_concat_typed<half>(dst, dim, num_src, srcs, count); break;
        case 1: lc_tensor_concat_typed<float>(dst, dim, num_src, srcs, count); break;
        case 2: lc_tensor_concat_typed<int>(dst, dim, num_src, srcs, count); break;
        default: lc_trap(); break;
    }
}

template<typename T>
__device__ __forceinline__ void lc_tensor_pad_typed(
    const LCTensorDesc &dst, const LCTensorDesc &src, lc_uint4 pad) noexcept {
    auto count = lc_tensor_numel(dst);
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *dptr = reinterpret_cast<T *>(dst.addr);
    auto *sptr = reinterpret_cast<const T *>(src.addr);
    lc_uint cd[4], cs[4];
    for (lc_uint i = tid; i < count; i += total) {
        auto dof = lc_tensor_decompose_offset(dst, i, cd);
        bool inside = true;
        for (int d = 0; d < static_cast<int>(dst.rank); ++d) {
            auto p = (&pad.x)[d];
            auto se = lc_tensor_extent(src, d);
            auto c = static_cast<lc_uint>(cd[d]) - p;
            if (c >= se) { inside = false; break; }
            cs[d] = c;
        }
        if (inside) {
            dptr[dof] = sptr[lc_tensor_offset_from_coords(src, cs)];
        } else {
            dptr[dof] = static_cast<T>(0);
        }
    }
}

__device__ __forceinline__ void lc_tensor_pad(
    lc_uint ddt, lc_uint drk, lc_uint4 dex, lc_uint4 dst_, lc_uint dof, lc_ulong dad,
    lc_uint sdt, lc_uint srk, lc_uint4 sex, lc_uint4 sst, lc_uint sof, lc_ulong sad,
    lc_uint4 pad) noexcept {
    LCTensorDesc dst{ddt, drk, dex, dst_, dof, dad};
    LCTensorDesc src{sdt, srk, sex, sst, sof, sad};
    switch (ddt) {
        case 0: lc_tensor_pad_typed<half>(dst, src, pad); break;
        case 1: lc_tensor_pad_typed<float>(dst, src, pad); break;
        case 2: lc_tensor_pad_typed<int>(dst, src, pad); break;
        default: lc_trap(); break;
    }
}

// ---------------------------------------------------------------------------
// Element-wise unary / binary / ternary
// ---------------------------------------------------------------------------

template<typename T, typename Fn>
__device__ __forceinline__ void lc_tensor_unary_loop(
    const LCTensorDesc &out, const LCTensorDesc &in, lc_uint count, Fn fn) noexcept {
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *optr = reinterpret_cast<T *>(out.addr);
    auto *iptr = reinterpret_cast<const T *>(in.addr);
    if (lc_tensor_contiguous(out) && lc_tensor_contiguous(in)) {
        for (lc_uint i = tid; i < count; i += total) {
            optr[out.offset + i] = fn(iptr[in.offset + i]);
        }
    } else {
        lc_uint co[4], ci[4];
        for (lc_uint i = tid; i < count; i += total) {
            auto oof = lc_tensor_decompose_offset(out, i, co);
            auto iof = lc_tensor_decompose_offset(in, i, ci);
            optr[oof] = fn(iptr[iof]);
        }
    }
}

template<typename T, typename Fn>
__device__ __forceinline__ void lc_tensor_binary_loop(
    const LCTensorDesc &out, const LCTensorDesc &a, const LCTensorDesc &b,
    lc_uint count, Fn fn) noexcept {
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *optr = reinterpret_cast<T *>(out.addr);
    auto *aptr = reinterpret_cast<const T *>(a.addr);
    auto *bptr = reinterpret_cast<const T *>(b.addr);
    if (lc_tensor_contiguous(out) && lc_tensor_contiguous(a) && lc_tensor_contiguous(b)) {
        for (lc_uint i = tid; i < count; i += total) {
            optr[out.offset + i] = fn(aptr[a.offset + i], bptr[b.offset + i]);
        }
    } else {
        lc_uint co[4], ca[4], cb[4];
        for (lc_uint i = tid; i < count; i += total) {
            auto oof = lc_tensor_decompose_offset(out, i, co);
            auto aof = lc_tensor_decompose_offset(a, i, ca);
            auto bof = lc_tensor_decompose_offset(b, i, cb);
            optr[oof] = fn(aptr[aof], bptr[bof]);
        }
    }
}

template<typename T, typename Fn>
__device__ __forceinline__ void lc_tensor_ternary_loop(
    const LCTensorDesc &out, const LCTensorDesc &a, const LCTensorDesc &b,
    const LCTensorDesc &c, lc_uint count, Fn fn) noexcept {
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *optr = reinterpret_cast<T *>(out.addr);
    auto *aptr = reinterpret_cast<const T *>(a.addr);
    auto *bptr = reinterpret_cast<const T *>(b.addr);
    auto *cptr = reinterpret_cast<const T *>(c.addr);
    if (lc_tensor_contiguous(out) && lc_tensor_contiguous(a) &&
        lc_tensor_contiguous(b) && lc_tensor_contiguous(c)) {
        for (lc_uint i = tid; i < count; i += total) {
            optr[out.offset + i] = fn(aptr[a.offset + i], bptr[b.offset + i], cptr[c.offset + i]);
        }
    } else {
        lc_uint co[4], ca[4], cb[4], cc[4];
        for (lc_uint i = tid; i < count; i += total) {
            auto oof = lc_tensor_decompose_offset(out, i, co);
            auto aof = lc_tensor_decompose_offset(a, i, ca);
            auto bof = lc_tensor_decompose_offset(b, i, cb);
            auto cof = lc_tensor_decompose_offset(c, i, cc);
            optr[oof] = fn(aptr[aof], bptr[bof], cptr[cof]);
        }
    }
}

#define LC_TENSOR_UNARY_IMPL(NAME, EXPR)                                               \
    template<typename T>                                                               \
    __device__ __forceinline__ void lc_tensor_##NAME##_typed(                          \
        const LCTensorDesc &out, const LCTensorDesc &in, lc_uint count) noexcept {     \
        lc_tensor_unary_loop<T>(out, in, count, [](T x) -> T { return (EXPR); });      \
    }                                                                                  \
    __device__ __forceinline__ void lc_tensor_##NAME(                                  \
        lc_uint odt, lc_uint ork, lc_uint4 oex, lc_uint4 ost, lc_uint oof, lc_ulong oad,     \
        lc_uint idt, lc_uint irk, lc_uint4 iex, lc_uint4 ist, lc_uint iof, lc_ulong iad,     \
        lc_uint count) noexcept {                                                      \
        LCTensorDesc out{odt, ork, oex, ost, oof, oad};                                \
        LCTensorDesc in{idt, irk, iex, ist, iof, iad};                                 \
        switch (odt) {                                                                 \
            case 0: lc_tensor_##NAME##_typed<half>(out, in, count); break;             \
            case 1: lc_tensor_##NAME##_typed<float>(out, in, count); break;            \
            case 2: lc_tensor_##NAME##_typed<int>(out, in, count); break;              \
            default: lc_trap(); break;                                                 \
        }                                                                              \
    }

LC_TENSOR_UNARY_IMPL(neg, -x)
LC_TENSOR_UNARY_IMPL(abs, lc_tensor_abs_generic(x))
LC_TENSOR_UNARY_IMPL(exp, lc_tensor_exp(x))
LC_TENSOR_UNARY_IMPL(log, lc_tensor_log(x))
LC_TENSOR_UNARY_IMPL(sqrt, lc_tensor_sqrt(x))
LC_TENSOR_UNARY_IMPL(rsqrt, lc_tensor_rsqrt(x))
LC_TENSOR_UNARY_IMPL(sin, lc_tensor_sin(x))
LC_TENSOR_UNARY_IMPL(cos, lc_tensor_cos(x))
LC_TENSOR_UNARY_IMPL(tan, lc_tensor_tan(x))
LC_TENSOR_UNARY_IMPL(tanh, lc_tensor_tanh(x))
LC_TENSOR_UNARY_IMPL(sigmoid, lc_tensor_sigmoid(x))
LC_TENSOR_UNARY_IMPL(gelu, lc_tensor_gelu(x))
LC_TENSOR_UNARY_IMPL(relu, lc_tensor_relu(x))
LC_TENSOR_UNARY_IMPL(leaky_relu, lc_tensor_leaky_relu(x))
LC_TENSOR_UNARY_IMPL(erf, lc_tensor_erf(x))
LC_TENSOR_UNARY_IMPL(ceil, lc_tensor_ceil(x))
LC_TENSOR_UNARY_IMPL(floor, lc_tensor_floor(x))
LC_TENSOR_UNARY_IMPL(round, lc_tensor_round(x))
LC_TENSOR_UNARY_IMPL(isnan, static_cast<T>(lc_tensor_isnan_impl(x) ? 1 : 0))
LC_TENSOR_UNARY_IMPL(isinf, static_cast<T>(lc_tensor_isinf_impl(x) ? 1 : 0))

#undef LC_TENSOR_UNARY_IMPL

#define LC_TENSOR_BINARY_IMPL(NAME, EXPR)                                              \
    template<typename T>                                                               \
    __device__ __forceinline__ void lc_tensor_##NAME##_typed(                          \
        const LCTensorDesc &out, const LCTensorDesc &a, const LCTensorDesc &b,         \
        lc_uint count) noexcept {                                                      \
        lc_tensor_binary_loop<T>(out, a, b, count,                                     \
                                 [](T x, T y) -> T { return (EXPR); });                \
    }                                                                                  \
    __device__ __forceinline__ void lc_tensor_##NAME(                                  \
        lc_uint odt, lc_uint ork, lc_uint4 oex, lc_uint4 ost, lc_uint oof, lc_ulong oad,     \
        lc_uint adt, lc_uint ark, lc_uint4 aex, lc_uint4 ast, lc_uint aof, lc_ulong aad,     \
        lc_uint bdt, lc_uint brk, lc_uint4 bex, lc_uint4 bst, lc_uint bof, lc_ulong bad,     \
        lc_uint count) noexcept {                                                      \
        LCTensorDesc out{odt, ork, oex, ost, oof, oad};                                \
        LCTensorDesc a{adt, ark, aex, ast, aof, aad};                                  \
        LCTensorDesc b{bdt, brk, bex, bst, bof, bad};                                  \
        switch (odt) {                                                                 \
            case 0: lc_tensor_##NAME##_typed<half>(out, a, b, count); break;           \
            case 1: lc_tensor_##NAME##_typed<float>(out, a, b, count); break;          \
            case 2: lc_tensor_##NAME##_typed<int>(out, a, b, count); break;            \
            default: lc_trap(); break;                                                 \
        }                                                                              \
    }

LC_TENSOR_BINARY_IMPL(add, x + y)
LC_TENSOR_BINARY_IMPL(sub, x - y)
LC_TENSOR_BINARY_IMPL(mul, x * y)
LC_TENSOR_BINARY_IMPL(div, x / y)
LC_TENSOR_BINARY_IMPL(pow, lc_tensor_pow(x, y))
LC_TENSOR_BINARY_IMPL(min, lc_tensor_min_generic(x, y))
LC_TENSOR_BINARY_IMPL(max, lc_tensor_max_generic(x, y))

#undef LC_TENSOR_BINARY_IMPL

__device__ __forceinline__ void lc_tensor_clamp(
    lc_uint odt, lc_uint ork, lc_uint4 oex, lc_uint4 ost, lc_uint oof, lc_ulong oad,
    lc_uint idt, lc_uint irk, lc_uint4 iex, lc_uint4 ist, lc_uint iof, lc_ulong iad,
    lc_uint lo_bits, lc_uint hi_bits, lc_uint count) noexcept {
    LCTensorDesc out{odt, ork, oex, ost, oof, oad};
    LCTensorDesc in{idt, irk, iex, ist, iof, iad};
    switch (odt) {
        case 0: {
            auto lo = lc_tensor_bits_to<half>(lo_bits);
            auto hi = lc_tensor_bits_to<half>(hi_bits);
            lc_tensor_unary_loop<half>(out, in, count, [lo, hi](half x) -> half {
                return lc_tensor_min_generic(lc_tensor_max_generic(x, lo), hi);
            });
            break;
        }
        case 1: {
            auto lo = lc_tensor_bits_to<float>(lo_bits);
            auto hi = lc_tensor_bits_to<float>(hi_bits);
            lc_tensor_unary_loop<float>(out, in, count, [lo, hi](float x) -> float {
                return fminf(fmaxf(x, lo), hi);
            });
            break;
        }
        case 2: {
            auto lo = lc_tensor_bits_to<int>(lo_bits);
            auto hi = lc_tensor_bits_to<int>(hi_bits);
            lc_tensor_unary_loop<int>(out, in, count, [lo, hi](int x) -> int {
                return lc_tensor_min_generic(lc_tensor_max_generic(x, lo), hi);
            });
            break;
        }
        default: lc_trap(); break;
    }
}

__device__ __forceinline__ void lc_tensor_fma(
    lc_uint odt, lc_uint ork, lc_uint4 oex, lc_uint4 ost, lc_uint oof, lc_ulong oad,
    lc_uint adt, lc_uint ark, lc_uint4 aex, lc_uint4 ast, lc_uint aof, lc_ulong aad,
    lc_uint bdt, lc_uint brk, lc_uint4 bex, lc_uint4 bst, lc_uint bof, lc_ulong bad,
    lc_uint cdt, lc_uint crk, lc_uint4 cex, lc_uint4 cst, lc_uint cof, lc_ulong cad,
    lc_uint count) noexcept {
    LCTensorDesc out{odt, ork, oex, ost, oof, oad};
    LCTensorDesc a{adt, ark, aex, ast, aof, aad};
    LCTensorDesc b{bdt, brk, bex, bst, bof, bad};
    LCTensorDesc c{cdt, crk, cex, cst, cof, cad};
    switch (odt) {
        case 0: lc_tensor_ternary_loop<half>(out, a, b, c, count,
                                             [](half x, half y, half z) -> half { return lc_tensor_fma_impl(x, y, z); });
            break;
        case 1: lc_tensor_ternary_loop<float>(out, a, b, c, count,
                                              [](float x, float y, float z) -> float { return lc_tensor_fma_impl(x, y, z); });
            break;
        case 2: lc_tensor_ternary_loop<int>(out, a, b, c, count,
                                            [](int x, int y, int z) -> int { return lc_tensor_fma_impl(x, y, z); });
            break;
        default: lc_trap(); break;
    }
}

// ---------------------------------------------------------------------------
// Reductions / scans
// ---------------------------------------------------------------------------

enum LCTensorReduceOp : int {
    LC_TENSOR_REDUCE_SUM = 0,
    LC_TENSOR_REDUCE_MAX = 1,
    LC_TENSOR_REDUCE_MIN = 2,
};

template<typename T>
__device__ __forceinline__ T lc_tensor_reduce_combine(int op, T a, T b) noexcept {
    switch (op) {
        case LC_TENSOR_REDUCE_SUM: return a + b;
        case LC_TENSOR_REDUCE_MAX: return lc_tensor_max_generic(a, b);
        default: return lc_tensor_min_generic(a, b);
    }
}

template<typename T>
__device__ __forceinline__ T lc_tensor_reduce_initial(int op) noexcept {
    switch (op) {
        case LC_TENSOR_REDUCE_SUM: return static_cast<T>(0);
        case LC_TENSOR_REDUCE_MAX: return lc_tensor_from_float<T>(0, -3.402823466e+38f);
        default: return lc_tensor_from_float<T>(0, 3.402823466e+38f);
    }
}

template<typename T>
__device__ __forceinline__ void lc_tensor_reduce_typed(
    const LCTensorDesc &out, const LCTensorDesc &in, lc_uint4 dims, int op) noexcept {
    auto out_count = lc_tensor_numel(out);
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *optr = reinterpret_cast<T *>(out.addr);
    auto *iptr = reinterpret_cast<const T *>(in.addr);
    lc_uint out_to_in[4];
    lc_uint num_non_reduced = 0u;
    for (int d = 0; d < static_cast<int>(in.rank); ++d) {
        if ((&dims.x)[d] == 0u) { out_to_in[num_non_reduced++] = static_cast<lc_uint>(d); }
    }
    lc_uint co[4], ci[4] = {0u, 0u, 0u, 0u};
    lc_uint reduce_count = 1u;
    for (int d = 0; d < static_cast<int>(in.rank); ++d) {
        if ((&dims.x)[d] != 0u) { reduce_count *= lc_tensor_extent(in, d); }
    }
    auto initial = lc_tensor_reduce_initial<T>(op);
    for (lc_uint oi = tid; oi < out_count; oi += total) {
        lc_tensor_decompose_offset(out, oi, co);
        for (lc_uint j = 0u; j < num_non_reduced; ++j) {
            ci[out_to_in[j]] = co[j];
        }
        T acc = initial;
        for (lc_uint r = 0u; r < reduce_count; ++r) {
            lc_uint rem = r;
            for (int d = static_cast<int>(in.rank) - 1; d >= 0; --d) {
                if ((&dims.x)[d] != 0u) {
                    auto e = lc_tensor_extent(in, d);
                    ci[d] = e > 0u ? rem % e : 0u;
                    rem /= e;
                }
            }
            acc = lc_tensor_reduce_combine(op, acc, iptr[lc_tensor_offset_from_coords(in, ci)]);
        }
        optr[lc_tensor_offset_from_coords(out, co)] = acc;
    }
}

#define LC_TENSOR_REDUCE_IMPL(NAME, OP)                                                \
    __device__ __forceinline__ void lc_tensor_reduce_##NAME(                           \
        lc_uint odt, lc_uint ork, lc_uint4 oex, lc_uint4 ost, lc_uint oof, lc_ulong oad,     \
        lc_uint idt, lc_uint irk, lc_uint4 iex, lc_uint4 ist, lc_uint iof, lc_ulong iad,     \
        lc_uint num_dims, lc_uint4 dims) noexcept {                                       \
        LCTensorDesc out{odt, ork, oex, ost, oof, oad};                                \
        LCTensorDesc in{idt, irk, iex, ist, iof, iad};                                 \
        (void)num_dims;                                                                \
        switch (odt) {                                                                 \
            case 0: lc_tensor_reduce_typed<half>(out, in, dims, (OP)); break;          \
            case 1: lc_tensor_reduce_typed<float>(out, in, dims, (OP)); break;         \
            case 2: lc_tensor_reduce_typed<int>(out, in, dims, (OP)); break;           \
            default: lc_trap(); break;                                                 \
        }                                                                              \
    }

LC_TENSOR_REDUCE_IMPL(sum, LC_TENSOR_REDUCE_SUM)
LC_TENSOR_REDUCE_IMPL(max, LC_TENSOR_REDUCE_MAX)
LC_TENSOR_REDUCE_IMPL(min, LC_TENSOR_REDUCE_MIN)

#undef LC_TENSOR_REDUCE_IMPL

template<typename T>
__device__ __forceinline__ void lc_tensor_cumsum_typed(
    const LCTensorDesc &out, const LCTensorDesc &in, lc_uint dim) noexcept {
    auto numel = lc_tensor_numel(in);
    auto dim_extent = lc_tensor_extent(in, static_cast<int>(dim));
    auto fibers = dim_extent > 0u ? numel / dim_extent : 0u;
    lc_uint tid, total;
    lc_tensor_grid_stride_begin(tid, total);
    auto *optr = reinterpret_cast<T *>(out.addr);
    auto *iptr = reinterpret_cast<const T *>(in.addr);
    lc_uint ci[4] = {0u, 0u, 0u, 0u};
    for (lc_uint f = tid; f < fibers; f += total) {
        lc_uint rem = f;
        for (int d = static_cast<int>(in.rank) - 1; d >= 0; --d) {
            if (static_cast<lc_uint>(d) == dim) { continue; }
            auto e = lc_tensor_extent(in, d);
            ci[d] = e > 0u ? rem % e : 0u;
            rem /= e;
        }
        T acc = static_cast<T>(0);
        for (lc_uint s = 0u; s < dim_extent; ++s) {
            ci[dim] = s;
            auto idx = lc_tensor_offset_from_coords(in, ci);
            acc = acc + iptr[idx];
            optr[idx] = acc;
        }
    }
}

__device__ __forceinline__ void lc_tensor_cumsum(
    lc_uint odt, lc_uint ork, lc_uint4 oex, lc_uint4 ost, lc_uint oof, lc_ulong oad,
    lc_uint idt, lc_uint irk, lc_uint4 iex, lc_uint4 ist, lc_uint iof, lc_ulong iad,
    lc_uint dim) noexcept {
    LCTensorDesc out{odt, ork, oex, ost, oof, oad};
    LCTensorDesc in{idt, irk, iex, ist, iof, iad};
    switch (odt) {
        case 0: lc_tensor_cumsum_typed<half>(out, in, dim); break;
        case 1: lc_tensor_cumsum_typed<float>(out, in, dim); break;
        case 2: lc_tensor_cumsum_typed<int>(out, in, dim); break;
        default: lc_trap(); break;
    }
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

// Scalar GEMM with an FP32 accumulator. `InT` is the input element type
// (half for the F16 path, float for the F32 path); C is always written as
// float (plan.md §A.3: the F16 GEMM writes an FP32 accumulator).
template<typename InT>
__device__ __forceinline__ void lc_tensor_gemm_scalar_typed(
    const LCTensorDesc &c, const LCTensorDesc &a, const LCTensorDesc &b,
    float alpha, float beta, lc_uint epilogue) noexcept {
    auto M = lc_tensor_extent(a, 0);
    auto K = lc_tensor_extent(a, 1);
    auto N = lc_tensor_extent(b, 1);
    auto total = M * N;
    lc_uint tid, total_threads;
    lc_tensor_grid_stride_begin(tid, total_threads);
    auto *cptr = reinterpret_cast<float *>(c.addr);
    auto *aptr = reinterpret_cast<const InT *>(a.addr);
    auto *bptr = reinterpret_cast<const InT *>(b.addr);
    auto a_ld = lc_tensor_stride(a, 0);
    auto b_ld = lc_tensor_stride(b, 0);
    auto c_ld = lc_tensor_stride(c, 0);
    for (lc_uint idx = tid; idx < total; idx += total_threads) {
        auto row = idx / N;
        auto col = idx - row * N;
        float sum = 0.0f;
        for (lc_uint k = 0u; k < K; ++k) {
            sum += lc_tensor_to_float(aptr[a.offset + row * a_ld + k]) *
                   lc_tensor_to_float(bptr[b.offset + k * b_ld + col]);
        }
        auto out_idx = c.offset + row * c_ld + col;
        auto v = alpha * sum + beta * cptr[out_idx];
        if (epilogue == 1u) { v = fmaxf(v, 0.0f); }
        cptr[out_idx] = v;
    }
}

// Load two adjacent halfs (low, high) as one 32-bit register (plan.md §A.4).
__device__ __forceinline__ unsigned lc_tensor_load_half2(const half *p) noexcept {
    return *reinterpret_cast<const unsigned *>(p);
}

// Pack two halfs (lo, hi) into one 32-bit register (plan.md §A.4).
__device__ __forceinline__ unsigned lc_tensor_pack_half2(half lo, half hi) noexcept {
    return static_cast<unsigned>(__half_as_ushort(lo)) |
           (static_cast<unsigned>(__half_as_ushort(hi)) << 16);
}

// Classic tensor-core MMA: D(16x8) = A(16x16) * B(16x8) + C(16x8) with the
// fragment layout of plan.md §A.4 (g = lane>>2, t = lane%4).
__device__ __forceinline__ void lc_tensor_mma_m16n8k16(float *d, const unsigned *a,
                                                       const unsigned *b) noexcept {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// F16 tensor-core GEMM body with an explicit flat tile id (one warp per
// 16x16 output tile, FP32 accumulator — plan.md §A.1 / §A.4). Mirrors
// `wmma_gemm_f16` from the dispatch test.
__device__ __forceinline__ void lc_tensor_gemm_f16_wmma_tiled(
    const LCTensorDesc &c, const LCTensorDesc &a, const LCTensorDesc &b,
    lc_uint tile_id, lc_uint tiles_n, float alpha, float beta, lc_uint epilogue) noexcept {
    constexpr lc_uint TILE = 16u;
    auto M = lc_tensor_extent(a, 0);
    auto K = lc_tensor_extent(a, 1);
    auto N = lc_tensor_extent(b, 1);
    auto a_ld = lc_tensor_stride(a, 0);
    auto b_ld = lc_tensor_stride(b, 0);
    auto c_ld = lc_tensor_stride(c, 0);
    auto tile_m = tile_id / tiles_n;
    auto tile_n = tile_id - tile_m * tiles_n;
    auto lane = threadIdx.x;
    if (lane >= 32u) { return; }
    auto g = lane >> 2u;
    auto t = lane & 3u;
    auto m_off = tile_m * TILE;
    auto n_off = tile_n * TILE;
    auto *A = reinterpret_cast<const half *>(a.addr);
    auto *B = reinterpret_cast<const half *>(b.addr);
    auto *C = reinterpret_cast<float *>(c.addr);
    float acc[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[h][i] = 0.0f; }
    }
    if (m_off < M && n_off < N) {
        for (lc_uint k = 0u; k < K; k += TILE) {
            if (k + TILE > K) { break; }
            const half *a_base = A + (m_off + g) * a_ld + k + t * 2u;
            unsigned af[4];
            af[0] = lc_tensor_load_half2(a_base);
            af[1] = lc_tensor_load_half2(a_base + 8u * a_ld);
            af[2] = lc_tensor_load_half2(a_base + 8u);
            af[3] = lc_tensor_load_half2(a_base + 8u * a_ld + 8u);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                auto col = n_off + static_cast<lc_uint>(h) * 8u + g;
                const half *b_base = B + (k + t * 2u) * b_ld + col;
                unsigned bf[2];
                bf[0] = lc_tensor_pack_half2(b_base[0], b_base[b_ld]);
                bf[1] = lc_tensor_pack_half2(b_base[8u * b_ld], b_base[9u * b_ld]);
                lc_tensor_mma_m16n8k16(acc[h], af, bf);
            }
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        auto col0 = n_off + static_cast<lc_uint>(h) * 8u + t * 2u;
        if (col0 + 1u >= N) { break; }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            auto row = m_off + g + static_cast<lc_uint>(i) * 8u;
            if (row >= M) { break; }
            auto idx0 = c.offset + row * c_ld + col0;
            float v0 = alpha * acc[h][i * 2 + 0] + beta * C[idx0];
            float v1 = alpha * acc[h][i * 2 + 1] + beta * C[idx0 + 1u];
            if (epilogue == 1u) {
                v0 = fmaxf(v0, 0.0f);
                v1 = fmaxf(v1, 0.0f);
            }
            C[idx0] = v0;
            C[idx0 + 1u] = v1;
        }
    }
}

__device__ __forceinline__ void lc_tensor_gemm_f16_wmma(
    const LCTensorDesc &c, const LCTensorDesc &a, const LCTensorDesc &b,
    float alpha, float beta, lc_uint epilogue) noexcept {
    auto N = lc_tensor_extent(b, 1);
    auto tiles_n = (N + 15u) / 16u;
    lc_tensor_gemm_f16_wmma_tiled(c, a, b, blockIdx.x, tiles_n, alpha, beta, epilogue);
}

__device__ __forceinline__ void lc_tensor_matmul(
    lc_uint cdt, lc_uint crk, lc_uint4 cex, lc_uint4 cst, lc_uint cof, lc_ulong cad,
    lc_uint adt, lc_uint ark, lc_uint4 aex, lc_uint4 ast, lc_uint aof, lc_ulong aad,
    lc_uint bdt, lc_uint brk, lc_uint4 bex, lc_uint4 bst, lc_uint bof, lc_ulong bad,
    lc_uint compute_dtype, lc_uint trans_a, lc_uint trans_b,
    float alpha, float beta, lc_uint epilogue) noexcept {
    LCTensorDesc c{cdt, crk, cex, cst, cof, cad};
    LCTensorDesc a{adt, ark, aex, ast, aof, aad};
    LCTensorDesc b{bdt, brk, bex, bst, bof, bad};
    (void)compute_dtype;
    lc_assert(trans_a == 0u && trans_b == 0u);
    if (adt == 0u) {
        auto M = lc_tensor_extent(a, 0);
        auto N = lc_tensor_extent(b, 1);
        auto K = lc_tensor_extent(a, 1);
        if ((M & 15u) == 0u && (N & 15u) == 0u && (K & 15u) == 0u) {
            lc_tensor_gemm_f16_wmma(c, a, b, alpha, beta, epilogue);
        } else {
            lc_tensor_gemm_scalar_typed<half>(c, a, b, alpha, beta, epilogue);
        }
    } else {
        lc_tensor_gemm_scalar_typed<float>(c, a, b, alpha, beta, epilogue);
    }
}

__device__ __forceinline__ void lc_tensor_batch_matmul(
    lc_uint cdt, lc_uint crk, lc_uint4 cex, lc_uint4 cst, lc_uint cof, lc_ulong cad,
    lc_uint adt, lc_uint ark, lc_uint4 aex, lc_uint4 ast, lc_uint aof, lc_ulong aad,
    lc_uint bdt, lc_uint brk, lc_uint4 bex, lc_uint4 bst, lc_uint bof, lc_ulong bad,
    lc_uint compute_dtype, lc_uint trans_a, lc_uint trans_b,
    float alpha, float beta, lc_uint epilogue, lc_uint batch) noexcept {
    LCTensorDesc c{cdt, crk, cex, cst, cof, cad};
    LCTensorDesc a{adt, ark, aex, ast, aof, aad};
    LCTensorDesc b{bdt, brk, bex, bst, bof, bad};
    (void)compute_dtype;
    lc_assert(trans_a == 0u && trans_b == 0u);
    auto M = lc_tensor_extent(a, 1);
    auto K = lc_tensor_extent(a, 2);
    auto N = lc_tensor_extent(b, 2);
    auto a_batch = lc_tensor_stride(a, 0);
    auto b_batch = lc_tensor_stride(b, 0);
    auto c_batch = lc_tensor_stride(c, 0);
    if (adt == 0u && (M & 15u) == 0u && (N & 15u) == 0u && (K & 15u) == 0u) {
        // One warp per (batch, tile_m, tile_n).
        constexpr lc_uint TILE = 16u;
        auto tiles_n = (N + TILE - 1u) / TILE;
        auto tiles_m = (M + TILE - 1u) / TILE;
        auto per_batch_tiles = tiles_m * tiles_n;
        auto tile_id = blockIdx.x;
        auto batch_id = tile_id / per_batch_tiles;
        if (batch_id >= batch) { return; }
        auto rem = tile_id - batch_id * per_batch_tiles;
        LCTensorDesc cb = c;
        LCTensorDesc ab = a;
        LCTensorDesc bb = b;
        cb.offset = c.offset + batch_id * c_batch;
        ab.offset = a.offset + batch_id * a_batch;
        bb.offset = b.offset + batch_id * b_batch;
        lc_tensor_gemm_f16_wmma_tiled(cb, ab, bb, rem, tiles_n, alpha, beta, epilogue);
    } else {
        // Scalar path: flat grid-stride over batch*M*N.
        auto total = batch * M * N;
        lc_uint tid, total_threads;
        lc_tensor_grid_stride_begin(tid, total_threads);
        auto a_ld = lc_tensor_stride(a, 1);
        auto b_ld = lc_tensor_stride(b, 1);
        auto c_ld = lc_tensor_stride(c, 1);
        if (adt == 0u) {
            auto *A = reinterpret_cast<const half *>(a.addr);
            auto *B = reinterpret_cast<const half *>(b.addr);
            auto *C = reinterpret_cast<float *>(c.addr);
            for (lc_uint idx = tid; idx < total; idx += total_threads) {
                auto b_id = idx / (M * N);
                auto mn = idx - b_id * (M * N);
                auto row = mn / N;
                auto col = mn - row * N;
                auto a_base = a.offset + b_id * a_batch;
                auto b_base = b.offset + b_id * b_batch;
                auto c_base = c.offset + b_id * c_batch;
                float sum = 0.0f;
                for (lc_uint k = 0u; k < K; ++k) {
                    sum += __half2float(A[a_base + row * a_ld + k]) *
                           __half2float(B[b_base + k * b_ld + col]);
                }
                auto o = c_base + row * c_ld + col;
                auto v = alpha * sum + beta * C[o];
                if (epilogue == 1u) { v = fmaxf(v, 0.0f); }
                C[o] = v;
            }
        } else {
            auto *A = reinterpret_cast<const float *>(a.addr);
            auto *B = reinterpret_cast<const float *>(b.addr);
            auto *C = reinterpret_cast<float *>(c.addr);
            for (lc_uint idx = tid; idx < total; idx += total_threads) {
                auto b_id = idx / (M * N);
                auto mn = idx - b_id * (M * N);
                auto row = mn / N;
                auto col = mn - row * N;
                auto a_base = a.offset + b_id * a_batch;
                auto b_base = b.offset + b_id * b_batch;
                auto c_base = c.offset + b_id * c_batch;
                float sum = 0.0f;
                for (lc_uint k = 0u; k < K; ++k) {
                    sum += A[a_base + row * a_ld + k] * B[b_base + k * b_ld + col];
                }
                auto o = c_base + row * c_ld + col;
                auto v = alpha * sum + beta * C[o];
                if (epilogue == 1u) { v = fmaxf(v, 0.0f); }
                C[o] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generalized contraction (einsum-style)
// ---------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ void lc_tensor_contract_typed(
    const LCTensorDesc &c, const LCTensorDesc &a, const LCTensorDesc &b,
    lc_uint4 ma, lc_uint4 mb, lc_uint4 mc) noexcept {
    auto c_count = lc_tensor_numel(c);
    lc_uint tid, total_threads;
    lc_tensor_grid_stride_begin(tid, total_threads);
    auto *cptr = reinterpret_cast<T *>(c.addr);
    auto *aptr = reinterpret_cast<const T *>(a.addr);
    auto *bptr = reinterpret_cast<const T *>(b.addr);
    lc_uint reduce_dims_a[8];
    lc_uint reduce_dims_b[8];
    lc_uint reduce_extents[8];
    lc_uint num_reduce = 0u;
    lc_uint reduce_count = 1u;
    for (lc_uint la = 0u; la < 4u; ++la) {
        auto label = (&ma.x)[la];
        if (label >= 255u) { continue; }
        bool in_b = false;
        lc_uint db = 0u;
        for (lc_uint lb = 0u; lb < 4u; ++lb) {
            if ((&mb.x)[lb] == label) { in_b = true; db = lb; break; }
        }
        if (!in_b) { continue; }
        bool in_c = false;
        for (lc_uint lc_ = 0u; lc_ < 4u; ++lc_) {
            if ((&mc.x)[lc_] == label) { in_c = true; break; }
        }
        if (in_c) { continue; }
        reduce_dims_a[num_reduce] = la;
        reduce_dims_b[num_reduce] = db;
        reduce_extents[num_reduce] = lc_tensor_extent(a, static_cast<int>(la));
        reduce_count *= reduce_extents[num_reduce];
        ++num_reduce;
    }
    lc_uint cc[4], ca[4], cb[4];
    for (lc_uint oi = tid; oi < c_count; oi += total_threads) {
        lc_tensor_decompose_offset(c, oi, cc);
        for (int d = 0; d < 4; ++d) { ca[d] = 0u; cb[d] = 0u; }
        for (lc_uint i = 0u; i < a.rank; ++i) {
            auto label = (&ma.x)[i];
            for (lc_uint j = 0u; j < c.rank; ++j) {
                if ((&mc.x)[j] == label) { ca[i] = cc[j]; break; }
            }
        }
        for (lc_uint i = 0u; i < b.rank; ++i) {
            auto label = (&mb.x)[i];
            for (lc_uint j = 0u; j < c.rank; ++j) {
                if ((&mc.x)[j] == label) { cb[i] = cc[j]; break; }
            }
        }
        float acc = 0.0f;
        for (lc_uint r = 0u; r < reduce_count; ++r) {
            lc_uint rem = r;
            for (lc_uint ri = num_reduce; ri-- > 0u;) {
                auto e = reduce_extents[ri];
                auto v = e > 0u ? rem % e : 0u;
                rem /= e;
                ca[reduce_dims_a[ri]] = v;
                cb[reduce_dims_b[ri]] = v;
            }
            acc += lc_tensor_to_float(aptr[lc_tensor_offset_from_coords(a, ca)]) *
                   lc_tensor_to_float(bptr[lc_tensor_offset_from_coords(b, cb)]);
        }
        cptr[lc_tensor_offset_from_coords(c, cc)] = lc_tensor_from_float<T>(0, acc);
    }
}

__device__ __forceinline__ void lc_tensor_contract(
    lc_uint cdt, lc_uint crk, lc_uint4 cex, lc_uint4 cst, lc_uint cof, lc_ulong cad,
    lc_uint adt, lc_uint ark, lc_uint4 aex, lc_uint4 ast, lc_uint aof, lc_ulong aad,
    lc_uint bdt, lc_uint brk, lc_uint4 bex, lc_uint4 bst, lc_uint bof, lc_ulong bad,
    lc_uint4 ma, lc_uint4 mb, lc_uint4 mc, lc_uint compute_dtype) noexcept {
    LCTensorDesc c{cdt, crk, cex, cst, cof, cad};
    LCTensorDesc a{adt, ark, aex, ast, aof, aad};
    LCTensorDesc b{bdt, brk, bex, bst, bof, bad};
    (void)compute_dtype;
    switch (cdt) {
        case 0: lc_tensor_contract_typed<half>(c, a, b, ma, mb, mc); break;
        case 1: lc_tensor_contract_typed<float>(c, a, b, ma, mb, mc); break;
        case 2: lc_tensor_contract_typed<int>(c, a, b, ma, mb, mc); break;
        default: lc_trap(); break;
    }
}
