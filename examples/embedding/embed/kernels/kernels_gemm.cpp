// bge-m3 embedding engine -- GEMM with fused epilogues (plan §4 kernel #3) and
// residual-add (plan §4 kernel #7) implementations. See kernels_gemm.h for the
// contract and launch geometry.
//
// ONNX-op mapping (plan §4 table, row #3): MatMul x192 (QKV projection,
// attention output projection, FFN up/down, retrieval heads) fused with the
// following bias `Add`, `Erf`-GELU / `Relu`, or residual `Add` chains; row #7
// covers the residual `Add` edges that are not fused into a GEMM epilogue.
//
// Two kernel families live here (optimization plan §3.1 G1..G4, G8):
//
// 1. Naive tiled family (`naive == true` in the table): a TM x TN output tile
//    per block with one output per thread (TM*TN threads), K streamed in
//    16-wide slabs through two shared tiles. The 16x16 entry is the
//    pre-optimization kernel: same index math, same accumulation order, so it
//    stays the parity anchor and the kill switch.
//
// 2. Register-blocked family (`naive == false`): each thread accumulates an
//    RM x RN sub-tile of fp32 outputs in registers while the block stages a
//    TM x TN output tile. A and B tiles live in shared memory (padded strides),
//    K streams through in TK-wide slabs with the fill -> sync_block() ->
//    consume -> sync_block() discipline (lc_optimize §4.6/§4.7). The inner K
//    loop is a plain C++ loop over a host-known trip count, i.e. unrolled at
//    emission time (lc_optimize §10.6: there is no XIR unroll pass).
//
// Why the register family should win (arithmetic-intensity argument, confirmed
// by the sweep in src/tests/benchmark/benchmark_embed_gemm.cpp): a thread reads
// TK*(RM+RN) shared floats (4 B each) and issues TK*RM*RN FMAs per slab, i.e.
// 2*(RM+RN)/(RM*RN) bytes of shared traffic per FLOP -- 1.0 B/FLOP for the 1x1
// (naive) and the 4x4 thread tiles, 0.75 for 8x4, 0.5 for 8x8. A warp retires
// ~128 B of shared per cycle against ~256 FLOP per cycle of fp32 ALU, so
// 0.5 B/FLOP is the first point where the ALU rather than the load/store unit
// becomes the limiter. Bigger tiles also multiply global reuse: a block with a
// TM x TN tile re-reads the weight matrix ceil(M/TM) times instead of M/16 as
// the naive 16x16 kernel does -- that is the amortization the O0 measurement
// said was missing (per-embedding latency flat from B=1 to B=16).
//
// Thread -> output mapping is deliberately *interleaved along N*: thread
// (grow, gcol) owns rows m0+grow*RM .. +RM-1 and the columns n0+gcol+q*COLS
// (q = 0..RN-1, COLS = TN/RN). Every B-tile read instruction then sees the warp
// touch consecutive-or-broadcast addresses (lanes with the same q differ only by
// gcol), and A-tile reads for a lane group differ by RM*(TK+1) floats, which
// the +1 row padding keeps in different banks.
//
// Ragged M/N/K: A/B tile loads are predicated and zero-filled and the store is
// guarded, so arbitrary sizes are correct -- not just multiples of the tile.
// Blocks whose tile and K slab are fully in range take a block-uniform
// *unguarded* fast path for the fill and the store (plan §3.1 G8 guard
// elision); the per-element predicates run only on edge blocks and the K tail
// slab. sync_block() never sits inside a divergent branch (the fast/slow fill
// branches close before the barrier) -- lc_optimize §10.7.
//
// Weight format (plan §3.1 G7, §4 Q1/Q2; kernels_gemm.h "WEIGHT FORMAT" for
// the buffer contract): the B-tile load is templated on GemmWeightFormat. For
// Fp32 it is `B.read(e)` exactly as before -- bit-identical. For Bf16/Fp16 it
// reads the packed word e/2 and up-converts one 16-bit half in the fill
// (bf16: exact bit extension; fp16: exact integer upcast including subnormals
// and inf/NaN), so the staged tile, the FMAs and the accumulation order are
// pure fp32 -- storage-only conversion, no half arithmetic anywhere. The
// conversion runs once per staged element and is register-blocked away like
// the load itself (G4: each staged value feeds RM or RN FMAs).
//
// Layout audit (plan §3.1 G2): B is the packed [K][N] weight (the packer
// pre-transposes every weight, plan §3.3), so the tile fill walks N contiguously
// -- a thread reads one B row's consecutive floats and consecutive threads of a
// warp cover consecutive columns, fully coalesced -- and the C store walks N
// contiguously as well. A is [M][K] and its fill walks K contiguously, also
// coalesced. K-blocking (the TK slab loop) is what makes the B tile stream
// through L2 in reuse order: per slab a block touches TM*TK + TK*TN floats of
// global data and re-uses each staged element from registers RM or RN times
// before it is overwritten.
#include "embedding/embed/kernels/kernels_gemm.h"

#include <charconv>
#include <cstdlib>
#include <cstring>

#include <luisa/core/logging.h>
#include <luisa/core/stl.h>
#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {

using compute::def;
using compute::Var;

// Build-time A/B knob for the guard-elision fast paths of plan §3.1 G8: define
// LUISA_EMBED_GEMM_NO_FASTPATH=1 (e.g. -DLUISA_EMBED_GEMM_NO_FASTPATH=1) to emit
// the register kernel with per-element predicates everywhere. The default (0)
// keeps the block-uniform unguarded fill/store paths. The guarded variant is the
// reference the fast variant is validated against -- it produces identical
// results (same math order), so the switch is a pure code-shape toggle. Measured
// on the dev GPU over the four engine shapes at rows in {512, 4096} on dx and vk
// (min over 3 repeats x min-of-12, benchmark_results/gemm_fastpath_ab_2026-10-17.csv):
// +0.2% to +2.6% runtime, i.e. the small constant win plan §3.1 G8 predicts --
// the register blocking (G3/G4) is what actually moves the needle (5-7x).
#ifndef LUISA_EMBED_GEMM_NO_FASTPATH
#define LUISA_EMBED_GEMM_NO_FASTPATH 0
#endif
constexpr bool kRegFastpath = LUISA_EMBED_GEMM_NO_FASTPATH == 0;
// The DSL builtins (set_block_size, sync_block, block_id, ite, abs, ...) are
// found by ADL on Var<>, but the bare calls below follow the engine convention
// of a using-directive in the kernel-building scope.
using namespace luisa::compute;// NOLINT(misc-header-include-cycle)

// ERF-GELU, exactly the bge-m3/ONNX form 0.5 * x * (1 + erf(x / sqrt(2))).
// The DSL exposes no native erf instruction (luisa/ast/op.h CallOp has no
// ERF member), so erf is evaluated here with the Abramowitz & Stegun
// 7.1.26 rational approximation -- maximum absolute error 1.5e-7, four
// orders of magnitude below the parity-gate tolerance -- composed only from
// DSL abs/exp/ite so dx, vk, cuda and fallback compute the same value.
// The approximation is evaluated on |x / sqrt(2)| and restored to the odd
// sign of erf.
[[nodiscard]] Var<float> dsl_gelu_erf(Var<float> x) noexcept {
    const Var u = x * 0.70710678118654752440f;// x / sqrt(2)
    const Var z = abs(u);
    const Var one = def(1.0f);
    const Var t = one / (one + 0.3275911f * z);
    // Horner form of a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5.
    const Var poly = t * (0.254829592f +
                          t * (-0.284496736f +
                               t * (1.421413741f +
                                    t * (-1.453152027f + t * 1.061405429f))));
    const Var erf_u = one - poly * compute::exp(-(z * z));
    return 0.5f * x * (one + compute::ite(u >= 0.0f, erf_u, -erf_u));
}

// True for the epilogues whose `aux` operand is a bias[N] vector.
constexpr bool gemm_uses_bias(GemmEpilogue ep) noexcept {
    return ep == GemmEpilogue::Bias || ep == GemmEpilogue::BiasGeluErf || ep == GemmEpilogue::BiasRelu;
}

// Exact fp16 -> fp32 upcast on DSL integers (every half value -- normals,
// subnormals, +-0, inf, NaN -- is representable in fp32). Magnitude-first
// form: the sign is applied once at the end by a float negation select.
//   mag = h & 0x7FFF
//   mag >= 0x7C00 : inf/NaN -> bits = 0x7F800000 | (mag & 0x3FF) << 13
//   mag >= 0x0400 : normal  -> bits = (mag << 13) + (112 << 23)
//                   (rebias: e16 + 112 = e32, no carry into the exponent byte
//                   because e16 <= 30 here -> e32 <= 142 < 255)
//   else          : subnormal/zero -> value = mag * 2^-24 (exact in fp32)
[[nodiscard]] Var<float> dsl_fp16_upcast(const Var<uint> &h) noexcept {
    const Var mag = h & 0x7fffu;
    const Var infnan = compute::as<float>(0x7f800000u | ((mag & 0x03ffu) << 13u));
    const Var normal = compute::as<float>((mag << 13u) + 0x38000000u);
    const Var sub = (mag & 0x03ffu).cast<float>() * 5.9604644775390625e-8f;// 2^-24
    const Var pos = compute::ite(mag >= 0x7c00u, infnan,
                                 compute::ite(mag >= 0x0400u, normal, sub));
    return compute::ite((h >> 15u) != 0u, -pos, pos);
}

// Load one logical B element e (= k*N + n of the packed [K][N] weight). The
// Fp32 branch is the original `B.read(e)`; the half branches read the packed
// word e/2 and up-convert the e%2 half (see kernels_gemm.h for the word
// layout: element 2j in bits [15:0], element 2j+1 in bits [31:16], little-
// endian -- the raw v2 blob uploaded without repacking).
template<GemmWeightFormat WF>
[[nodiscard]] Var<float> load_b_element(const compute::BufferFloat &B, const Var<uint> &e) noexcept {
    if constexpr (WF == GemmWeightFormat::Fp32) {
        return B.read(e);
    } else {
        const Var w = compute::as<uint>(B.read(e >> 1u));
        if constexpr (WF == GemmWeightFormat::Bf16) {
            // bf16 -> fp32 is the exact bit extension (append 16 zero bits).
            const Var bits = compute::ite((e & 1u) == 0u, (w << 16u), (w & 0xffff0000u));
            return compute::as<float>(bits);
        } else {
            static_assert(WF == GemmWeightFormat::Fp16, "unhandled GemmWeightFormat");
            const Var h16 = compute::ite((e & 1u) == 0u, (w & 0xffffu), (w >> 16u));
            return dsl_fp16_upcast(h16);
        }
    }
}

// Cooperative tile-fill iteration: element e = tid + rep*BLOCK over a
// host-known `ELEMS` count. The element-count guard is emitted only when the
// count is not an exact multiple of the block (lc_optimize §10.3: prove the
// divisibility at generation time and omit the tail predicate).
template<uint32_t BLOCK>
struct FillEach {
    template<uint32_t ELEMS, typename Body>
    static void run(const Var<uint> &tid, const Body &body) noexcept {
        for (auto rep = 0u; rep < (ELEMS + BLOCK - 1u) / BLOCK; rep++) {
            const Var e = tid + rep * BLOCK;
            if constexpr (ELEMS % BLOCK != 0u) {
                $if (e < ELEMS) {
                    body(e);
                };
            } else {
                body(e);
            }
        }
    }
};

}// namespace

const char *gemm_epilogue_name(GemmEpilogue ep) noexcept {
    switch (ep) {
        case GemmEpilogue::None: return "embed_gemm_none";
        case GemmEpilogue::Bias: return "embed_gemm_bias";
        case GemmEpilogue::BiasGeluErf: return "embed_gemm_bias_gelu_erf";
        case GemmEpilogue::BiasRelu: return "embed_gemm_bias_relu";
        case GemmEpilogue::ResidualAdd: return "embed_gemm_residual_add";
    }
    return "embed_gemm_unknown";
}

const char *gemm_weight_format_name(GemmWeightFormat wf) noexcept {
    switch (wf) {
        case GemmWeightFormat::Fp32: return "fp32";
        case GemmWeightFormat::Bf16: return "bf16";
        case GemmWeightFormat::Fp16: return "fp16";
    }
    return "unknown";
}

namespace {

// Compile-time epilogue selection (`if constexpr` on the flag template
// argument): the unselected epilogue math never reaches the recorded DSL.
// `aux` carries bias[N] for the bias flags and the residual R[M,N] for
// ResidualAdd; the None flag never touches it (callers bind a dummy view).
// This form reads the epilogue operand itself -- used by the naive family, one
// output per thread, exactly as before the tuning work.
template<GemmEpilogue EP>
[[nodiscard]] Var<float> gemm_epilogue(const Var<float> &acc, const compute::BufferFloat &aux,
                                       const Var<uint> &m, const Var<uint> &n,
                                       const Var<uint> &N) noexcept {
    if constexpr (EP == GemmEpilogue::None) {
        static_cast<void>(aux);
        static_cast<void>(m);
        static_cast<void>(n);
        static_cast<void>(N);
        return acc;
    } else if constexpr (EP == GemmEpilogue::Bias) {
        static_cast<void>(m);
        return acc + aux.read(n);
    } else if constexpr (EP == GemmEpilogue::BiasGeluErf) {
        static_cast<void>(m);
        return dsl_gelu_erf(acc + aux.read(n));
    } else if constexpr (EP == GemmEpilogue::BiasRelu) {
        static_cast<void>(m);
        return compute::max(acc + aux.read(n), def(0.0f));
    } else {
        static_assert(EP == GemmEpilogue::ResidualAdd, "unhandled GemmEpilogue flag");
        return acc + aux.read(m * N + n);
    }
}

// Same epilogues for the register family, with the bias value pre-read once per
// (thread, column) instead of once per output element -- mathematically
// identical (bias values are the same loads, hoisted out of the row loop).
template<GemmEpilogue EP>
[[nodiscard]] Var<float> reg_epilogue(const Var<float> &acc, const compute::BufferFloat &aux,
                                      const Var<uint> &m, const Var<uint> &n,
                                      const Var<uint> &N, const Var<float> &bias) noexcept {
    if constexpr (EP == GemmEpilogue::None) {
        static_cast<void>(aux);
        static_cast<void>(m);
        static_cast<void>(n);
        static_cast<void>(N);
        static_cast<void>(bias);
        return acc;
    } else if constexpr (EP == GemmEpilogue::Bias) {
        static_cast<void>(m);
        return acc + bias;
    } else if constexpr (EP == GemmEpilogue::BiasGeluErf) {
        static_cast<void>(m);
        return dsl_gelu_erf(acc + bias);
    } else if constexpr (EP == GemmEpilogue::BiasRelu) {
        static_cast<void>(m);
        return compute::max(acc + bias, def(0.0f));
    } else {
        static_assert(EP == GemmEpilogue::ResidualAdd, "unhandled GemmEpilogue flag");
        static_cast<void>(bias);
        return acc + aux.read(m * N + n);
    }
}

//------------------------------------------------------------------------------------------------------------------
// Family 1: naive tiled kernel (legacy shape; one output per thread).
//------------------------------------------------------------------------------------------------------------------

template<uint32_t TM, uint32_t TN>
struct NaiveTile {
    static constexpr uint32_t kslab = 16u;// legacy K slab width (== kGemmTile)
    static constexpr uint32_t block = TM * TN;
    static constexpr uint32_t a_elems = TM * kslab;// A tile: TM rows x kslab K columns
    static constexpr uint32_t b_elems = kslab * TN;// B tile: kslab K rows x TN columns
    static_assert(block >= 32u && block <= 1024u && block % 32u == 0u, "naive gemm block size");
    static_assert((a_elems + b_elems) * 4u <= 32768u, "naive gemm tile exceeds the 32 KiB DX shared floor");
};

template<uint32_t TM, uint32_t TN, GemmWeightFormat WF, GemmEpilogue EP>
[[nodiscard]] GemmKernel make_naive_kernel(const char *vname) noexcept {
    using T = NaiveTile<TM, TN>;

    GemmKernel kernel = [vname](compute::BufferFloat A, compute::BufferFloat B,
                                compute::BufferFloat C, compute::BufferFloat aux,
                                Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
        set_block_size(T::block, 1u, 1u);
        set_name(vname);

        compute::Shared<float> tile_a{T::a_elems};
        compute::Shared<float> tile_b{T::b_elems};

        const Var tid = thread_x();
        const Var tr = tid / TN;// row inside the output tile
        const Var tc = tid % TN;// column inside the tile (along N)
        const Var m = block_id().y * TM + tr;
        const Var n = block_id().x * TN + tc;

        // Cooperative tile fill: tile element e is A row e/kslab, K column
        // e%kslab (resp. B K row e/TN, column e%TN). For the legacy 16x16 shape
        // e == tid and the element-count guard is compile-time elided, which
        // reproduces the original kernel's index math and emission order exactly.
        using Each = FillEach<T::block>;

        Var acc = def(0.0f);
        $for (kt, (K + T::kslab - 1u) / T::kslab) {
            const Var k0 = kt * T::kslab;
            // A tile row m0 + r, column k0 + c: contiguous along K in
            // row-major A.
            Each::template run<T::a_elems>(tid, [&](const Var<uint> &e) noexcept {
                const Var r = e / T::kslab;
                const Var c = e % T::kslab;
                const Var mrow = block_id().y * TM + r;
                compute::Float a_val;
                $if (mrow < M & (k0 + c) < K) {
                    a_val = A.read(mrow * K + k0 + c);
                }
                $else {
                    a_val = def(0.0f);
                };
                tile_a[e] = a_val;
            });
            // B tile row k0 + r (along K), column n0 + c: contiguous along N
            // (coalesced).
            Each::template run<T::b_elems>(tid, [&](const Var<uint> &e) noexcept {
                const Var r = e / TN;
                const Var c = e % TN;
                const Var ncol = block_id().x * TN + c;
                compute::Float b_val;
                $if ((k0 + r) < K & ncol < N) {
                    b_val = load_b_element<WF>(B, (k0 + r) * N + ncol);
                }
                $else {
                    b_val = def(0.0f);
                };
                tile_b[e] = b_val;
            });
            sync_block();
            // Fixed compile-time unroll over the K sub-tile (plain C++ loop).
            for (auto kk = 0u; kk < T::kslab; kk++) {
                acc += tile_a[tr * T::kslab + kk] * tile_b[kk * TN + tc];
            }
            sync_block();
        };

        $if (m < M & n < N) {
            C.write(m * N + n, gemm_epilogue<EP>(acc, aux, m, n, N));
        };
    };
    return kernel;
}

template<uint32_t TM, uint32_t TN, GemmWeightFormat WF>
[[nodiscard]] GemmKernel build_naive_ep(GemmEpilogue ep, const char *vname) noexcept {
    switch (ep) {
        case GemmEpilogue::None: return make_naive_kernel<TM, TN, WF, GemmEpilogue::None>(vname);
        case GemmEpilogue::Bias: return make_naive_kernel<TM, TN, WF, GemmEpilogue::Bias>(vname);
        case GemmEpilogue::BiasGeluErf: return make_naive_kernel<TM, TN, WF, GemmEpilogue::BiasGeluErf>(vname);
        case GemmEpilogue::BiasRelu: return make_naive_kernel<TM, TN, WF, GemmEpilogue::BiasRelu>(vname);
        case GemmEpilogue::ResidualAdd: return make_naive_kernel<TM, TN, WF, GemmEpilogue::ResidualAdd>(vname);
    }
    return make_naive_kernel<TM, TN, WF, GemmEpilogue::Bias>(vname);
}

template<uint32_t TM, uint32_t TN>
[[nodiscard]] GemmKernel build_naive(GemmEpilogue ep, GemmWeightFormat wf, const char *vname) noexcept {
    switch (wf) {
        case GemmWeightFormat::Fp32: return build_naive_ep<TM, TN, GemmWeightFormat::Fp32>(ep, vname);
        case GemmWeightFormat::Bf16: return build_naive_ep<TM, TN, GemmWeightFormat::Bf16>(ep, vname);
        case GemmWeightFormat::Fp16: return build_naive_ep<TM, TN, GemmWeightFormat::Fp16>(ep, vname);
    }
    return build_naive_ep<TM, TN, GemmWeightFormat::Fp32>(ep, vname);
}

//------------------------------------------------------------------------------------------------------------------
// Family 2: register-blocked tiled kernel.
//------------------------------------------------------------------------------------------------------------------

// Geometry constants of one register-blocked variant.
template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN>
struct RegTile {
    static constexpr uint32_t cols = TN / RN;// thread tiles along N (column interleave)
    static constexpr uint32_t rows = TM / RM;// thread tiles along M
    static constexpr uint32_t block = rows * cols;
    static constexpr uint32_t a_elems = TM * TK;
    static constexpr uint32_t b_elems = TK * TN;
    static constexpr uint32_t astride = TK + 1u;// +1: A tile rows land in different banks
    static constexpr uint32_t bstride = TN + 1u;
    static constexpr uint32_t a_sh = TM * astride;
    static constexpr uint32_t b_sh = TK * bstride;
    static_assert(TM % RM == 0u && TN % RN == 0u, "tile must divide into register sub-tiles");
    static_assert(block >= 32u && block <= 1024u && block % 32u == 0u, "register gemm block size");
    static_assert((a_sh + b_sh) * 4u <= 32768u, "register gemm tile exceeds the 32 KiB DX shared floor");
};

// Shared, epilogue-independent part: the K slab loop (fill -> barrier ->
// consume -> barrier) accumulating into the per-thread register sub-tile.
// Instantiated once per geometry (and per B-operand format WF), so an extra
// epilogue only costs its own store code.
template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN, GemmWeightFormat WF>
void reg_accumulate(const compute::BufferFloat &A, const compute::BufferFloat &B,
                    const Var<uint> &M, const Var<uint> &N, const Var<uint> &K,
                    const Var<uint> &m0, const Var<uint> &n0, const Var<uint> &tid,
                    Var<float> (&acc)[RM][RN]) noexcept {
    using T = RegTile<TM, TN, TK, RM, RN>;
    using Each = FillEach<T::block>;

    compute::Shared<float> tile_a{T::a_sh};
    compute::Shared<float> tile_b{T::b_sh};

    const Var grow = tid / T::cols;
    const Var gcol = tid % T::cols;
    // Slabs whose K range lies fully inside the matrix (K % TK != 0 -> the last
    // slab takes the guarded path; TK % 4 == 0 is not assumed).
    const Var kfull = K / TK;

    // Block-uniform fast/slow fill (plan §3.1 G8): the fast path is entered only
    // when this block's rows, columns and this slab's K range are all in range,
    // so the per-element predicates disappear for interior blocks. Both paths
    // write every element of both tiles, so no uninitialised shared value can
    // reach the dot product.
    const auto fill_fast = [&](const Var<uint> &k0) noexcept {
        Each::template run<T::a_elems>(tid, [&](const Var<uint> &e) noexcept {
            const Var r = e / TK;
            const Var c = e % TK;
            tile_a[r * T::astride + c] = A.read((m0 + r) * K + k0 + c);
        });
        Each::template run<T::b_elems>(tid, [&](const Var<uint> &e) noexcept {
            const Var r = e / TN;// K row inside the slab
            const Var c = e % TN;// column along N (coalesced)
            tile_b[r * T::bstride + c] = load_b_element<WF>(B, (k0 + r) * N + n0 + c);
        });
    };
    const auto fill_guarded = [&](const Var<uint> &k0) noexcept {
        Each::template run<T::a_elems>(tid, [&](const Var<uint> &e) noexcept {
            const Var r = e / TK;
            const Var c = e % TK;
            compute::Float a_val;
            $if ((m0 + r) < M & (k0 + c) < K) {
                a_val = A.read((m0 + r) * K + k0 + c);
            }
            $else {
                a_val = def(0.0f);
            };
            tile_a[r * T::astride + c] = a_val;
        });
        Each::template run<T::b_elems>(tid, [&](const Var<uint> &e) noexcept {
            const Var r = e / TN;
            const Var c = e % TN;
            compute::Float b_val;
            $if ((k0 + r) < K & (n0 + c) < N) {
                b_val = load_b_element<WF>(B, (k0 + r) * N + n0 + c);
            }
            $else {
                b_val = def(0.0f);
            };
            tile_b[r * T::bstride + c] = b_val;
        });
    };

    $for (s, (K + TK - 1u) / TK) {
        const Var k0 = s * TK;
        if constexpr (kRegFastpath) {
            $if ((s < kfull) & (m0 + TM <= M) & (n0 + TN <= N)) {
                fill_fast(k0);
            }
            $else {
                fill_guarded(k0);
            };
        } else {
            fill_guarded(k0);
        }
        sync_block();
        // Compile-time unrolled consumption of the slab: plain C++ loops over
        // host-known trip counts (no XIR unroll pass exists -- lc_optimize
        // §10.6). The accumulator locals are created once, outside the kk loop
        // (storage is fixed at emission time -- lc_optimize §10.5).
        Var<float> bfrag[RN];
        Var<float> afrag;
        for (auto kk = 0u; kk < TK; kk++) {
            for (auto q = 0u; q < RN; q++) {
                bfrag[q] = tile_b[kk * T::bstride + q * T::cols + gcol];
            }
            for (auto r = 0u; r < RM; r++) {
                afrag = tile_a[(grow * RM + r) * T::astride + kk];
                for (auto q = 0u; q < RN; q++) {
                    acc[r][q] += afrag * bfrag[q];
                }
            }
        }
        sync_block();
    };
}

// Epilogue-specific store. `nb` is the thread's first column; its remaining
// columns are nb + q*COLS (the interleaved ownership of the accumulate phase).
template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN, GemmEpilogue EP>
void reg_store(const Var<float> (&acc)[RM][RN], const compute::BufferFloat &C,
               const compute::BufferFloat &aux, const Var<uint> &M, const Var<uint> &N,
               const Var<uint> &m0, const Var<uint> &n0, const Var<uint> &tid) noexcept {
    using T = RegTile<TM, TN, TK, RM, RN>;
    const Var grow = tid / T::cols;
    const Var gcol = tid % T::cols;
    const Var mb = m0 + grow * RM;
    const Var nb = n0 + gcol;

    // Bias is a per-column constant: read it once per thread column instead of
    // once per output element (RM reuse per read). Out-of-range columns read
    // zero; those lanes are never stored to.
    Var<float> bcol[RN];
    if constexpr (gemm_uses_bias(EP)) {
        for (auto q = 0u; q < RN; q++) {
            const Var nn = nb + q * T::cols;
            compute::Float b;
            $if (nn < N) {
                b = aux.read(nn);
            }
            $else {
                b = def(0.0f);
            };
            bcol[q] = b;
        }
    }

    const auto emit = [&](const auto guard_tag) noexcept {
        constexpr bool guarded = decltype(guard_tag)::value;
        for (auto r = 0u; r < RM; r++) {
            const Var mm = mb + r;
            for (auto q = 0u; q < RN; q++) {
                const Var nn = nb + q * T::cols;
                if constexpr (guarded) {
                    $if (mm < M & nn < N) {
                        C.write(mm * N + nn, reg_epilogue<EP>(acc[r][q], aux, mm, nn, N, bcol[q]));
                    };
                } else {
                    C.write(mm * N + nn, reg_epilogue<EP>(acc[r][q], aux, mm, nn, N, bcol[q]));
                }
            }
        }
    };
    // Uniform fast path: this thread's whole sub-tile is in range -> no
    // per-element store guards (plan §3.1 G8).
    if constexpr (kRegFastpath) {
        $if ((mb + RM) <= M & (nb + (RN - 1u) * T::cols) < N) {
            emit(std::false_type{});// whole sub-tile in range: no store guards
        }
        $else {
            emit(std::true_type{});// edge sub-tile: per-element guards
        };
    } else {
        emit(std::true_type{});
    }
}

template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN, bool SWAP, GemmWeightFormat WF, GemmEpilogue EP>
[[nodiscard]] GemmKernel make_reg_kernel(const char *vname) noexcept {
    using T = RegTile<TM, TN, TK, RM, RN>;
    GemmKernel kernel = [vname](compute::BufferFloat A, compute::BufferFloat B,
                                compute::BufferFloat C, compute::BufferFloat aux,
                                Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
        set_block_size(T::block, 1u, 1u);
        set_name(vname);

        const Var tid = thread_x();
        // Default grid: x carries the N-tiles, y the M-tiles (see the header).
        // swap_grid probes the opposite L2 grouping (plan §3.1 G2).
        Var<uint> mt;
        Var<uint> nt;
        if constexpr (SWAP) {
            mt = block_id().x;
            nt = block_id().y;
        } else {
            mt = block_id().y;
            nt = block_id().x;
        }
        const Var m0 = mt * TM;
        const Var n0 = nt * TN;

        Var<float> acc[RM][RN];// DSL locals are zero-initialized
        reg_accumulate<TM, TN, TK, RM, RN, WF>(A, B, M, N, K, m0, n0, tid, acc);
        reg_store<TM, TN, TK, RM, RN, EP>(acc, C, aux, M, N, m0, n0, tid);
    };
    return kernel;
}

template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN, bool SWAP, GemmWeightFormat WF>
[[nodiscard]] GemmKernel build_reg_ep(GemmEpilogue ep, const char *vname) noexcept {
    switch (ep) {
        case GemmEpilogue::None: return make_reg_kernel<TM, TN, TK, RM, RN, SWAP, WF, GemmEpilogue::None>(vname);
        case GemmEpilogue::Bias: return make_reg_kernel<TM, TN, TK, RM, RN, SWAP, WF, GemmEpilogue::Bias>(vname);
        case GemmEpilogue::BiasGeluErf: return make_reg_kernel<TM, TN, TK, RM, RN, SWAP, WF, GemmEpilogue::BiasGeluErf>(vname);
        case GemmEpilogue::BiasRelu: return make_reg_kernel<TM, TN, TK, RM, RN, SWAP, WF, GemmEpilogue::BiasRelu>(vname);
        case GemmEpilogue::ResidualAdd: return make_reg_kernel<TM, TN, TK, RM, RN, SWAP, WF, GemmEpilogue::ResidualAdd>(vname);
    }
    return make_reg_kernel<TM, TN, TK, RM, RN, SWAP, WF, GemmEpilogue::Bias>(vname);
}

template<uint32_t TM, uint32_t TN, uint32_t TK, uint32_t RM, uint32_t RN, bool SWAP>
[[nodiscard]] GemmKernel build_reg(GemmEpilogue ep, GemmWeightFormat wf, const char *vname) noexcept {
    switch (wf) {
        case GemmWeightFormat::Fp32: return build_reg_ep<TM, TN, TK, RM, RN, SWAP, GemmWeightFormat::Fp32>(ep, vname);
        case GemmWeightFormat::Bf16: return build_reg_ep<TM, TN, TK, RM, RN, SWAP, GemmWeightFormat::Bf16>(ep, vname);
        case GemmWeightFormat::Fp16: return build_reg_ep<TM, TN, TK, RM, RN, SWAP, GemmWeightFormat::Fp16>(ep, vname);
    }
    return build_reg_ep<TM, TN, TK, RM, RN, SWAP, GemmWeightFormat::Fp32>(ep, vname);
}

//------------------------------------------------------------------------------------------------------------------
// Variant table and runtime config resolution.
//------------------------------------------------------------------------------------------------------------------

using KernelBuilder = GemmKernel (*)(GemmEpilogue, GemmWeightFormat, const char *) noexcept;

struct Entry {
    GemmTileConfig cfg;
    KernelBuilder builder;
};

// (name, tile_m, tile_n, tile_k, reg_m, reg_n, block, naive, swap_grid).
// `block` is always (tile_m/reg_m)*(tile_n/reg_n) -- asserted below.
// Shared budget per variant is printed in the sweep CSV; all entries stay under
// the 32 KiB DX floor (largest: rb_128x128_b256_r88 at 16.6 KiB).
constexpr Entry kEntries[] = {
    //--- legacy naive family (kill switch lives here) --------------------
    // 16x16/256 is today's shipped kernel, emitted with identical index math.
    {{"naive_16x16", 16u, 16u, 16u, 1u, 1u, 256u, true, false}, &build_naive<16u, 16u>},
    {{"naive_8x8", 8u, 8u, 16u, 1u, 1u, 64u, true, false}, &build_naive<8u, 8u>},
    {{"naive_16x8", 16u, 8u, 16u, 1u, 1u, 128u, true, false}, &build_naive<16u, 8u>},
    {{"naive_32x8", 32u, 8u, 16u, 1u, 1u, 256u, true, false}, &build_naive<32u, 8u>},
    {{"naive_32x16", 32u, 16u, 16u, 1u, 1u, 512u, true, false}, &build_naive<32u, 16u>},
    //--- register-blocked family, 4x4 thread tile -----------------------
    {{"rb_32x32_b64_r44", 32u, 32u, 16u, 4u, 4u, 64u, false, false}, &build_reg<32u, 32u, 16u, 4u, 4u, false>},
    {{"rb_32x64_b128_r44", 32u, 64u, 16u, 4u, 4u, 128u, false, false}, &build_reg<32u, 64u, 16u, 4u, 4u, false>},
    {{"rb_64x64_b256_r44", 64u, 64u, 16u, 4u, 4u, 256u, false, false}, &build_reg<64u, 64u, 16u, 4u, 4u, false>},
    {{"rb_64x128_b512_r44", 64u, 128u, 16u, 4u, 4u, 512u, false, false}, &build_reg<64u, 128u, 16u, 4u, 4u, false>},
    {{"rb_64x64_b256_r44_k32", 64u, 64u, 32u, 4u, 4u, 256u, false, false}, &build_reg<64u, 64u, 32u, 4u, 4u, false>},
    //--- 8x4 thread tile -------------------------------------------------
    {{"rb_64x64_b128_r84", 64u, 64u, 16u, 8u, 4u, 128u, false, false}, &build_reg<64u, 64u, 16u, 8u, 4u, false>},
    {{"rb_128x64_b256_r84", 128u, 64u, 16u, 8u, 4u, 256u, false, false}, &build_reg<128u, 64u, 16u, 8u, 4u, false>},
    {{"rb_128x128_b512_r84", 128u, 128u, 16u, 8u, 4u, 512u, false, false}, &build_reg<128u, 128u, 16u, 8u, 4u, false>},
    //--- 8x8 thread tile (0.5 B/FLOP shared traffic) --------------------
    {{"rb_64x64_b64_r88", 64u, 64u, 16u, 8u, 8u, 64u, false, false}, &build_reg<64u, 64u, 16u, 8u, 8u, false>},
    {{"rb_128x64_b128_r88", 128u, 64u, 16u, 8u, 8u, 128u, false, false}, &build_reg<128u, 64u, 16u, 8u, 8u, false>},
    {{"rb_128x128_b256_r88", 128u, 128u, 16u, 8u, 8u, 256u, false, false}, &build_reg<128u, 128u, 16u, 8u, 8u, false>},
    {{"rb_256x64_b256_r88", 256u, 64u, 16u, 8u, 8u, 256u, false, false}, &build_reg<256u, 64u, 16u, 8u, 8u, false>},
    //--- L2 grouping probe (plan §3.1 G2), same geometry as rb_64x64_b256_r44
    {{"rb_64x64_b256_r44_swap", 64u, 64u, 16u, 4u, 4u, 256u, false, true}, &build_reg<64u, 64u, 16u, 4u, 4u, true>},
};

constexpr uint32_t kEntryCount = static_cast<uint32_t>(sizeof(kEntries) / sizeof(kEntries[0]));

// Shipped default variant, resolved by name (robust against table edits).
// Measured on the dev GPU (RTX 5060-class) in the debug sweep
// benchmark_results/gemm_sweep_2026-10-17.csv (18 variants x 4 engine shapes x
// rows {512, 4096} x {dx, vk}, min-of-12 after 4 warm-ups, min over repeats):
// it is the RANK-1 variant on both backends -- geomean over the 8 shape/rows
// combos dx 6.35x / vk 4.38x vs the pre-optimization naive_16x16 (runner-ups
// rb_128x64_b128_r88 6.24x / 4.36x, rb_64x64_b64_r88 6.19x / 4.33x).
// Absolute before/after on the acceptance shapes at rows=4096 (B=8, S=512;
// benchmark_results/gemm_before_after_2026-10-17.csv): qkv 8.34 -> 1.16 ms (dx)
// and 5.73 -> 1.16 ms (vk); ffn_up 11.07 -> 1.52 / 7.63 -> 1.51;
// ffn_down 11.08 -> 1.53 / 7.65 -> 1.53. The parent should re-measure in release
// before freezing the table (plan §7.2); a per-backend change is one env var
// away at deployment (see the header).
constexpr luisa::string_view kDefaultName{"rb_64x64_b128_r84"};

[[nodiscard]] constexpr uint32_t default_entry_index() noexcept {
    for (auto i = 0u; i < kEntryCount; i++) {
        if (luisa::string_view{kEntries[i].cfg.name} == kDefaultName) {
            return i;
        }
    }
    return 0u;
}

constexpr uint32_t kDefaultEntry = default_entry_index();

static_assert(kDefaultEntry < kEntryCount &&
                  luisa::string_view{kEntries[kDefaultEntry].cfg.name} == kDefaultName,
              "the default gemm variant must exist in the table");

// `block` must be derived geometry, never a free knob.
[[nodiscard]] constexpr bool gemm_table_is_consistent() noexcept {
    for (const auto &e : kEntries) {
        const auto cfg = e.cfg;
        if (cfg.block != (cfg.tile_m / cfg.reg_m) * (cfg.tile_n / cfg.reg_n)) {
            return false;
        }
        if (cfg.naive != (cfg.reg_m == 1u && cfg.reg_n == 1u)) {
            return false;
        }
        if (cfg.naive && cfg.tile_k != 16u) {
            return false;
        }
        if (e.builder == nullptr) {
            return false;
        }
    }
    return true;
}
static_assert(gemm_table_is_consistent(), "gemm variant table is internally inconsistent");

[[nodiscard]] int32_t entry_by_name(const char *name) noexcept {
    if (name == nullptr) {
        return -1;
    }
    for (auto i = 0u; i < kEntryCount; i++) {
        if (luisa::string_view{kEntries[i].cfg.name} == luisa::string_view{name}) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

[[nodiscard]] bool parse_u32_text(const char *first, const char *last, uint32_t &v) noexcept {
    const auto r = std::from_chars(first, last, v);
    return r.ec == std::errc{} && r.ptr == last;
}

// "<u>x<u>" (also 'X') -> two unsigneds.
[[nodiscard]] bool parse_wxh(const char *text, uint32_t &a, uint32_t &b) noexcept {
    if (text == nullptr) {
        return false;
    }
    const auto *sep = std::strchr(text, 'x');
    if (sep == nullptr) {
        sep = std::strchr(text, 'X');
    }
    if (sep == nullptr || sep == text) {
        return false;
    }
    uint32_t va = 0u, vb = 0u;
    if (!parse_u32_text(text, sep, va)) {
        return false;
    }
    const auto *tail = sep + 1u;
    if (!parse_u32_text(tail, tail + std::strlen(tail), vb)) {
        return false;
    }
    a = va;
    b = vb;
    return true;
}

[[nodiscard]] bool parse_u32(const char *text, uint32_t &v) noexcept {
    if (text == nullptr) {
        return false;
    }
    return parse_u32_text(text, text + std::strlen(text), v);
}

struct Selection {
    uint32_t index;
    luisa::string source;
};

[[nodiscard]] const Selection &selection() noexcept {
    static const Selection sel = [] noexcept {
        // 1. Whole-variant override.
        if (const auto *name = std::getenv("LUISA_EMBED_GEMM_CONFIG"); name != nullptr && name[0] != '\0') {
            if (const auto i = entry_by_name(name); i >= 0) {
                const auto &cfg = kEntries[static_cast<uint32_t>(i)].cfg;
                LUISA_INFO("embed gemm: variant '{}' selected by LUISA_EMBED_GEMM_CONFIG "
                           "(tile {}x{}, kslab {}, reg {}x{}, block {}).",
                           cfg.name, cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.reg_m, cfg.reg_n, cfg.block);
                return Selection{static_cast<uint32_t>(i), "LUISA_EMBED_GEMM_CONFIG"};
            }
            LUISA_WARNING("embed gemm: LUISA_EMBED_GEMM_CONFIG=\"{}\" is not a known variant; "
                          "falling back to the shipped default.",
                          name);
        }
        // 2. Field-wise override (every specified field must match).
        uint32_t tm = 0u, tn = 0u, tk = 0u, rm = 0u, rn = 0u, blk = 0u;
        uint32_t wh[2]{};
        bool any = false;
        if (const auto *t = std::getenv("LUISA_EMBED_GEMM_TILE"); t != nullptr && t[0] != '\0') {
            if (parse_wxh(t, wh[0], wh[1])) {
                tm = wh[0];
                tn = wh[1];
                any = true;
            } else {
                LUISA_WARNING("embed gemm: ignoring malformed LUISA_EMBED_GEMM_TILE=\"{}\".", t);
            }
        }
        if (const auto *b = std::getenv("LUISA_EMBED_GEMM_BLOCK"); b != nullptr && b[0] != '\0') {
            if (parse_u32(b, blk)) {
                any = true;
            } else {
                LUISA_WARNING("embed gemm: ignoring malformed LUISA_EMBED_GEMM_BLOCK=\"{}\".", b);
            }
        }
        if (const auto *k = std::getenv("LUISA_EMBED_GEMM_KSLAB"); k != nullptr && k[0] != '\0') {
            if (parse_u32(k, tk)) {
                any = true;
            } else {
                LUISA_WARNING("embed gemm: ignoring malformed LUISA_EMBED_GEMM_KSLAB=\"{}\".", k);
            }
        }
        if (const auto *r = std::getenv("LUISA_EMBED_GEMM_REGTILE"); r != nullptr && r[0] != '\0') {
            if (parse_wxh(r, wh[0], wh[1])) {
                rm = wh[0];
                rn = wh[1];
                any = true;
            } else {
                LUISA_WARNING("embed gemm: ignoring malformed LUISA_EMBED_GEMM_REGTILE=\"{}\".", r);
            }
        }
        if (any) {
            int32_t found = -1;
            uint32_t matches = 0u;
            for (auto i = 0u; i < kEntryCount; i++) {
                const auto &cfg = kEntries[i].cfg;
                if ((tm != 0u && cfg.tile_m != tm) || (tn != 0u && cfg.tile_n != tn) ||
                    (blk != 0u && cfg.block != blk) || (tk != 0u && cfg.tile_k != tk) ||
                    (rm != 0u && cfg.reg_m != rm) || (rn != 0u && cfg.reg_n != rn)) {
                    continue;
                }
                found = static_cast<int32_t>(i);
                matches++;
            }
            if (matches == 1u) {
                const auto &cfg = kEntries[static_cast<uint32_t>(found)].cfg;
                LUISA_INFO("embed gemm: variant '{}' selected by LUISA_EMBED_GEMM_TILE/BLOCK "
                           "(tile {}x{}, kslab {}, reg {}x{}, block {}).",
                           cfg.name, cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.reg_m, cfg.reg_n, cfg.block);
                return Selection{static_cast<uint32_t>(found), "LUISA_EMBED_GEMM_TILE/BLOCK"};
            }
            LUISA_WARNING("embed gemm: the LUISA_EMBED_GEMM_TILE/BLOCK/KSLAB/REGTILE override "
                          "matched {} variants; falling back to the shipped default '{}' "
                          "(use LUISA_EMBED_GEMM_CONFIG for an exact pick).",
                          matches, kEntries[kDefaultEntry].cfg.name);
        }
        // 3. Shipped default.
        const auto &cfg = kEntries[kDefaultEntry].cfg;
        LUISA_INFO("embed gemm: default variant '{}' (tile {}x{}, kslab {}, reg {}x{}, block {}).",
                   cfg.name, cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.reg_m, cfg.reg_n, cfg.block);
        return Selection{kDefaultEntry, "default"};
    }();
    return sel;
}

}// namespace

uint32_t gemm_tile_variant_count() noexcept {
    return kEntryCount;
}

const GemmTileConfig *gemm_tile_variant(uint32_t i) noexcept {
    return i < kEntryCount ? &kEntries[i].cfg : nullptr;
}

int32_t gemm_tile_index_by_name(const char *name) noexcept {
    return entry_by_name(name);
}

const GemmTileConfig &gemm_resolved_config() noexcept {
    return kEntries[selection().index].cfg;
}

luisa::string gemm_resolved_config_source() noexcept {
    const auto &sel = selection();
    return luisa::format("{} ({})", kEntries[sel.index].cfg.name, sel.source);
}

GemmKernel make_gemm_kernel_config(GemmEpilogue ep, const GemmTileConfig &cfg) noexcept {
    return make_gemm_kernel_config(ep, cfg, cfg.format);
}

GemmKernel make_gemm_kernel_config(GemmEpilogue ep, const GemmTileConfig &cfg, GemmWeightFormat wf) noexcept {
    auto i = entry_by_name(cfg.name);
    if (i < 0) {
        LUISA_WARNING("embed gemm: config \"{}\" is not in the variant table; using the default "
                      "variant \"{}\".",
                      cfg.name, kEntries[kDefaultEntry].cfg.name);
        i = static_cast<int32_t>(kDefaultEntry);
    }
    const auto &e = kEntries[static_cast<uint32_t>(i)];
    // The DSL kernel name is the variant name (a static literal from the table);
    // the epilogue shows up in the executor's ShaderOption name and in the
    // harness's compiled-shader labels.
    return e.builder(ep, wf, e.cfg.name);
}

GemmKernel make_gemm_kernel(GemmEpilogue ep) noexcept {
    return make_gemm_kernel_config(ep, gemm_resolved_config());
}

GemmKernel make_gemm_kernel(GemmEpilogue ep, GemmWeightFormat wf) noexcept {
    return make_gemm_kernel_config(ep, gemm_resolved_config(), wf);
}

uint2 gemm_dispatch_size_config(const GemmTileConfig &cfg, uint32_t m, uint32_t n) noexcept {
    const auto tiles_n = (n + cfg.tile_n - 1u) / cfg.tile_n;
    const auto tiles_m = (m + cfg.tile_m - 1u) / cfg.tile_m;
    if (cfg.swap_grid) {
        return {tiles_m * cfg.block, tiles_n};
    }
    return {tiles_n * cfg.block, tiles_m};
}

uint2 gemm_dispatch_size(uint32_t m, uint32_t n) noexcept {
    return gemm_dispatch_size_config(gemm_resolved_config(), m, n);
}

ResidualAddKernel make_residual_add_kernel() noexcept {
    using namespace luisa::compute;
    ResidualAddKernel kernel = [](compute::BufferFloat a, compute::BufferFloat b,
                                  compute::BufferFloat out, Var<uint> count) noexcept {
        set_block_size(kResidualAddBlock, 1u, 1u);
        set_name("embed_residual_add");
        // Elementwise residual `Add` (plan §4 #7). dispatch() counts TOTAL
        // THREADS, not elements (lc_optimize trap #1), so the tail thread is
        // guarded. Flattened 2D-grid idiom (same as kernels_embed/kernels_norm):
        // correct for a plain 1D launch (y == 0) AND for a host-flattened 2D
        // launch (x = bs*cols, y = blocks/cols) -- required because the DX grid
        // cap is 65535 groups and count = rows*H overruns it for big batches
        // (test-plan wave T5/T6b finding).
        const Var tx = thread_x();
        const Var bs = block_size().x;
        const Var blk = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        const Var i = blk * bs + tx;
        $if (i < count) {
            out.write(i, a.read(i) + b.read(i));
        };
    };
    return kernel;
}

}// namespace luisa::embed
