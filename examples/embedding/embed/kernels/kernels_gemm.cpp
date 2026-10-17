// bge-m3 embedding engine -- GEMM with fused epilogues (plan §4 kernel #3) and
// residual-add (plan §4 kernel #7) implementations. See kernels_gemm.h for the
// contract and launch geometry.
//
// ONNX-op mapping (plan §4 table, row #3): MatMul x192 (QKV projection,
// attention output projection, FFN up/down, retrieval heads) fused with the
// following bias `Add`, `Erf`-GELU / `Relu`, or residual `Add` chains; row #7
// covers the residual `Add` edges that are not fused into a GEMM epilogue.
//
// Tiling: one 16x16 C tile per block, 256 threads (thread (tr, tc) computes
// C[m0+tr][n0+tc]); K streams through two 16x16 shared tiles (2 * 16 * 16 * 4 B
// = 2 KiB per block, far under the 32 KiB DX floor). A/B tile loads are guarded
// and zero-filled, so arbitrary ragged M/N/K are correct. B reads and the C
// store are contiguous along N -- the packer pre-transposes every weight to
// [K,N] (plan §3.3), so all global traffic coalesces along N. fp32 accumulate.
#include "embedding/embed/kernels/kernels_gemm.h"

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {

using compute::def;
using compute::Var;

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

// Compile-time epilogue selection (`if constexpr` on the flag template
// argument): the unselected epilogue math never reaches the recorded DSL.
// `aux` carries bias[N] for the bias flags and the residual R[M,N] for
// ResidualAdd; the None flag never touches it (callers bind a dummy view).
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

template<GemmEpilogue EP>
[[nodiscard]] GemmKernel make_gemm_kernel_impl() noexcept {
    using namespace luisa::compute;
    GemmKernel kernel = [](compute::BufferFloat A, compute::BufferFloat B,
                           compute::BufferFloat C, compute::BufferFloat aux,
                           Var<uint> M, Var<uint> N, Var<uint> K) noexcept {
        set_block_size(kGemmBlock, 1u, 1u);
        set_name(gemm_epilogue_name(EP));

        compute::Shared<float> tile_a{kGemmTile * kGemmTile};
        compute::Shared<float> tile_b{kGemmTile * kGemmTile};

        const Var tid = thread_x();
        const Var tr = tid / kGemmTile;// row inside the 16x16 output tile
        const Var tc = tid % kGemmTile;// column inside the tile (along N)
        const Var m = block_id().y * kGemmTile + tr;
        const Var n = block_id().x * kGemmTile + tc;

        Var acc = def(0.0f);
        $for (kt, (K + kGemmTile - 1u) / kGemmTile) {
            const Var k0 = kt * kGemmTile;
            // A tile row m, column k0 + tc: contiguous along K in
            // row-major A.
            compute::Float a_val;
            $if (m < M & (k0 + tc) < K) {
                a_val = A.read(m * K + k0 + tc);
            }
            $else {
                a_val = def(0.0f);
            };
            tile_a[tr * kGemmTile + tc] = a_val;
            // B tile row k0 + tr, column n: contiguous along N (coalesced).
            compute::Float b_val;
            $if ((k0 + tr) < K & n < N) {
                b_val = B.read((k0 + tr) * N + n);
            }
            $else {
                b_val = def(0.0f);
            };
            tile_b[tr * kGemmTile + tc] = b_val;
            sync_block();
            // Fixed compile-time unroll over the K sub-tile (plain C++ loop).
            for (auto kk = 0u; kk < kGemmTile; kk++) {
                acc += tile_a[tr * kGemmTile + kk] * tile_b[kk * kGemmTile + tc];
            }
            sync_block();
        };

        $if (m < M & n < N) {
            C.write(m * N + n, gemm_epilogue<EP>(acc, aux, m, n, N));
        };
    };
    return kernel;
}

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

GemmKernel make_gemm_kernel(GemmEpilogue ep) noexcept {
    switch (ep) {
        case GemmEpilogue::None: return make_gemm_kernel_impl<GemmEpilogue::None>();
        case GemmEpilogue::Bias: return make_gemm_kernel_impl<GemmEpilogue::Bias>();
        case GemmEpilogue::BiasGeluErf: return make_gemm_kernel_impl<GemmEpilogue::BiasGeluErf>();
        case GemmEpilogue::BiasRelu: return make_gemm_kernel_impl<GemmEpilogue::BiasRelu>();
        case GemmEpilogue::ResidualAdd: return make_gemm_kernel_impl<GemmEpilogue::ResidualAdd>();
    }
    LUISA_ERROR("Invalid GemmEpilogue value: {}.", static_cast<uint32_t>(ep));
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
