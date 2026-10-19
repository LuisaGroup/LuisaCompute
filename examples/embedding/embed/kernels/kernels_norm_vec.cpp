// bge-m3 embedding engine -- vec4 (float4) normalization kernel implementations
// (tensor_optimize.md §4.1 V0 vec4 infrastructure + §4.2 V1 bandwidth kernels;
// see kernels_norm_vec.h for the host contract, the API and the load-width-only
// delta versus kernels_norm.cpp).
//
// Same conventions as kernels_norm.cpp: device-free Kernel1D factories, explicit
// block kNormVecBlock, dispatch() = TOTAL THREADS, runtime row length, row index
// computed from a flattened (possibly host-2D) grid. The block reduction is the
// same shared-memory halving tree -- NOT a warp collective -- because the CPU
// fallback backend only implements SYNCHRONIZE_BLOCK among the thread-group ops
// (src/backends/fallback/fallback_codegen.cpp::_translate_thread_group_inst) and
// the fallback run is part of the numerical parity gate (plan §8).
#include "embedding/embed/kernels/kernels_norm_vec.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {

using compute::def;
using compute::Var;

}// namespace

LayerNormVec4Kernel make_layernorm_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // One block per row, float4 loads. Two-pass mean/variance in fp32 -- the same
    // math order as the scalar legacy kernel and the CPU reference, just widened:
    //   mean = sum(x) / H                (ReduceMean)
    //   var  = sum((x-mean)^2) / H       (Sub, Pow, ReduceMean; biased, no Bessel)
    //   y    = ((x - mean) * (1/sqrt(var + eps))) * gamma + beta
    // Written back into x (in-place). H (element count) is a runtime arg and the
    // caller guarantees H % 4 == 0; H4 = H/4 lanes. The strided $while over h4
    // covers H4 > block size (ragged-safe per vec4; the caller's H%4==0 covers the
    // element tail), while H4 < block size leaves idle threads that contribute 0.
    LayerNormVec4Kernel kernel = [](BufferFloat4 x, BufferFloat4 gamma,
                                    BufferFloat4 beta, Var<uint> H,
                                    Var<float> eps) noexcept {
        set_block_size(kNormVecBlock, 1u, 1u);
        set_name("layernorm_vec4");

        Shared<float> red{kNormVecBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        // Flattened block index -> row; valid for 1D dispatch (y == 0) and for a
        // host-flattened 2D grid (DX 65535-group cap).
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var H4  = H / 4u;
        Var base = row * H4;
        Var Hf   = H.cast<float>();

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // scalar partials (each thread reduces its own float4 lanes to a scalar
        // before the tree -- the lane-wise horizontal sum order is PINNED to
        // (acc4.x + acc4.y) + (acc4.z + acc4.w); see kernels_norm_vec.h). The
        // tree is copied verbatim from make_layernorm_kernel() so the per-stride
        // barrier count (1 + log2(block) + 1 = 10) and fold order are identical.
        // Every thread calls it the same number of times (block-uniform control
        // flow only), and each call ends with a barrier so `red` is not
        // overwritten before all threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kNormVecBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        // Pass 1: sum(x). Each thread accumulates its own float4 lanes (PINNED
        // horizontal order, never dot()) into a scalar that feeds the tree.
        Var s1 = def(0.0f);
        Var h4 = tx;
        $while (h4 < H4) {
            Var<float4> xv = x.read(base + h4);
            s1 += (xv.x + xv.y) + (xv.z + xv.w);
            h4 += bs;
        };
        Var mean = block_sum(s1) / Hf;

        // Pass 2: sum((x - mean)^2). mean broadcasts to every lane.
        Var s2 = def(0.0f);
        h4 = tx;
        $while (h4 < H4) {
            Var<float4> xv = x.read(base + h4);
            Var<float4> d4 = xv - mean;
            s2 += (d4.x * d4.x + d4.y * d4.y) + (d4.z * d4.z + d4.w * d4.w);
            h4 += bs;
        };
        Var var = block_sum(s2) / Hf;

        // Pass 3: affine normalize + write back (in-place). The CPU reference uses
        // this exact expression order; divide-free (multiply by reciprocal). The
        // scalar mean/inv broadcast across the float4 lanes, gamma/beta are float4.
        Var inv = 1.0f / sqrt(var + eps);
        h4 = tx;
        $while (h4 < H4) {
            Var<float4> xv = x.read(base + h4);
            Var<float4> yv = ((xv - mean) * inv) * gamma.read(h4) + beta.read(h4);
            x.write(base + h4, yv);
            h4 += bs;
        };
    };
    return kernel;
}

L2NormalizeVec4Kernel make_l2_normalize_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // One block per row, float4 loads: sum of squares (ReduceSum/Mul chain),
    // sqrt (ReduceL2), then y = x / max(norm, 1e-12) -- the torch F.normalize
    // clamp, kept as-is, so an all-zero row stays zero instead of NaN.
    L2NormalizeVec4Kernel kernel = [](BufferFloat4 x, BufferFloat4 y, Var<uint> H) noexcept {
        set_block_size(kNormVecBlock, 1u, 1u);
        set_name("l2_normalize_vec4");

        Shared<float> red{kNormVecBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var H4  = H / 4u;
        Var base = row * H4;

        // Same shared-memory tree reduction as the scalar l2_normalize (fallback
        // has no warp collectives); see the comment in make_layernorm_vec4_kernel().
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormVecBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        // s = lane-wise sum of squares; PINNED horizontal order (never dot()).
        Var s = def(0.0f);
        Var h4 = tx;
        $while (h4 < H4) {
            Var<float4> xv = x.read(base + h4);
            s += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
            h4 += bs;
        };
        Var denom = max(sqrt(block_sum(s)), def(1e-12f));

        // Divide by the scalar norm broadcast across the float4 lanes (matches the
        // legacy divide, not a reciprocal multiply).
        Var d4 = make_float4(denom);
        h4 = tx;
        $while (h4 < H4) {
            y.write(base + h4, x.read(base + h4) / d4);
            h4 += bs;
        };
    };
    return kernel;
}

DensePoolVec4Kernel make_dense_pool_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // Fused CLS-slice + L2-normalize, float4 loads (vec4 twin of
    // make_dense_pool_kernel()). One block per batch row b: reads the CLS hidden
    // row hidden[b*S .. b*S+1) straight out of the [B*S, H] final hidden state
    // (all indices in float4 units) and writes its L2-normalization into out[b].
    //
    // The reduction below is a verbatim copy of make_l2_normalize_vec4_kernel()'s
    // tree, strided accumulation and formula, re-targeted at the CLS row of batch
    // b instead of a pre-sliced _cls row. The clamp y = x / max(||x||_2, 1e-12) is
    // kept; summation order is block-uniform and fixed, so dx/vk/cuda/fallback
    // agree bit-for-bit among themselves (against the scalar legacy the widened
    // load changes the partial set -- validated at the numeric tolerance gate).
    DensePoolVec4Kernel kernel = [](BufferFloat4 hidden, BufferFloat4 out, Var<uint> S,
                                    Var<uint> H) noexcept {
        set_block_size(kNormVecBlock, 1u, 1u);
        set_name("dense_pool_vec4");

        Shared<float> red{kNormVecBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        // Per-block batch index (flattened, valid for 1D dispatch and a host-
        // flattened 2D grid that must dodge the DX 65535-group cap; lc_optimize
        // trap #2). For the sync 1D launch grid.y == 0.
        Var b  = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var H4 = H / 4u;
        // Row stride of `hidden` is S*H elements = S*H4 float4; base is the CLS
        // row of batch b; obase is the output row.
        Var base  = b * (S * H4);
        Var obase = b * H4;

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormVecBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        Var s = def(0.0f);
        Var h4 = tx;
        $while (h4 < H4) {
            Var<float4> xv = hidden.read(base + h4);
            s += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
            h4 += bs;
        };
        Var denom = max(sqrt(block_sum(s)), def(1e-12f));

        Var d4 = make_float4(denom);
        h4 = tx;
        $while (h4 < H4) {
            out.write(obase + h4, hidden.read(base + h4) / d4);
            h4 += bs;
        };
    };
    return kernel;
}

ResidualAddVec4Kernel make_residual_add_vec4_kernel() noexcept {
    using namespace luisa::compute;

    // Elementwise residual `Add` (vec4 twin of make_residual_add_kernel() in
    // kernels_gemm.cpp, plan §4 #7). dispatch() counts TOTAL THREADS (lc_optimize
    // trap #1), so the tail thread is guarded; i indexes float4 elements.
    // Flattened 2D-grid idiom (same as kernels_norm / kernels_embed): correct for
    // a plain 1D launch (y == 0) AND a host-flattened 2D launch (required because
    // the DX grid cap is 65535 groups).
    ResidualAddVec4Kernel kernel = [](compute::BufferFloat4 a, compute::BufferFloat4 b,
                                      compute::BufferFloat4 out, Var<uint> count4) noexcept {
        set_block_size(kNormVecBlock, 1u, 1u);
        set_name("embed_residual_add_vec4");
        const Var tx  = thread_x();
        const Var bs  = block_size().x;
        const Var blk = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        const Var i   = blk * bs + tx;
        $if (i < count4) {
            out.write(i, a.read(i) + b.read(i));
        };
    };
    return kernel;
}

}// namespace luisa::embed
