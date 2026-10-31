// bge-m3 embedding engine -- normalization kernel implementations
// (plan §4 kernels #2 layernorm and #9 l2_normalize; see kernels_norm.h for the
// ONNX-op mapping and the launch contract).
//
// Same conventions as kernels_embed.cpp (plan §4 #1): device-free Kernel1D
// factories, explicit block 256, dispatch() = TOTAL THREADS, runtime row
// length, row index computed from a flattened (possibly host-2D) grid. The
// block reduction is a shared-memory halving tree instead of the warp->shared
// two-level idiom, because the CPU fallback backend only implements
// SYNCHRONIZE_BLOCK among the thread-group ops (no warp collectives), and the
// fallback run is part of the numerical parity gate (plan §8); the tree is
// bit-stable across backends.
#include "embedding/embed/kernels/kernels_norm.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {

using compute::def;
using compute::Var;

}// namespace

LayerNormKernel make_layernorm_kernel() noexcept {
    using namespace luisa::compute;

    // One block per row. Two-pass mean/variance in fp32 -- the same math order
    // as the naive ONNX cluster (ReduceMean -> Sub -> Pow -> ReduceMean ->
    // Div -> Sqrt -> Mul -> Add) and as the test's CPU reference:
    //   mean = sum(x) / H            (ReduceMean)
    //   var  = sum((x-mean)^2) / H   (Sub, Pow, ReduceMean; biased, no Bessel)
    //   y    = ((x-mean) * (1/sqrt(var+eps))) * gamma + beta
    LayerNormKernel kernel = [](BufferFloat x, BufferFloat y, BufferFloat gamma,
                                BufferFloat beta, Var<uint> H, Var<float> eps) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("layernorm");

        Shared<float> red{kNormBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        // Flattened block index -> row; valid for 1D dispatch (y == 0) and for a
        // host-flattened 2D grid (DX 65535-group cap).
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;
        Var Hf = H.cast<float>();

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials. A warp-collective reduction (warp_active_sum) would be
        // cheaper on dx/vk, but the CPU fallback backend implements only
        // SYNCHRONIZE_BLOCK among the thread-group ops
        // (src/backends/fallback/fallback_codegen.cpp::_translate_thread_group_inst),
        // so warp intrinsics abort there -- the tree below lowers to plain
        // shared memory + barriers and therefore behaves identically (and
        // bit-identically, same fixed summation order) on every backend,
        // including the deterministic parity oracle.
        //
        // Every thread calls it the same number of times (block-uniform control
        // flow only), and each call ends with a barrier so `red` is not
        // overwritten before all threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        // Pass 1: sum(x). Threads with tx >= H (H < block size) contribute 0;
        // the $while stride covers H > block size.
        Var s1 = def(0.0f);
        Var h = tx;
        $while (h < H) {
            s1 += x.read(base + h);
            h += bs;
        };
        Var mean = block_sum(s1) / Hf;

        // Pass 2: sum((x - mean)^2).
        Var s2 = def(0.0f);
        h = tx;
        $while (h < H) {
            Var d = x.read(base + h) - mean;
            s2 += d * d;
            h += bs;
        };
        Var var = block_sum(s2) / Hf;

        // Pass 3: affine normalize + write. The CPU reference uses this exact
        // expression order.
        Var inv = 1.0f / sqrt(var + eps);
        h = tx;
        $while (h < H) {
            Var xv = x.read(base + h);
            Var yv = ((xv - mean) * inv) * gamma.read(h) + beta.read(h);
            y.write(base + h, yv);
            h += bs;
        };
    };
    return kernel;
}

bool residual_layernorm_supported(uint32_t h) noexcept {
    // h == 0 would divide by Hf; an empty row is never a real shape, so refuse it
    // rather than emit NaNs. The upper bound is the static shared cache.
    return h >= 1u && h <= kResidualNormMaxHidden;
}

ResidualLayerNormKernel make_residual_layernorm_kernel() noexcept {
    using namespace luisa::compute;

    // Fused `residual Add (#7)` -> `LayerNorm (#2)` (opt plan C3). One block per
    // row; the residual row is computed once and cached in shared memory so the
    // layernorm's three passes read it on chip instead of re-reading global
    // memory three times. The pair it replaces traverses the row plane 7 times
    // (residual 2R+1W, layernorm 3R+1W); this kernel traverses it 3 times
    // (2R + 1W) and costs 1 dispatch instead of 2, twice per encoder layer.
    //
    // The reduction tree, the per-thread strided accumulation order and the
    // final expression order are VERBATIM copies of make_layernorm_kernel()'s
    // (see the comment there: a shared-memory halving tree, not warp
    // collectives, because the CPU fallback implements only SYNCHRONIZE_BLOCK
    // among the thread-group ops and is part of the numerical parity gate).
    // Combined with the exact fp32 residual `a + b` -- the same add, of the same
    // two operands, as make_residual_add_kernel() -- the output is NUMERICALLY
    // EQUIVALENT to running the legacy pair within the focused 1e-5 ABSOLUTE
    // tolerance that test_embed_kernel_layernorm asserts, because the reduction
    // order above is preserved exactly. It is deliberately NOT claimed to be
    // bit-identical: the same op sequence emitted as one kernel instead of two
    // lets dx re-emit ~1-ULP differences against the pair (measured: pair
    // -0.13910183 vs fused -0.13910185), so bit-identity is not a property of
    // the fusion on every backend.
    //
    // H <= kResidualNormMaxHidden is the host contract (see kernels_norm.h);
    // `slot()` below only keeps a contract-violating dispatch from indexing past
    // the shared array -- it does not make H > kResidualNormMaxHidden correct,
    // which is why residual_layernorm_supported() gates the dispatch.
    ResidualLayerNormKernel kernel = [](BufferFloat a, BufferFloat b, BufferFloat out,
                                        BufferFloat gamma, BufferFloat beta,
                                        Var<uint> H, Var<float> eps) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("residual_layernorm");

        Shared<float> res{kResidualNormMaxHidden};// cached residual row (H <= 1024)
        Shared<float> red{kNormBlock};            // block reduction tree

        Var tx = thread_x();
        Var bs = block_size().x;
        // Flattened block index -> row; valid for 1D dispatch (y == 0) and for a
        // host-flattened 2D grid (DX 65535-group cap).
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;
        Var Hf = H.cast<float>();

        // Shared-cache index for row element h, clamped into the cache so an
        // out-of-contract dispatch cannot read/write past `res` (the values are
        // then meaningless -- but only an unsupported H can reach that state).
        auto slot = [](Var<uint> h) noexcept {
            return min(h, kResidualNormMaxHidden - 1u);
        };

        // Verbatim copy of make_layernorm_kernel()'s block reduction (see the
        // long comment there): every thread calls it the same number of times
        // (block-uniform control flow only) and each call ends with a barrier.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block();
            return total;
        };

        // Pass 1: residual + sum(res). The residual `a + b` is the same fp32 add
        // the legacy residual_add kernel performs, and it is computed ONCE here
        // and cached; both caches write and the reduction are published by
        // block_sum's leading barrier, so every thread can read the whole row
        // from `res` in the following passes. Threads with tx >= H (H < block
        // size) contribute 0; the $while stride covers H > block size.
        Var s1 = def(0.0f);
        Var h = tx;
        $while (h < H) {
            Var rv = a.read(base + h) + b.read(base + h);
            res[slot(h)] = rv;
            s1 += rv;
            h += bs;
        };
        Var mean = block_sum(s1) / Hf;

        // Pass 2: sum((res - mean)^2) from the cache.
        Var s2 = def(0.0f);
        h = tx;
        $while (h < H) {
            Var d = res[slot(h)] - mean;
            s2 += d * d;
            h += bs;
        };
        Var var = block_sum(s2) / Hf;

        // Pass 3: affine normalize + write. Reached only after pass 2's barrier,
        // so every residual read of `a`/`b` is complete -- writing `out` here is
        // safe even when `out` aliases `a`. The CPU reference uses this exact
        // expression order.
        Var inv = 1.0f / sqrt(var + eps);
        h = tx;
        $while (h < H) {
            Var yv = ((res[slot(h)] - mean) * inv) * gamma.read(h) + beta.read(h);
            out.write(base + h, yv);
            h += bs;
        };
    };
    return kernel;
}

L2NormalizeKernel make_l2_normalize_kernel() noexcept {
    using namespace luisa::compute;

    // One block per row: sum of squares (ReduceSum/Mul chain), sqrt (ReduceL2),
    // then y = x / max(norm, 1e-12) -- the torch F.normalize clamp, so an
    // all-zero row stays zero instead of NaN.
    L2NormalizeKernel kernel = [](BufferFloat x, BufferFloat y, Var<uint> H) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("l2_normalize");

        Shared<float> red{kNormBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;

        // Same shared-memory tree reduction as layernorm (fallback has no warp
        // collectives); see the comment in make_layernorm_kernel().
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
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
        Var h = tx;
        $while (h < H) {
            Var xv = x.read(base + h);
            s += xv * xv;
            h += bs;
        };
        Var denom = max(sqrt(block_sum(s)), def(1e-12f));

        h = tx;
        $while (h < H) {
            y.write(base + h, x.read(base + h) / denom);
            h += bs;
        };
    };
    return kernel;
}

DensePoolKernel make_dense_pool_kernel() noexcept {
    using namespace luisa::compute;

    // Fused CLS-slice + L2-normalize (plan §3.3 N1(b); replaces the per-row
    // slice_cls(#8) + l2_normalize(#9) pair). One block per batch row b: reads the
    // CLS hidden row hidden[b*S .. b*S+1) directly out of the [B*S, H] final hidden
    // state and writes its L2-normalization into out[b] ([B, H]). No final LayerNorm
    // exists in bge-m3 -- layer 24 output is already post-LN2 (see kernels_norm.h).
    //
    // The reduction below is a verbatim copy of make_l2_normalize_kernel()'s tree,
    // strided accumulation and formula, just re-targeted at hidden[b*S*H + h]
    // instead of a pre-sliced _cls row, so the output is BIT-identical to running
    // slice_cls then l2_normalize (the slice is a pure copy; the same H floats are
    // reduced in the same order). The CPU fallback keeps the shared-memory tree
    // (no warp collectives); summation order is block-uniform and fixed, so
    // dx/vk/fallback agree bit-for-bit.
    DensePoolKernel kernel = [](BufferFloat hidden, BufferFloat out, Var<uint> S,
                                Var<uint> H) noexcept {
        set_block_size(kNormBlock, 1u, 1u);
        set_name("dense_pool");

        Shared<float> red{kNormBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        // Per-block batch index (flattened, valid for 1D dispatch and a host-
        // flattened 2D grid that must dodge the DX 65535-group cap; lc_optimize
        // trap #2). For the sync 1D launch grid.y == 0.
        Var b = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = b * (S * H);// CLS row of batch b in hidden
        Var obase = b * H;     // output row

        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            for (auto stride = kNormBlock / 2u; stride > 0u; stride >>= 1u) {
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
        Var h = tx;
        $while (h < H) {
            Var xv = hidden.read(base + h);
            s += xv * xv;
            h += bs;
        };
        Var denom = max(sqrt(block_sum(s)), def(1e-12f));

        h = tx;
        $while (h < H) {
            out.write(obase + h, hidden.read(base + h) / denom);
            h += bs;
        };
    };
    return kernel;
}

}// namespace luisa::embed
