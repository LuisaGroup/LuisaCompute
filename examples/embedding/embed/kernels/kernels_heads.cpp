// bge-m3 embedding engine -- retrieval-head kernel implementations
// (plan §4 kernels #8, #10; see kernels_heads.h for the ONNX-op mapping).
//
// Same conventions as kernels_embed.cpp: device-free Kernel1D factories,
// explicit block 256, dispatch() = TOTAL THREADS, runtime dims, shared-memory
// halving-tree block reduction (fallback-backend safe; same idiom and
// summation order as kernels_norm.cpp), row index valid for host-flattened 2D
// grids (DX 65535-group cap). Shared row cache covers H <= 2048; larger H
// re-gathers from global memory (correct but slower; bge-m3 never hits it).
//
// NOTE for Phase B: the executor may route these projections through the
// shared GEMM kernel (#3); these factories stay as a self-contained fallback
// and as the correctness oracle used by test_embed_kernel_heads.
#include "embedding/embed/kernels/kernels_heads.h"

#include <luisa/dsl/sugar.h>

namespace luisa::embed {

namespace {
// Maximum cached row width in shared memory (4 B * (kHeadsBlock + 2048) <=
// 32 KiB shared floor).
constexpr uint kMaxCachedRow = 2048u;
}// namespace

SliceClsKernel make_slice_cls_kernel() noexcept {
    using namespace luisa::compute;

    // Slice / pooling Gather (plan §4 #8): the first row (CLS token) of the
    // final hidden state is just the first H floats of the [B*S, H] buffer.
    SliceClsKernel kernel = [](BufferFloat hidden, BufferFloat cls_out, Var<uint> H) noexcept {
        set_block_size(kHeadsBlock, 1u, 1u);
        set_name("slice_cls");
        Var i = dispatch_id().x; // 1D element copy over H (H <= 8194 << 2^31)
        $if (i < H) {
            cls_out.write(i, hidden.read(i));
        };
    };
    return kernel;
}

SparseHeadKernel make_sparse_head_kernel() noexcept {
    using namespace luisa::compute;

    // sparse_linear (plan §4 #10, ONNX MatMul[1024x1] + Add + Relu):
    // out[t] = relu(dot(X[t, :], W_s) + b_s), one scalar per token row.
    // A block per token; the dot is a strided thread partial sum folded
    // with the shared-memory tree block reduction. Host-side thresholding/
    // compaction of the (token_id, weight) pairs follows the readback
    // (Phase A decision).
    SparseHeadKernel kernel = [](BufferFloat X, BufferFloat W_s, Var<float> b_s,
                                 BufferFloat out, Var<uint> H) noexcept {
        set_block_size(kHeadsBlock, 1u, 1u);
        set_name("sparse_head");
        Shared<float> red{kHeadsBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        // Flattened-block row index (1D dispatch -> y == 0).
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials (same idiom and summation order as kernels_norm.cpp; the
        // CPU fallback backend implements only SYNCHRONIZE_BLOCK among the
        // thread-group ops, so warp intrinsics abort there -- the tree lowers
        // to plain shared memory + barriers and is bit-stable on every
        // backend). Each call ends with a barrier so `red` is not overwritten
        // before all threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kHeadsBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block(); // protect red against reuse by the next block_sum
            return total;
        };

        Var partial = def(0.0f);
        Var h = tx;
        $while (h < H) {
            partial += X.read(base + h) * W_s.read(h);
            h += bs;
        };
        Var v = block_sum(partial) + b_s; // dot + bias Add
        Var y = def(0.0f);                // Relu epilogue
        $if (v > 0.0f) {
            y = v;
        };
        $if (tx == 0u) {
            out.write(row, y);
        };
    };
    return kernel;
}

ColbertProjKernel make_colbert_proj_kernel() noexcept {
    using namespace luisa::compute;

    // colbert_linear (plan §4 #10, ONNX MatMul[1024x1024] + Add; the caller
    // L2-normalizes per token and drops special tokens host-side):
    // y[t, n] = sum_h X[t, h] * W_c[h, n] + b_c[n], with the runtime `normalize`
    // flag folding kernel #9 (l2_normalize) into the same launch.
    ColbertProjKernel kernel = [](BufferFloat X, BufferFloat W_c, BufferFloat b_c,
                         BufferFloat out, Var<uint> H, Var<uint> N,
                         Var<uint> normalize) noexcept {
        set_block_size(kHeadsBlock, 1u, 1u);
        set_name("colbert_proj");
        Shared<float> xs{kMaxCachedRow};
        Shared<float> red{kHeadsBlock};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;
        Var obase = row * N;

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials (same idiom and summation order as kernels_norm.cpp; the
        // CPU fallback backend implements only SYNCHRONIZE_BLOCK among the
        // thread-group ops, so warp intrinsics abort there -- the tree lowers
        // to plain shared memory + barriers and is bit-stable on every
        // backend). Each call ends with a barrier so `red` is not overwritten
        // before all threads have read the result.
        auto block_sum = [&](Var<float> partial) noexcept {
            red[tx] = partial;
            sync_block();
            // Plain C++ loop: compile-time-constant strides, fixed unroll.
            for (auto stride = kHeadsBlock / 2u; stride > 0u; stride >>= 1u) {
                $if (tx < stride) {
                    red[tx] += red[tx + stride];
                };
                sync_block();
            };
            Var total = red[0u];
            sync_block(); // protect red against reuse by the next block_sum
            return total;
        };

        // Cooperative X-row load into shared memory when it fits (the block
        // reduction below needs the barrier either way; for the H > fallback
        // the row is re-read from global inside the dot loop).
        $if (H <= kMaxCachedRow) {
            Var h = tx;
            $while (h < H) {
                xs[h] = X.read(base + h);
                h += bs;
            };
        };
        sync_block();

        // Each thread owns outputs n = tx, tx + bs, ... — a [1, bs, N] tile of
        // the GEMM; W_c reads coalesce along n (row-major [H, N], pre-
        // transposed per plan §3.3).
        Var sum_sq = def(0.0f);
        Var n = tx;
        $while (n < N) {
            Var acc = def(0.0f);
            $if (H <= kMaxCachedRow) {
                Var hh = def(0u);
                $while (hh < H) {
                    acc += xs[hh] * W_c.read(hh * N + n);
                    hh += 1u;
                };
            } $else {
                Var hh = def(0u);
                $while (hh < H) {
                    acc += X.read(base + hh) * W_c.read(hh * N + n);
                    hh += 1u;
                };
            };
            Var y = acc + b_c.read(n); // bias Add
            out.write(obase + n, y);
            $if (normalize != 0u) {
                sum_sq += y * y;
            };
            n += bs;
        };

        // Fused per-row L2 normalization (plan §4 #9): divide by
        // max(||y||_2, 1e-12) — torch F.normalize semantics. Each thread
        // re-reads only the elements it wrote (same-thread global RAW is
        // coherent), so only the reduction needs the barriers (inside
        // block_sum).
        $if (normalize != 0u) {
            Var tot = block_sum(sum_sq);
            Var d = sqrt(tot);
            $if (d < 1e-12f) {
                d = 1e-12f;
            };
            Var n2 = tx;
            $while (n2 < N) {
                out.write(obase + n2, out.read(obase + n2) / d);
                n2 += bs;
            };
        };
    };
    return kernel;
}

}// namespace luisa::embed
