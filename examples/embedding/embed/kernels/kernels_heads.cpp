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
        Var i = dispatch_id().x;// 1D element copy over H (H <= 8194 << 2^31)
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
            sync_block();// protect red against reuse by the next block_sum
            return total;
        };

        Var partial = def(0.0f);
        Var h = tx;
        $while (h < H) {
            partial += X.read(base + h) * W_s.read(h);
            h += bs;
        };
        Var v = block_sum(partial) + b_s;// dot + bias Add
        Var y = def(0.0f);               // Relu epilogue
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
            sync_block();// protect red against reuse by the next block_sum
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
            }
            $else {
                Var hh = def(0u);
                $while (hh < H) {
                    acc += X.read(base + hh) * W_c.read(hh * N + n);
                    hh += 1u;
                };
            };
            Var y = acc + b_c.read(n);// bias Add
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

SparseHeadCompactKernel make_sparse_head_compact_kernel() noexcept {
    using namespace luisa::compute;

    // P1 pairs mode (opt plan §3.4 P1): identical y computation as
    // sparse_head (same strided partials, same shared halving tree, same bias
    // + relu -- bit-identical weights), then the (row, weight) pair of every
    // REAL row with a positive weight is appended to a compact stream via
    // block-local privatization -> ONE global atomic per block
    // (lc_optimize §4.3, the test_atomic_queue::push_if pattern). No warp
    // collectives: the CPU fallback backend lowers atomics + barriers but not
    // warp shuffles. The emission predicate mask[row] != 0 bounds the stream
    // by the host-known real-token count, so the pair buffers (capacity
    // rows) can never overflow and the recorded readback span can be that
    // exact bound.
    SparseHeadCompactKernel kernel = [](BufferFloat X, BufferFloat W_s, Var<float> b_s,
                                        BufferFloat mask, BufferUInt pair_ids,
                                        BufferFloat pair_w, BufferUInt counter,
                                        Var<uint> H) noexcept {
        set_block_size(kHeadsBlock, 1u, 1u);
        set_name("sparse_head_compact");
        Shared<float> red{kHeadsBlock};
        Shared<uint> index{1u};

        Var tx = thread_x();
        Var bs = block_size().x;
        Var row = dispatch_id().y * (dispatch_size().x / bs) + dispatch_id().x / bs;
        Var base = row * H;

        // Block reduction: shared-memory halving tree over the 256 per-thread
        // partials -- EXACTLY the idiom, strides and summation order of
        // sparse_head above (see its comment; bit-stable on every backend).
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
            sync_block();// protect red against reuse by the next block_sum
            return total;
        };

        Var partial = def(0.0f);
        Var h = tx;
        $while (h < H) {
            partial += X.read(base + h) * W_s.read(h);
            h += bs;
        };
        Var v = block_sum(partial) + b_s;// dot + bias Add
        Var y = def(0.0f);               // Relu epilogue
        $if (v > 0.0f) {
            y = v;
        };

        // Stream compaction (§4.3): one block emits at most one pair (one
        // scalar per row), so the per-item predicate is restricted to thread 0
        // -- the privatized shared counter then carries exactly 0 or 1 and the
        // block reserves exactly one slot with ONE global atomic (the same
        // machinery a future multi-value head would drive with a per-thread
        // pred). (A block-uniform pred here is the classic §4.3 trap: all 256
        // threads append duplicate items and blow the capacity bound 256x.)
        Var emit = def(false);
        $if (tx == 0u) {
            $if (y > 0.0f) {
                $if (mask.read(row) != 0.0f) {
                    emit = true;
                };
            };
        };
        $if (tx == 0u) {
            index.write(0u, 0u);
        };
        sync_block();
        Var local_index = def(0u);
        $if (emit) {
            local_index = index.atomic(0u).fetch_add(1u);
        };
        sync_block();
        $if (tx == 0u) {// ONE global atomic for the whole block
            Var local_count = index.read(0u);
            Var global_offset = counter.atomic(0u).fetch_add(local_count);
            index.write(0u, global_offset);// reuse the slot to broadcast the base
        };
        sync_block();
        $if (emit) {// scatter to the reserved slot
            Var slot = index.read(0u) + local_index;
            pair_ids.write(slot, row);
            pair_w.write(slot, y);
        };
    };
    return kernel;
}

SparseCountResetKernel make_sparse_count_reset_kernel() noexcept {
    using namespace luisa::compute;

    // Zeroes the pair-stream counter before the sparse_head_compact dispatch
    // (same pattern as test_atomic_queue's _reset; block 32 = the DSL's warp
    // multiple, only thread 0 writes).
    SparseCountResetKernel kernel = [](BufferUInt counter) noexcept {
        set_block_size(32u, 1u, 1u);
        set_name("sparse_count_reset");
        $if (thread_x() == 0u) {
            counter.write(0u, 0u);
        };
    };
    return kernel;
}

}// namespace luisa::embed
