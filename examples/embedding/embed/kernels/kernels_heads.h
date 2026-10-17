// bge-m3 embedding engine -- retrieval-head kernels (plan §4 kernels #8, #10).
//
// #8 slice_cls:    copies row 0 (the <s>/CLS token row) out of the final
//                  hidden state [B*S, H] — the first Gather/Slice of the dense
//                  pooling path (the caller then L2-normalizes it; plan §3.3).
// #10 sparse_head: per token t: w = relu(dot(hidden[t, :], W_s) + b_s) with the
//                  sparse_linear weight pre-transposed to a plain [H] vector.
//                  One scalar per token; host-side compaction of the nonzero
//                  (token_id, weight) pairs happens after readback (Phase A
//                  does not compact on device -- plan §4 #10 note; the opt
//                  plan §3.4 P1 sparse_head_compact below adds the device
//                  side, selected by the executor's EMBED_P1_SPARSE_COMPACT).
// #10 colbert_proj: per token row: y[t, :] = hidden[t, :] @ W_c + b_c with the
//                  colbert_linear weight pre-transposed to row-major [H, N]
//                  (coalesced along N, plan §3.3 packing rule); optional
//                  runtime flag normalizes each row to unit L2 norm. The
//                  caller drops the special tokens (<s>/</s>) host-side.
//
// All dimensions (H, N, S) are runtime kernel args so tests can use small
// synthetic sizes; production bge-m3: H = N = 1024.
//
// NOTE: these head projections carry their own simple dot/matmul. Phase B
// (executor) may later route them through the shared GEMM kernel (#3) with
// epilogue flags — the kernel contracts below stay valid either way.
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Thread block width of the head kernels (plan §4: block 256). Block-per-token
// kernels dispatch rows * kHeadsBlock threads; slice_cls dispatches H threads.
inline constexpr uint kHeadsBlock = 256u;

// slice_cls: out[0..H) = hidden[0..H) (row 0 of a [rows, H] matrix).
// Launch: 1D, block 256, dispatch(H) = total threads.
using SliceClsKernel = compute::Kernel1D<void(compute::Buffer<float> /*hidden*/,
                                              compute::Buffer<float> /*cls_out*/,
                                              uint32_t /*H*/)>;

[[nodiscard]] SliceClsKernel make_slice_cls_kernel() noexcept;

// sparse_head: out[t] = relu(sum_h X[t, h] * W_s[h] + b_s), t in [0, rows).
// Launch: 1D over rows (one block per token), block 256, dispatch(rows * 256)
// = total threads. X: [rows, H]; W_s: [H] (transposed 1xH weight row); out:
// [rows]. Host compacts the (token_id, weight) pairs after readback.
using SparseHeadKernel = compute::Kernel1D<void(compute::Buffer<float> /*X*/,
                                                compute::Buffer<float> /*W_s*/,
                                                float /*b_s*/,
                                                compute::Buffer<float> /*out*/,
                                                uint32_t /*H*/)>;

[[nodiscard]] SparseHeadKernel make_sparse_head_kernel() noexcept;

// colbert_proj: out[t, n] = sum_h X[t, h] * W_c[h, n] + b_c[n], n in [0, N).
// If normalize != 0, each row is additionally divided by max(||row||_2, 1e-12)
// (torch F.normalize semantics). Launch: 1D over rows (one block per token),
// block 256, dispatch(rows * 256) = total threads. X: [rows, H];
// W_c: [H, N] row-major (pre-transposed); b_c: [N]; out: [rows, N].
using ColbertProjKernel = compute::Kernel1D<void(compute::Buffer<float> /*X*/,
                                                 compute::Buffer<float> /*W_c*/,
                                                 compute::Buffer<float> /*b_c*/,
                                                 compute::Buffer<float> /*out*/,
                                                 uint32_t /*H*/,
                                                 uint32_t /*N*/,
                                                 uint32_t /*normalize*/)>;

[[nodiscard]] ColbertProjKernel make_colbert_proj_kernel() noexcept;

// ---------------------------------------------------------------------------
// P1 sparse-head readback compaction (opt plan §3.4 P1; the executor selects
// these behind EMBED_P1_SPARSE_COMPACT=2, "pairs" mode; the public wire stays
// the caller's [B*S] float span -- it is re-expanded host-side from the
// compact stream in the list's completion callback, so the scheduler and all
// parity gates consume byte-identical values on the positions they read).
//
// sparse_head_compact: computes y[t] = relu(dot(X[t,:], W_s) + b_s) with the
// EXACT same block-reduction order as sparse_head (bit-identical values), then
// appends the pair (t, y) -- token row index + weight -- for every row with
// y > 0 AND mask[t] != 0 (padded rows carry no contract value; the scheduler
// never reads them, and dropping them bounds the stream by the real-token
// count the host can compute at record time). The append uses block-local
// privatization -> one global atomic per block (lc_optimize §4.3), so the
// pair order is run-arbitrary -- expansion is an order-free scatter by t, and
// the expanded array is bit-identical regardless. Emissions <= sum(mask != 0)
// <= rows: the pair buffers sized [rows] can never overflow. counter (1
// uint, total emitted) must be zeroed before each dispatch -- see
// make_sparse_count_reset_kernel.
// Launch: 1D over rows (one block per token), block 256,
// dispatch(rows * 256) -- same grid as sparse_head (rows <= 16383 << 65535).
using SparseHeadCompactKernel = compute::Kernel1D<void(compute::Buffer<float> /*X*/,
                                                       compute::Buffer<float> /*W_s*/,
                                                       float /*b_s*/,
                                                       compute::Buffer<float> /*mask*/,
                                                       compute::Buffer<uint32_t> /*pair_ids*/,
                                                       compute::Buffer<float> /*pair_w*/,
                                                       compute::Buffer<uint32_t> /*counter*/,
                                                       uint32_t /*H*/)>;

[[nodiscard]] SparseHeadCompactKernel make_sparse_head_compact_kernel() noexcept;

// Zeroes the 1-uint pair-stream counter (32-thread block, thread 0 writes;
// recorded BEFORE sparse_head_compact in the same CommandList).
using SparseCountResetKernel = compute::Kernel1D<void(compute::Buffer<uint32_t> /*counter*/)>;

[[nodiscard]] SparseCountResetKernel make_sparse_count_reset_kernel() noexcept;

}// namespace luisa::embed
