// bge-m3 embedding engine -- retrieval-head kernels (plan §4 kernels #8, #10).
//
// #8 slice_cls:    copies row 0 (the <s>/CLS token row) out of the final
//                  hidden state [B*S, H] — the first Gather/Slice of the dense
//                  pooling path (the caller then L2-normalizes it; plan §3.3).
// #10 sparse_head: per token t: w = relu(dot(hidden[t, :], W_s) + b_s) with the
//                  sparse_linear weight pre-transposed to a plain [H] vector.
//                  One scalar per token; host-side compaction of the nonzero
//                  (token_id, weight) pairs happens after readback (Phase A
//                  does not compact on device — plan §4 #10 note).
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

}// namespace luisa::embed
