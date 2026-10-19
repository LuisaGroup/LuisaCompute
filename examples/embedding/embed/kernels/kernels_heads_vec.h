// bge-m3 embedding engine -- retrieval-head vec4 kernels (tensor_optimize.md
// section 4.2 sparse head + section 4.3 ColBERT Float4 accumulator + FMA).
//
// float4-width variants of the two compute heads in kernels_heads.h (plan
// section 4 kernel #10). They cover the same ONNX-op mapping as the legacy
// scalar kernels, only at vector width (ragged sizes stay on the legacy scalar
// kernels -- see the host contract below):
//
// sparse_head_vec4  (kernels_heads.h sparse_head): per token t:
//   out[t] = relu(dot(X[t, :], W_s) + b_s) -- the sparse_linear head (ONNX
//   MatMul[1024x1] + Add + Relu). The 1024-wide dot product runs as a
//   lane-wise float4 partial accumulate (NO dot(): the vec4_io horizontal
//   order is pinned to ((p.x+p.y)+(p.z+p.w)) and the block tree dominates),
//   folded through the shared-memory halving-tree block reduction, then the
//   EXACT legacy strict ReLU (v > 0) and the single thread-0 scalar store.
//
// colbert_proj_vec4 (kernels_heads.h colbert_proj): per token row t:
//   out[t, :] = X[t, :] @ W_c + b_c  with the optional runtime L2-normalize.
//   Each thread now owns FOUR adjacent N outputs as one float4 (plan section
//   4.3 item 1); the inner loop broadcasts ONE float of x against one float4
//   of weight with FMA (item 2); bias, output and the pre-normalization sum of
//   squares are all float4 (item 3). W_c is row-major [H, N] (pre-transposed),
//   so the per-vec4-column weight read W_c4.read(h * N4 + c) is coalesced along
//   the N lanes (plan section 3.3 packing rule). The X row is cached in shared
//   memory as float4 (2048-float coverage, mirroring the legacy kMaxCachedRow
//   dual path). The fused L2 normalization mirrors the legacy exactly: lane-wise
//   sum of squares -> block_sum -> sqrt -> explicit 1e-12 clamp -> second
//   strided divide pass over the same columns this thread owns.
//
// HOST CONTRACT (caller must guarantee; the kernels never query buffer sizes):
// - sparse_head_vec4: hidden size H is a multiple of 4 (H % 4 == 0).
// - colbert_proj_vec4: H % 4 == 0 AND output width N % 4 == 0.
// - X, W_s, W_c, b_c and out are 16-byte-aligned buffers viewed as float4 (byte
//   length divisible by 16; per-row/per-column offsets land on float4
//   boundaries, which H % 4 == 0 / N % 4 == 0 ensures). The sparse head's scalar
//   output (one value per token) and the token-id column stay plain float.
// - Ragged H / N (any size not a multiple of 4) is NOT supported by these
//   variants and stays on the legacy scalar kernels in kernels_heads.{h,cpp}.
//
// Device-free Kernel1D factories per the project conventions; the caller
// (Phase B executor / tests) compiles them with device.compile(...). All
// dimensions (H, N) are runtime kernel args so tests can use small synthetic
// sizes; production bge-m3 uses H = N = 1024 (H4 = N4 = 256, fits the shared
// row cache). dispatch() counts TOTAL THREADS (rows * kHeadsVecBlock); the
// flattened block id is valid for a host 2D grid dodging the DX 65535-group cap
// (lc_optimize trap #2). The block reduction is the shared-memory halving tree
// (fallback-backend safe; warp intrinsics abort on the CPU fallback).
//
// These variants change the in-thread partial grouping relative to the legacy
// scalar kernels, so acceptance uses the existing numerical tolerance; the
// legacy kernels remain the bit-level regression oracle.
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Thread block width of the retrieval-head vec4 kernels (same 256 as legacy).
// Host dispatch counts are rows * kHeadsVecBlock total threads.
inline constexpr uint kHeadsVecBlock = 256u;

// Maximum cached X-row width, in float4 lanes, for colbert_proj_vec4. 512
// float4 == 2048 floats == the legacy kMaxCachedRow scalar coverage (4 B *
// (kHeadsVecBlock + 4*512) <= 32 KiB shared floor). H4 > this cap takes the
// global re-read fallback (still correct, only slower -- never taken by bge-m3).
inline constexpr uint kHeadsVecMaxCachedRow4 = 512u;// = 2048 floats / 4

// sparse_head_vec4: out[t] = relu(sum_h X[t, h] * W_s[h] + b_s), t in [0, rows).
// Launch: 1D over rows (one block per token), block 256, dispatch(rows * 256) =
// total threads. X: [rows, H] viewed as float4 (H % 4 == 0); W_s: [H] viewed as
// float4 (the transposed 1xH weight row); out: [rows] scalar float. Host
// compaction of the (token_id, weight) pairs after readback is unchanged.
using SparseHeadVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*X4*/,
                                                    compute::Buffer<float4> /*W_s4*/,
                                                    float /*b_s*/,
                                                    compute::Buffer<float> /*out*/,
                                                    uint32_t /*H*/)>;

[[nodiscard]] SparseHeadVec4Kernel make_sparse_head_vec4_kernel() noexcept;

// colbert_proj_vec4: out[t, n] = sum_h X[t, h] * W_c[h, n] + b_c[n], n in [0, N).
// If normalize != 0, each row is additionally divided by max(||row||_2, 1e-12)
// (torch F.normalize semantics), fused into the same launch. Launch: 1D over
// rows (one block per token), block 256, dispatch(rows * 256) = total threads.
// Each thread owns four adjacent N outputs as one float4 (N % 4 == 0). X:
// [rows, H] float4 (H % 4 == 0); W_c: [H, N] float4 row-major (pre-transposed,
// coalesced along N); b_c: [N] float4; out: [rows, N] float4.
using ColbertProjVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*X4*/,
                                                     compute::Buffer<float4> /*W_c4*/,
                                                     compute::Buffer<float4> /*b_c4*/,
                                                     compute::Buffer<float4> /*out4*/,
                                                     uint32_t /*H*/,
                                                     uint32_t /*N*/,
                                                     uint32_t /*normalize*/)>;

[[nodiscard]] ColbertProjVec4Kernel make_colbert_proj_vec4_kernel() noexcept;

}// namespace luisa::embed
