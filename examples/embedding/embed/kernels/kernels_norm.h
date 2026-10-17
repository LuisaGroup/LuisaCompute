// bge-m3 embedding engine -- normalization kernels (plan §4 kernels #2 and #9):
// layernorm and l2_normalize.
//
// layernorm replaces the 48 decomposed LayerNormalization clusters of the
// ONNX export (ReduceMean/Sub/Pow/ReduceMean/Div/Sqrt/Mul/Add, plan §0/§4 row
// #2): one thread block per (B*S) row, block 256, two-pass fp32 mean/variance
// (biased variance, matching ONNX ReduceMean semantics), eps=1e-5 for bge-m3
// (config layer_norm_eps), y = (x - mu) / sqrt(var + eps) * gamma + beta.
//
// l2_normalize replaces the ReduceL2/Div chain of the sentence-pooling output
// path (plan §4 row #9): one block per row, y = x / max(||x||_2, 1e-12) --
// the torch F.normalize clamp, so an all-zero row yields zeros instead of NaN.
//
// Device-free Kernel1D factories per CONVENTIONS.md; the caller compiles them
// with device.compile(...) and caches the Shader. Row length H (and eps) are
// runtime kernel args, so one compilation serves every shape. Launch: one block
// per row -> dispatch(rows * kNormBlock) TOTAL THREADS (lc_optimize trap #1).
// Rows come from a flattened block index, so a host that must dodge the DX
// 65535-group cap can dispatch a 2D grid without changing the kernels
// (lc_optimize trap #2, same convention as kernels_embed.cpp).
// Ragged rows are safe for any H: H < block size (idle threads contribute 0)
// and H > block size (strided $while loop) are both covered.
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Thread block width of the normalization kernels (plan §4: block 256).
inline constexpr uint kNormBlock = 256u;

// layernorm: y[row] = (x[row] - mean) * (1 / sqrt(var + eps)) * gamma + beta,
// two-pass mean/var over the row. Buffers: x [rows*H], y [rows*H],
// gamma [H], beta [H]. Scalars: H (row length), eps (layer_norm_eps).
using LayerNormKernel = compute::Kernel1D<void(compute::Buffer<float> /*x*/,
                                               compute::Buffer<float> /*y*/,
                                               compute::Buffer<float> /*gamma*/,
                                               compute::Buffer<float> /*beta*/,
                                               uint32_t /*H*/,
                                               float /*eps*/)>;

[[nodiscard]] LayerNormKernel make_layernorm_kernel() noexcept;

// l2_normalize: y[row] = x[row] / max(||x[row]||_2, 1e-12). Buffers: x [rows*H],
// y [rows*H]. Scalar: H (row length).
using L2NormalizeKernel = compute::Kernel1D<void(compute::Buffer<float> /*x*/,
                                                 compute::Buffer<float> /*y*/,
                                                 uint32_t /*H*/)>;

[[nodiscard]] L2NormalizeKernel make_l2_normalize_kernel() noexcept;

// dense_pool (plan §3.3 N1(b)): the fused replacement for the per-batch-row
// slice_cls (#8) + l2_normalize (#9) pair at the head of the dense-pooling tail.
// bge-m3 has NO pooler final-LayerNorm after layer 24 -- layer 24's output is
// already post-LN2 (executor.h record_encode: LN2 writes back into m_h0 and
// trace slot i+1 reads m_h0; golden_layers.bin slot 24 == m_h0), so the dense
// pooling path is exactly CLS-slice + L2-normalize. This kernel performs both in
// one launch: for batch row b it L2-normalizes hidden[b*S .. b*S+1) (the CLS row
// of the [B*S, H] final hidden state) straight out of `hidden` into out[b] (a
// [B, H] dense buffer), removing the m_cls intermediate round-trip and folding 2B
// dispatches into 1.
//
// Numerics are BIT-IDENTICAL to running slice_cls then l2_normalize per row: the
// L2 reduction uses the same kNormBlock-wide shared-memory halving tree, the same
// strided per-thread accumulation order over the CLS row, and the same
// y = x / max(||x||_2, 1e-12) formula. Verified 0 differing floats over B in
// {1,16}, H=1024 on dx + vk + fallback (scratch harness). Per-block flattening is
// by BATCH (grid.y carries batch groups so a >65535-batch host could dodge the DX
// cap the same way; B <= 16383 in practice, so 1D is always safe).
// Launch: one block per batch row, block kNormBlock; dispatch(B * kNormBlock)
// TOTAL threads. Buffers: hidden [B*S*H], out [B*H]. Scalars: S (seq len), H.
using DensePoolKernel = compute::Kernel1D<void(compute::Buffer<float> /*hidden*/,
                                               compute::Buffer<float> /*out*/,
                                               uint32_t /*S*/,
                                               uint32_t /*H*/)>;

[[nodiscard]] DensePoolKernel make_dense_pool_kernel() noexcept;

}// namespace luisa::embed
