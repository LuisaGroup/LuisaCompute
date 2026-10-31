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

//--- fused residual_add + layernorm (opt plan C3) ---------------------------
// The per-layer `residual Add (#7)` -> `LayerNorm (#2)` pair runs twice per
// encoder layer as 2 dispatches that together traverse the row plane SEVEN times
// (residual_add: 2 reads + 1 write; layernorm: 3 reads + 1 write, because its
// mean/var/normalize passes each re-read x from global -- and x is the plane the
// residual just wrote). This kernel does it in ONE launch and THREE traversals
// (2 reads for the residual operands, 1 write for the result): it computes the
// fp32 residual r = a + b once, caches the whole row in Shared<float>, and runs
// the SAME three passes as make_layernorm_kernel() over the cache. It also drops
// the residual plane round-trip entirely, so the intermediate never leaves the
// SM.
//
// NUMERICALLY EQUIVALENT to `make_residual_add_kernel()` followed by
// `make_layernorm_kernel()`, within a focused 1e-5 ABSOLUTE tolerance -- it
// preserves the pair's REDUCTION ORDER exactly: the residual is the same
// IEEE-754 fp32 add of the same two operands (caching the value on chip instead
// of in global memory does not round it again), the per-thread strided
// accumulation order is unchanged, and the shared-memory halving-tree reduction
// order and the final expression order `((r - mean) * inv) * gamma + beta` are
// verbatim copies of the layernorm kernel's; no warp collectives are involved
// either.
//
// That is NOT a bit-identity claim, and this header does not make one: the same
// op sequence emitted as one kernel instead of two lets a backend re-emit the
// rounding differently (measured on dx: ~1-ULP differences, e.g. pair
// -0.13910183 vs fused -0.13910185), so bit-exact agreement is not a property
// of the fusion on every backend. The gate is therefore
// test_embed_kernel_layernorm's explicit absolute tolerance of 1e-5 (see
// check_residual_layernorm()), which pins the value that does hold: the fusion
// is a rounding-level substitution, never a numerical change.
//
// H <= kResidualNormMaxHidden is a HARD host contract (see that constant): the
// shared cache is sized at compile time, so a longer row must stay on the legacy
// pair -- residual_layernorm_supported() is the host-side predicate the executor
// uses to pick between them. Out may alias `a` (or `b`): the whole residual row
// is cached before the first write to `out`, and the pass-1/pass-2 block
// reductions are barrier-separated, so no thread can write `out` while another
// still needs to read `a`/`b`.
//
// Launch: one block per row, block kNormBlock; dispatch(rows * kNormBlock)
// TOTAL THREADS (lc_optimize trap #1 -- dispatch() counts threads, not blocks).
// Rows come from a flattened block index, so a host that must dodge the DX
// 65535-group cap can dispatch a 2D grid without changing the kernel
// (lc_optimize trap #2, same convention as make_layernorm_kernel()).
//
// out[row*H + h] = LN(a[row*H + h] + b[row*H + h]) with the row length H and
// eps as runtime args, so one compilation serves every shape in the contract.
// Buffers: a [rows*H], b [rows*H], out [rows*H], gamma [H], beta [H].
using ResidualLayerNormKernel = compute::Kernel1D<void(compute::Buffer<float> /*a*/,
                                                       compute::Buffer<float> /*b*/,
                                                       compute::Buffer<float> /*out*/,
                                                       compute::Buffer<float> /*gamma*/,
                                                       compute::Buffer<float> /*beta*/,
                                                       uint32_t /*H*/,
                                                       float /*eps*/)>;

// Residual-row cache capacity of the fused kernel, in floats. bge-m3's
// hidden_size is 1024 with a single production package, so 4 KiB of shared
// memory covers every real shape; together with the 256-float reduction tree
// (1 KiB) the kernel requests 5 KiB per block, far inside the 32 KiB HLSL/DX
// floor (lc_optimize §4.7 rule 6). The bound is static because Shared<float> is
// sized at kernel-compile time, hence the host contract.
inline constexpr uint kResidualNormMaxHidden = 1024u;

// Host predicate: true iff the fused kernel is legal for row length h. The
// executor falls back to the legacy residual_add + layernorm pair otherwise
// (numerically equivalent within 1e-5 either way, so the choice is purely a
// matter of traffic).
[[nodiscard]] bool residual_layernorm_supported(uint32_t h) noexcept;

[[nodiscard]] ResidualLayerNormKernel make_residual_layernorm_kernel() noexcept;

// l2_normalize: y[row] = x[row] / max(||x[row]||_2, 1e-12). Buffers: x [rows*H],
// y [rows*H]. Scalar: H (row length).
using L2NormalizeKernel = compute::Kernel1D<void(compute::Buffer<float> /*x*/,
                                                 compute::Buffer<float> /*y*/,
                                                 uint32_t /*H*/)>;

[[nodiscard]] L2NormalizeKernel make_l2_normalize_kernel() noexcept;

// dense_pool (plan §3.3 N1(b)): the fused replacement for the per-batch-row
// slice_cls (#8) + l2_normalize (#9) pair at the head of the dense-pooling tail.
// bge-m3 has NO pooler final-LayerNorm after layer 24 -- layer 24's output is
// already post-LN2 (executor.h record_encode: LN2 writes back into _h0 and
// trace slot i+1 reads _h0; golden_layers.bin slot 24 == _h0), so the dense
// pooling path is exactly CLS-slice + L2-normalize. This kernel performs both in
// one launch: for batch row b it L2-normalizes hidden[b*S .. b*S+1) (the CLS row
// of the [B*S, H] final hidden state) straight out of `hidden` into out[b] (a
// [B, H] dense buffer), removing the _cls intermediate round-trip and folding 2B
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
