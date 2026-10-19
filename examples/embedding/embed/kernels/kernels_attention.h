// bge-m3 attention kernels interface (plan §4 kernels #4-#6 naive, M4 flash).
// PARENT-OWNED CONTRACT: B2 implements kernels_attention.cpp against this
// header; B1 (executor) codes both call paths against it. Do not change
// signatures without updating both sides.
//
// LAYOUT CONTRACT (matches the packed fused-QKV weight "layer{i}.qkv"
// [K=1024, N=3072] and the GEMM epilogue convention):
//   qkv  : float[B*S, 3072] row-major, one row per token, B*S rows.
//          Per token row: [ Q(1024) | K(1024) | V(1024) ] where for head h in
//          [0, num_heads), channel d in [0, head_dim):
//            Q[h*head_dim + d], K[1024? NO -- use offsets below]
//          Precisely: Q_h,d = row[h*head_dim + d]
//                     K_h,d = row[num_heads*head_dim + h*head_dim + d]
//                     V_h,d = row[2*num_heads*head_dim + h*head_dim + d]
//          (with bge-m3 dims: num_heads=16, head_dim=64, stride 1024.)
//   mask : float[B*S], 1.0 = real token, 0.0 = pad. Additive mask bias on a
//          logit is (1 - mask[key]) * neg_large; query-side pad rows produce
//          garbage that every consumer masks or drops, so they are never
//          cleaned up (same contract as kernel #1's padded rows).
//   scale: softmax scale, bge-m3 uses 1/sqrt(head_dim) = 1/8.
//
// NAIVE PATH (Milestone M3, S <= 1024): three kernels over an explicit
// [B*num_heads, S, S] logits workspace (fp32; 64 MiB at B=1,S=1024).
//   #4 attention_scores -> #5 softmax_row (in place) -> #6 attention_pv.
// FLASH PATH (Milestone M4, any S): single fused kernel, no S*S workspace
// (<= ~25 KiB tile scratch; the tuned default variant uses 8.3 KiB of shared per
// block), online softmax with running max/sum rescale, pure scalar ALU (adapted
// from examples/ml create_mha_online_attention_kernel scalar/Cooperative=false
// path; document the adaptation in the .cpp header). The kernel geometry (query
// rows per block, key rows per streamed K/V tile, and where the per-thread row
// state lives) is selected once per process from a variant table in
// kernels_attention.cpp: env LUISA_EMBED_FLASH_CONFIG=<name> (or the field-wise
// LUISA_EMBED_FLASH_BLOCK / _TILE / _REG), with legacy_b32_t8 as the kill switch
// back to the shipped M4 emission. flash_attention_dispatch() reads the same
// resolution, so host geometry and kernel geometry can never disagree.

#pragma once

#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

inline constexpr uint kAttentionBlock = 256u;

// #4: logits[(b*num_heads + h)*S*S + qi*S + ki] =
//       dot(Q(b,qi,h,:), K(b,ki,h,:)) * scale + (1 - mask[b*S+ki]) * neg_large
// Launch: 2D tiles over (qi, ki); see kernels_attention.cpp for the exact
// dispatch helper attention_scores_dispatch(S) (total-thread semantics).
using AttentionScoresKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qkv*/,   // [B*S, 3*num_heads*head_dim]
    compute::Buffer<float> /*mask*/,  // [B*S]
    compute::Buffer<float> /*logits*/,// [B*num_heads, S, S]
    uint32_t /*S*/, uint32_t /*num_heads*/, uint32_t /*head_dim*/,
    float /*scale*/, float /*neg_large*/)>;
[[nodiscard]] AttentionScoresKernel make_attention_scores_kernel() noexcept;

// #5: in-place softmax over each logits row (B*num_heads*S rows of length S).
// Launch: one block per row, dispatch = rows * kAttentionBlock total threads.
using SoftmaxRowKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*logits*/, uint32_t /*S*/)>;
[[nodiscard]] SoftmaxRowKernel make_softmax_row_kernel() noexcept;

// #6: out[b*S + qi, h*head_dim + d] =
//       sum_ki logits[(b*num_heads+h)*S*S + qi*S + ki] * V(b,ki,h,d)
// out has the same [B*S, num_heads*head_dim] layout as one Q/K/V section,
// ready for the "layer{i}.attn.out" GEMM.
using AttentionPvKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qkv*/,   // [B*S, 3*num_heads*head_dim] (V only)
    compute::Buffer<float> /*logits*/,// [B*num_heads, S, S] (softmaxed)
    compute::Buffer<float> /*out*/,   // [B*S, num_heads*head_dim]
    uint32_t /*S*/, uint32_t /*num_heads*/, uint32_t /*head_dim*/)>;
[[nodiscard]] AttentionPvKernel make_attention_pv_kernel() noexcept;

// ---------------------------------------------------------------------------
// VEC4 TWIN VARIANTS (tensor_optimize.md §4.3 "naive attention / flash
// attention" — P2 row). The naive QK/PV kernels above are the bit-level
// regression oracle and stay the shipped/executor path; these are the float4
// experiments, with IDENTICAL buffer contracts and dispatch geometry so a
// caller can swap one in per launch. Only the arithmetic changes:
//   * QK: the 16-step scalar shared-tile dot becomes 4 float4 dot() chunks
//     (the stage loop still covers head_dim in ceil(head_dim/16) sub-tiles of
//     exactly 16 floats, zero-filled by the predicated staging, so a
//     non-multiple-of-4 head_dim is correct by construction — the ragged
//     channels are the zero lanes; no remainder loop exists, which also keeps
//     every sub-tile's per-thread dot order identical across dt).
//   * PV: one thread computes FOUR adjacent head channels with a float4
//     accumulator and a Shared<float4> V tile; head_dim % 4 == 0 is a host
//     contract (LUISA_ASSERT), not a tail case.
// Summation order differs from the legacy twins (chunked dot), so acceptance
// uses the existing numerical tolerance (plan §8.2(3)); the legacy kernels
// remain the bit-parity oracle.
// ---------------------------------------------------------------------------

// #4v: identical math/layout contract to #4 (see the dot formula above), float4
// inner product. Launch geometry == attention_scores_vec4_dispatch below.
using AttentionScoresVec4Kernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qkv*/,    // [B*S, 3*num_heads*head_dim]
    compute::Buffer<float> /*mask*/,   // [B*S]
    compute::Buffer<float> /*logits*/, // [B*num_heads, S, S]
    uint32_t /*S*/, uint32_t /*num_heads*/, uint32_t /*head_dim*/,
    float /*scale*/, float /*neg_large*/)>;
[[nodiscard]] AttentionScoresVec4Kernel make_attention_scores_vec4_kernel() noexcept;

// #6v: identical math/layout contract to #6, float4 accumulator (4 adjacent
// head channels per thread). HOST CONTRACT: head_dim % 4 == 0 (asserted in the
// factory); ragged head_dim stays on the legacy #6.
using AttentionPvVec4Kernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qkv*/,    // [B*S, 3*num_heads*head_dim] (V only)
    compute::Buffer<float> /*logits*/, // [B*num_heads, S, S] (softmaxed)
    compute::Buffer<float> /*out*/,    // [B*S, num_heads*head_dim]
    uint32_t /*S*/, uint32_t /*num_heads*/, uint32_t /*head_dim*/)>;
[[nodiscard]] AttentionPvVec4Kernel make_attention_pv_vec4_kernel() noexcept;

// M4 fused flash attention (replaces #4-#6). Same logical inputs/outputs as
// the naive trio, but no logits buffer; `scratch` is per-(b,h) online-softmax
// rescale state sized scratch_floats(B, S, num_heads) -- see .cpp header for
// the formula; B1 allocates it in the attention workspace bucket.
using FlashAttentionKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qkv*/,
    compute::Buffer<float> /*mask*/,
    compute::Buffer<float> /*out*/,    // [B*S, num_heads*head_dim]
    compute::Buffer<float> /*scratch*/,// see .cpp header
    uint32_t /*B*/, uint32_t /*S*/,
    uint32_t /*num_heads*/, uint32_t /*head_dim*/,
    float /*scale*/, float /*neg_large*/)>;
[[nodiscard]] FlashAttentionKernel make_flash_attention_kernel() noexcept;

// ---------------------------------------------------------------------------
// Launch-geometry helpers (defined in kernels_attention.cpp, external linkage;
// declared here so the executor and tests share one source of truth -- the
// kernels CANNOT derive these bounds from buffer sizes on dx, where
// CallOp::BUFFER_SIZE lowers to 0xFFFFFFFF without debug info).
// dispatch() takes TOTAL THREADS: for the uint2 returns, dispatch(x, y).
// ---------------------------------------------------------------------------

// #4: 2D grid (x = S-tiles * kAttentionBlock threads, y = B*num_heads*S-tiles).
[[nodiscard]] uint2 attention_scores_dispatch(uint32_t batch, uint32_t seq_len,
                                              uint32_t num_heads) noexcept;
// #6: 2D grid (x = head_dim-tiles * kAttentionBlock, y = B*num_heads*S-tiles).
[[nodiscard]] uint2 attention_pv_dispatch(uint32_t batch, uint32_t seq_len,
                                          uint32_t num_heads, uint32_t head_dim) noexcept;
// #5: 1D total threads = B*num_heads*S*kAttentionBlock (one block per logits row).
[[nodiscard]] uint32_t softmax_row_dispatch(uint32_t batch, uint32_t seq_len,
                                            uint32_t num_heads) noexcept;
// M4: 2D grid (x = query-row-tiles * BLOCK, y = B*num_heads), where BLOCK is the
// query rows per block of the runtime-selected flash variant (32 on the shipped
// M4 baseline / legacy kill switch, 128 on the tuned default -- see
// kFlashVariants in kernels_attention.cpp; x is always a multiple of the block
// size and the kernel derives the flattened block index from dispatch_id()/
// dispatch_size(), so the same emission is correct for a plain 2D launch and for a
// host-flattened one).
[[nodiscard]] uint2 flash_attention_dispatch(uint32_t batch, uint32_t seq_len,
                                             uint32_t num_heads) noexcept;
// M4 scratch float count (currently 1: all online-softmax state is
// register/shared-resident -- see the .cpp "Flash `scratch` contract" block).
[[nodiscard]] uint32_t flash_attention_scratch_floats(uint32_t batch, uint32_t seq_len,
                                                      uint32_t num_heads) noexcept;

}// namespace luisa::embed
