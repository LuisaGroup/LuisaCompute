// bge-m3 embedding engine -- embedding-block vec4 kernel (tensor_optimize.md
// section 4.2: embed lookup vec4 gather + LN).
//
// embed_lookup_vec4 is a float4-width variant of embed_lookup (kernels_embed.h,
// plan section 4 kernel #1). It fuses the same chain: token-id Gather of the
// word-embedding row, absolute position Gather of the position-embedding row,
// their Add, and the embedding LayerNorm (gamma/beta/eps). It replaces the two
// ONNX `Gather` x2 + `Add` + decomposed `LayerNormalization` at the head of the
// graph (plan section 0: RoBERTa absolute position embeddings -- positions are
// 0..S-1, never an input).
//
// This variant differs from the legacy kernel ONLY in vector width: every
// buffer is viewed as float4 and all per-element work advances four floats per
// access. The row/position indexing, the two-pass mean/biased-variance, the
// shared-memory row cache, the halving-tree block reduction (barriers outside
// the divergent branches; fallback-backend safe), the strided $while loops, and
// the pass-3 expression order ((x - mean) * inv) * gamma + beta are mirrored
// bit-faithfully from kernels_embed.cpp. Because the lane-internal horizontal
// summation order is pinned ((a.x+a.y)+(a.z+a.w), no dot()) the result differs
// from legacy only by the in-thread partial-sum grouping -- acceptance uses the
// existing numerical tolerance; the legacy kernel remains the bit-level oracle.
//
// HOST CONTRACT (caller must guarantee; the kernel never queries buffer sizes):
//   - hidden size H is a multiple of 4 (H % 4 == 0). Ragged H is NOT supported
//     by this variant and stays on the legacy scalar kernel.
//   - word_emb, pos_emb, gamma, beta and hidden_out are 16-byte aligned buffers
//     viewed as float4 (byte length divisible by 16; per-row offsets land on
//     float4 boundaries, which H % 4 == 0 ensures). ids stays a uint32 buffer.
//
// Device-free Kernel1D factory per CONVENTIONS.md; the caller (Phase B
// executor / tests) compiles it with device.compile(...). All dimensions are
// runtime kernel args so tests can use small synthetic sizes; production bge-m3
// values are hidden_size H=1024 (H4 = 256 float4 lanes, fits the row cache),
// vocab 250002, max positions 8194.
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Thread block width of the embedding-block vec4 kernel (same 256 as legacy).
// Host dispatch counts are rows * kEmbedVecBlock threads.
inline constexpr uint kEmbedVecBlock = 256u;

// Maximum hidden size, in float4 lanes, cached in shared memory. 512 float4 ==
// 2048 floats == the legacy kMaxCachedRow scalar coverage (4 B * (kEmbedVecBlock
// + 4*512) <= 32 KiB shared floor). H4 > this cap takes the re-gather fallback
// below (still correct, only slower -- never taken by bge-m3).
inline constexpr uint kEmbedVecMaxCachedRow4 = 512u;// = 2048 floats / 4

// embed_lookup_vec4: hidden[b*S + s, :] = LayerNorm(word_emb[ids[b*S+s], :] +
//                                                    pos_emb[s, :]; gamma, beta, eps)
// Launch shape (same as legacy, vec4 width): 1D grid over B*S token rows, block
// 256; dispatch count is TOTAL THREADS (= rows * 256). One workgroup per token
// row; the LN mean/var use a shared-memory halving-tree block reduction.
// Buffers (float4 views, H % 4 == 0): word_emb4 [V*H4], pos_emb4 [P*H4] (only
//          rows 0..S-1 touched), ln_gamma4 [H4], ln_beta4 [H4],
//          hidden_out4 [B*S*H4]; ids [B*S] (uint32).
// Scalars: S (sequence length), H (hidden size, H % 4 == 0), eps
// (layer_norm_eps), pos_offset (position table row of token 0; bge-m3 uses
// XLM-R's +2 offset -- the ONNX graph builds positions as cumsum(mask)*mask + 1,
// see pack_model.cpp header note; pads may keep the uniform offset because every
// downstream consumer masks or drops padded rows).
// Shared-memory row cache covers H4 <= 512 (i.e. H <= 2048); larger H4 uses a
// multi-pass re-gather path (still correct, only slower -- never taken by
// bge-m3).
using EmbedLookupVec4Kernel = compute::Kernel1D<void(compute::Buffer<float4> /*word_emb4*/,
                                                     compute::Buffer<uint32_t> /*ids*/,
                                                     compute::Buffer<float4> /*pos_emb4*/,
                                                     compute::Buffer<float4> /*ln_gamma4*/,
                                                     compute::Buffer<float4> /*ln_beta4*/,
                                                     compute::Buffer<float4> /*hidden_out4*/,
                                                     uint32_t /*S*/,
                                                     uint32_t /*H*/,
                                                     float /*eps*/,
                                                     uint32_t /*pos_offset*/)>;

[[nodiscard]] EmbedLookupVec4Kernel make_embed_lookup_vec4_kernel() noexcept;

}// namespace luisa::embed
