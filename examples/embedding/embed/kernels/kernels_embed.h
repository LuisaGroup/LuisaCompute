// bge-m3 embedding engine -- embedding-block kernel (plan §4 kernel #1).
//
// embed_lookup fuses: token-id Gather of the word-embedding row, absolute
// position Gather of the position-embedding row, their Add, and the embedding
// LayerNorm (gamma/beta/eps). It replaces the two ONNX `Gather` ×2 + `Add` +
// decomposed `LayerNormalization` at the head of the graph (plan §0: RoBERTa
// absolute position embeddings -- positions are 0..S-1, never an input).
//
// Device-free Kernel1D factory per CONVENTIONS.md; the caller (Phase B
// executor / tests) compiles it with device.compile(...). All dimensions are
// runtime kernel args so tests can use small synthetic sizes; production
// bge-m3 values are hidden_size H=1024, vocab 250002, max positions 8194.
#pragma once
#include <luisa/dsl/syntax.h>

namespace luisa::embed {

// Thread block width of the embedding-block kernels (plan §4: block 256).
// Host dispatch counts are rows * kEmbedBlock threads.
inline constexpr uint kEmbedBlock = 256u;

// embed_lookup: hidden[b*S + s, :] = LayerNorm(word_emb[ids[b*S+s], :] +
//                                                pos_emb[s, :]; gamma, beta, eps)
// Launch shape (plan §4 #1): 1D grid over B*S token rows, block 256; dispatch
// count is TOTAL THREADS (= rows * 256). One workgroup per token row; the LN
// mean/var use a two-level (warp -> shared) block reduction.
// Buffers: word_emb [V*H], pos_emb [P*H] (only rows 0..S-1 touched),
//          ln_gamma [H], ln_beta [H], ids [B*S] (uint32), out hidden [B*S*H].
// Scalars: S (sequence length), H (hidden size), eps (layer_norm_eps),
// pos_offset (position table row of token 0; bge-m3 uses XLM-R's +2 offset --
// the ONNX graph builds positions as cumsum(mask)*mask + 1, see pack_model.cpp
// header note; pads may keep the uniform offset because every downstream
// consumer masks or drops padded rows).
// Shared-memory row cache covers H <= 2048; larger H uses a multi-pass
// re-gather path (still correct, only slower -- never taken by bge-m3).
using EmbedLookupKernel = compute::Kernel1D<void(compute::Buffer<float> /*word_emb*/,
                                                 compute::Buffer<float> /*pos_emb*/,
                                                 compute::Buffer<uint32_t> /*ids*/,
                                                 compute::Buffer<float> /*ln_gamma*/,
                                                 compute::Buffer<float> /*ln_beta*/,
                                                 compute::Buffer<float> /*hidden_out*/,
                                                   uint32_t /*S*/,
                                                   uint32_t /*H*/,
                                                   float /*eps*/,
                                                   uint32_t /*pos_offset*/)>;

[[nodiscard]] EmbedLookupKernel make_embed_lookup_kernel() noexcept;

}// namespace luisa::embed
