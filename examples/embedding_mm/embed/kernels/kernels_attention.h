#pragma once
// embedding_mm engine -- Grouped-Query Attention (GQA) kernels with sliding window.
//
// Gemma 2 GQA structure:
// - Sliding layers: num_q_heads = 4, num_kv_heads = 2, head_dim = 256, sliding_window = 512
// - Full layers:    num_q_heads = 4, num_kv_heads = 1, head_dim = 512, full sequence
//
// Trio of kernels for sequence length S <= 1024:
// 1. gqa_scores: compute dot(Q, K) * scale + masks
// 2. softmax_row: in-place numerically stable softmax
// 3. gqa_pv: multiply softmax weights by V

#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kAttentionBlock = 256u;

// #1: QK dot scores with GQA mapping and sliding window mask
// q: [B*S, num_q_heads * head_dim]
// k: [B*S, num_kv_heads * head_dim]
// mask: [B*S] (1.0 = valid token, 0.0 = pad)
// logits: [B*num_q_heads, S, S]
using GQAScoresKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*q*/,
    compute::Buffer<float> /*k*/,
    compute::Buffer<float> /*mask*/,
    compute::Buffer<float> /*logits*/,
    uint32_t /*S*/,
    uint32_t /*num_q_heads*/,
    uint32_t /*num_kv_heads*/,
    uint32_t /*head_dim*/,
    float /*scale*/,
    uint32_t /*is_sliding*/,
    uint32_t /*sliding_window*/,
    float /*neg_large*/)>;

[[nodiscard]] GQAScoresKernel make_gqa_scores_kernel() noexcept;

// #2: in-place softmax over each row of logits
// logits: [B*num_q_heads * S, S]
using SoftmaxRowKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*logits*/,
    uint32_t /*S*/)>;

[[nodiscard]] SoftmaxRowKernel make_softmax_row_kernel() noexcept;

// #3: PV weighted sum
// v: [B*S, num_kv_heads * head_dim]
// logits: [B*num_q_heads, S, S] (post-softmax)
// out: [B*S, num_q_heads * head_dim]
using GQAPvKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*v*/,
    compute::Buffer<float> /*logits*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*S*/,
    uint32_t /*num_q_heads*/,
    uint32_t /*num_kv_heads*/,
    uint32_t /*head_dim*/)>;

[[nodiscard]] GQAPvKernel make_gqa_pv_kernel() noexcept;

}// namespace luisa::embed_mm
