#pragma once
// embedding_mm engine -- Attention operators:
//   1. GQAScores: Grouped-Query / Multi-Head QK^T dot scores with sliding-window & padding masks
//   2. SoftmaxRow: In-place numerically stable shared-memory row-wise softmax
//   3. GQAPv: Attention weight * V contraction
//   4. AudioScaleQK: Conformer per-dimension softplus Q scaling & K scaling
//   5. AudioChunkedAttention: Conformer 12-query / 24-key chunked relative attention with tanh softcap

#include <cstdint>
#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kAttentionBlock = 256u;

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
    float /*neg_large*/,
    uint32_t /*row_offset*/)>;

[[nodiscard]] GQAScoresKernel make_gqa_scores_kernel() noexcept;

using SoftmaxRowKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*logits*/,
    uint32_t /*S*/)>;

[[nodiscard]] SoftmaxRowKernel make_softmax_row_kernel() noexcept;

using GQAPvKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*v*/,
    compute::Buffer<float> /*logits*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*S*/,
    uint32_t /*num_q_heads*/,
    uint32_t /*num_kv_heads*/,
    uint32_t /*head_dim*/,
    uint32_t /*row_offset*/)>;

[[nodiscard]] GQAPvKernel make_gqa_pv_kernel() noexcept;

using AudioScaleQKKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*q*/,
    compute::Buffer<float> /*k*/,
    compute::Buffer<float> /*per_dim_scale*/,
    uint32_t /*T*/,
    uint32_t /*num_heads*/,
    uint32_t /*head_dim*/,
    float /*q_scale*/,
    float /*k_scale*/)>;

[[nodiscard]] AudioScaleQKKernel make_audio_scale_qk_kernel() noexcept;

using AudioChunkedAttentionKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*q*/,
    compute::Buffer<float> /*k*/,
    compute::Buffer<float> /*v*/,
    compute::Buffer<float> /*rel_k*/,
    compute::Buffer<float> /*out_attn*/,
    uint32_t /*T*/,
    uint32_t /*num_heads*/,
    uint32_t /*head_dim*/,
    uint32_t /*chunk_size*/,
    float /*softcap*/)>;

[[nodiscard]] AudioChunkedAttentionKernel make_audio_chunked_attention_kernel() noexcept;

}// namespace luisa::embed_mm
