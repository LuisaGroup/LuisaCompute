#pragma once
// embedding_mm engine -- Vision Tower kernels for Gemma4VisionModel (~170M).
//
// Operations:
// 1. Scale pixel values: x = 2.0 * (raw_x - 0.5)
// 2. Add 2D patch position embeddings from position_embedding_table [2, 10240, 768]
// 3. 2D Axial RoPE: (theta = 100.0) applied to Q and K across 12 heads (head_dim = 64)
// 4. 2D Spatial Average Pooling: 3x3 patch grid -> soft token with sqrt(768) scaling
// 5. Scatter vision soft tokens into LLM sequence buffer

#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

// #1: Rescale raw [0, 1] pixels to [-1, 1]
using ScalePixelsKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*in_pixels*/,
    compute::Buffer<float> /*out_pixels*/,
    uint32_t /*total_elements*/)>;

[[nodiscard]] ScalePixelsKernel make_scale_pixels_kernel() noexcept;

// #2: Add 2D patch position embeddings from table [2, 10240, H]
// pos_ids: [B * max_patches * 2] (x, y)
// table: [2 * 10240 * H]
// hidden: [B * max_patches * H]
using AddPatchPosEmbedKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*hidden*/,
    compute::Buffer<float> /*table*/,
    compute::Buffer<int32_t> /*pos_ids*/,
    uint32_t /*H*/,
    uint32_t /*pos_table_size*/)>;

[[nodiscard]] AddPatchPosEmbedKernel make_add_patch_pos_embed_kernel() noexcept;

// #3: 2D Axial RoPE for Vision Attention
// qk: [B * max_patches, num_heads * head_dim] (num_heads = 12, head_dim = 64)
// pos_ids: [B * max_patches * 2]
using AxialRoPEKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qk_in*/,
    compute::Buffer<float> /*qk_out*/,
    compute::Buffer<int32_t> /*pos_ids*/,
    uint32_t /*num_heads*/,
    uint32_t /*head_dim*/,
    float /*rope_theta*/)>;

[[nodiscard]] AxialRoPEKernel make_axial_rope_kernel() noexcept;

// #4: 2D Spatial Pooling (3x3 grid of patches -> 1 soft token, scaled by sqrt(H))
// in_hidden: [B, max_patches, H] (H = 768)
// out_pooled: [B, num_soft_tokens, H] (num_soft_tokens = 256 for 48x48)
using VisionPoolKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*in_hidden*/,
    compute::Buffer<float> /*out_pooled*/,
    uint32_t /*grid_w_soft*/,
    uint32_t /*grid_h_soft*/,
    uint32_t /*patch_stride_w*/,
    uint32_t /*H*/,
    float /*scale_factor*/)>;

[[nodiscard]] VisionPoolKernel make_vision_pool_kernel() noexcept;

// #5: Scatter vision soft tokens into language model sequence buffer
// h0: [total_tokens * H] (H = 512)
// soft_tokens: [num_soft_tokens * H]
using ScatterSoftTokensKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*h0*/,
    compute::Buffer<float> /*soft_tokens*/,
    uint32_t /*start_token_idx*/,
    uint32_t /*num_soft_tokens*/,
    uint32_t /*H*/)>;

[[nodiscard]] ScatterSoftTokensKernel make_scatter_soft_tokens_kernel() noexcept;

// #5b: Scatter vision soft tokens at arbitrary sequence positions into sequence buffer
// h0: [total_tokens * H] (H = 512)
// soft_tokens: [num_soft_tokens * H]
// token_indices: [num_soft_tokens]
using ScatterIndexedTokensKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*h0*/,
    compute::Buffer<float> /*soft_tokens*/,
    compute::Buffer<uint32_t> /*token_indices*/,
    uint32_t /*num_soft_tokens*/,
    uint32_t /*H*/)>;

[[nodiscard]] ScatterIndexedTokensKernel make_scatter_indexed_tokens_kernel() noexcept;

// #6: Generate attention mask (1.0 = valid, 0.0 = pad) from position_ids
using MaskFromPosKernel = compute::Kernel1D<void(
    compute::Buffer<int32_t> /*pos_ids*/,
    compute::Buffer<float> /*out_mask*/,
    uint32_t /*total_patches*/)>;

[[nodiscard]] MaskFromPosKernel make_mask_from_pos_kernel() noexcept;

}// namespace luisa::embed_mm
