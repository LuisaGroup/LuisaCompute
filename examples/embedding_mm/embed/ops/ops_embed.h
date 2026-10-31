#pragma once
// embedding_mm engine -- Token Embedding, Patch Position Embedding, and Multimodal Scatter operators.
//
// 1. EmbedLookupKernel: Word embedding table lookup + sqrt(512) scaling
// 2. ScalePixelsKernel: Rescale pixels from [0, 1] to [-1, 1]
// 3. AddPatchPosEmbedKernel: Add 2D interpolated position embeddings & zero padded patches
// 4. MaskFromPosKernel: Generate float attention mask (1.0 valid, 0.0 padded) from position_ids
// 5. ScatterSoftTokensKernel: Contiguous soft token insertion into language model sequence
// 6. ScatterIndexedTokensKernel: Indexed non-contiguous soft token insertion (video/interleaved)
// 7. AudioScatterTokensKernel: Contiguous audio soft token insertion

#include <cstdint>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/shader.h>

namespace luisa::embed_mm {

using EmbedLookupKernel = compute::Kernel2D<void(
    compute::Buffer<uint32_t> /*input_ids*/,
    compute::Buffer<float> /*embed_table*/,
    compute::Buffer<float> /*out_hidden*/,
    uint32_t /*H*/,
    float /*scale*/)>;

using ScalePixelsKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*in_pixels*/,
    compute::Buffer<float> /*out_pixels*/,
    uint32_t /*total_elements*/)>;

using AddPatchPosEmbedKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*hidden*/,
    compute::Buffer<float> /*pos_table*/,
    compute::Buffer<int32_t> /*pos_ids*/,
    uint32_t /*H*/,
    uint32_t /*pos_table_size*/)>;

using MaskFromPosKernel = compute::Kernel1D<void(
    compute::Buffer<int32_t> /*pos_ids*/,
    compute::Buffer<float> /*out_mask*/,
    uint32_t /*total_patches*/)>;

using ScatterSoftTokensKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*h0*/,
    compute::Buffer<float> /*soft_tokens*/,
    uint32_t /*start_token_idx*/,
    uint32_t /*num_soft_tokens*/,
    uint32_t /*H*/)>;

using ScatterIndexedTokensKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*h0*/,
    compute::Buffer<float> /*soft_tokens*/,
    compute::Buffer<uint32_t> /*token_indices*/,
    uint32_t /*num_soft_tokens*/,
    uint32_t /*H*/)>;

using AudioScatterTokensKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*h0*/,
    compute::Buffer<float> /*soft_tokens*/,
    uint32_t /*start_token_idx*/,
    uint32_t /*num_soft_tokens*/,
    uint32_t /*H*/)>;

[[nodiscard]] EmbedLookupKernel make_embed_lookup_kernel() noexcept;
[[nodiscard]] ScalePixelsKernel make_scale_pixels_kernel() noexcept;
[[nodiscard]] AddPatchPosEmbedKernel make_add_patch_pos_embed_kernel() noexcept;
[[nodiscard]] MaskFromPosKernel make_mask_from_pos_kernel() noexcept;
[[nodiscard]] ScatterSoftTokensKernel make_scatter_soft_tokens_kernel() noexcept;
[[nodiscard]] ScatterIndexedTokensKernel make_scatter_indexed_tokens_kernel() noexcept;
[[nodiscard]] AudioScatterTokensKernel make_audio_scatter_tokens_kernel() noexcept;

}// namespace luisa::embed_mm
