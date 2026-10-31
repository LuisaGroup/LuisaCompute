#pragma once
// embedding_mm engine -- Positional Encoding operators:
//   1. 1D RoPE (Language Model sliding & full attention layers)
//   2. 2D Axial RoPE (Vision Tower 2D patch attention)
//   3. Conformer Relative Positional Encoding table generator (Audio Tower)

#include <cstdint>
#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kRoPEBlock = 256u;

// 1D RoPE for Language Model Q & K tensors: [B*S, num_heads * head_dim]
using RoPEKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*input*/,
    compute::Buffer<float> /*output*/,
    uint32_t /*S*/,
    uint32_t /*num_heads*/,
    uint32_t /*head_dim*/,
    float /*rope_theta*/)>;

[[nodiscard]] RoPEKernel make_rope_kernel() noexcept;

// 2D Axial RoPE for Vision Attention: qk [B * max_patches, num_heads * head_dim], pos_ids [B * max_patches * 2]
using AxialRoPEKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*qk_in*/,
    compute::Buffer<float> /*qk_out*/,
    compute::Buffer<int32_t> /*pos_ids*/,
    uint32_t /*num_heads*/,
    uint32_t /*head_dim*/,
    float /*rope_theta*/)>;

[[nodiscard]] AxialRoPEKernel make_axial_rope_kernel() noexcept;

// Relative Sinusoidal Positional Encoding Table Generator [13, 1024] for Audio Conformer
using AudioGenRelPosKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*out_pos_embed*/,
    uint32_t /*pos_len*/,
    uint32_t /*hidden_size*/)>;

[[nodiscard]] AudioGenRelPosKernel make_audio_gen_rel_pos_kernel() noexcept;

}// namespace luisa::embed_mm
