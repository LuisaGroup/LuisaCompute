#pragma once
// embedding_mm engine -- Audio Tower kernels for Gemma4AudioModel (~300M).
//
// Operations:
// 1. Subsampling Conv2D Layer 0: Conv2D(1 -> 128, 3x3, stride 2, pad 1) + LayerNorm(128) + ReLU
// 2. Subsampling Conv2D Layer 1: Conv2D(128 -> 32, 3x3, stride 2, pad 1) + LayerNorm(32) + ReLU
//    Flattened directly to [T_2, 1024] where T_2 = 77
// 3. Clamping kernel for Gemma4ClippableLinear bounds
// 4. GEMM with 1D bias vector: C = A @ B + bias (for audio_tower.output_proj)
// 5. Audio Scale Q & K: Q *= q_scale * softplus(per_dim_scale), K *= k_scale
// 6. Conformer Relative Positional Encoding table generator
// 7. Conformer Chunked Attention: 12-token chunks, 24-token context, relative positional bias, softcap 50.0
// 8. LightConv1D GLU: y = x1 * sigmoid(x2)
// 9. LightConv1D Causal Depthwise Conv1D: kernel 5, groups 1024, left pad 4
// 10. SiLU activation: y = x * sigmoid(x)
// 11. Add & Scale Residual: y = x * scale + residual
// 12. Add Residual: y = x + residual
// 13. Scatter audio soft tokens into language model sequence buffer

#include <cstdint>
#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

// #1: Subsample Conv2D Layer 0
// in_features: [H_in, W_in] where H_in = 307, W_in = 128
// weight: [128, 1, 3, 3] = 1152 floats
// norm_weight: [128]
// out_h0: [128, 154, 64]
using AudioSubsampleConv0Kernel = compute::Kernel3D<void(
    compute::Buffer<float> /*in_features*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*norm_weight*/,
    compute::Buffer<float> /*out_h0*/,
    uint32_t /*H_in*/,
    uint32_t /*W_in*/,
    uint32_t /*H_out*/,
    uint32_t /*W_out*/,
    float /*eps*/)>;

[[nodiscard]] AudioSubsampleConv0Kernel make_audio_subsample_conv0_kernel() noexcept;

// #2: Subsample Conv2D Layer 1
// in_h0: [128, 154, 64]
// weight: [32, 128, 3, 3] = 36864 floats
// norm_weight: [32]
// out_flat: [H_out, W_out * 32] = [77, 1024]
using AudioSubsampleConv1Kernel = compute::Kernel3D<void(
    compute::Buffer<float> /*in_h0*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*norm_weight*/,
    compute::Buffer<float> /*out_flat*/,
    uint32_t /*H_in*/,
    uint32_t /*W_in*/,
    uint32_t /*H_out*/,
    uint32_t /*W_out*/,
    float /*eps*/)>;

[[nodiscard]] AudioSubsampleConv1Kernel make_audio_subsample_conv1_kernel() noexcept;

// #3: In-place or out-of-place Clamping
using AudioClampKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    float /*min_v*/,
    float /*max_v*/,
    uint32_t /*total_elements*/)>;

[[nodiscard]] AudioClampKernel make_audio_clamp_kernel() noexcept;

// #4: GEMM with Bias: C = A @ B + bias
// A: [M, K], B: [K, N] (row-major), bias: [N], C: [M, N]
using GemmBiasKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*A*/,
    compute::Buffer<float> /*B*/,
    compute::Buffer<float> /*bias*/,
    compute::Buffer<float> /*C*/,
    uint32_t /*M*/,
    uint32_t /*N*/,
    uint32_t /*K*/)>;

[[nodiscard]] GemmBiasKernel make_gemm_bias_kernel() noexcept;

// #5: Audio Scale Q & K
// q: [T, 1024] (8 heads * 128 head_dim)
// k: [T, 1024]
// per_dim_scale: [128]
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

// #6: Relative Positional Encoding Table Generator [13, 1024]
using AudioGenRelPosKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*out_pos_embed*/,
    uint32_t /*pos_len*/,   // 13
    uint32_t /*hidden_size*/// 1024
)>;

[[nodiscard]] AudioGenRelPosKernel make_audio_gen_rel_pos_kernel() noexcept;

// #7: Conformer Chunked Attention
// q: [T, num_heads * head_dim]
// k: [T, num_heads * head_dim]
// v: [T, num_heads * head_dim]
// rel_k: [13, num_heads * head_dim]
// out_attn: [T, num_heads * head_dim]
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

// #8: LightConv1D GLU: y = x1 * sigmoid(x2)
// in_x: [T, 2048], out_y: [T, 1024]
using AudioGluKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*in_x*/,
    compute::Buffer<float> /*out_y*/,
    uint32_t /*T*/,
    uint32_t /*H*/)>;

[[nodiscard]] AudioGluKernel make_audio_glu_kernel() noexcept;

// #9: LightConv1D Causal Depthwise Conv1D (kernel 5, left pad 4)
// in_x: [T, 1024]
// weight: [1024, 5]
// out_y: [T, 1024]
using AudioDepthwiseConv1DKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*in_x*/,
    compute::Buffer<float> /*weight*/,
    compute::Buffer<float> /*out_y*/,
    uint32_t /*T*/,
    uint32_t /*channels*/)>;

[[nodiscard]] AudioDepthwiseConv1DKernel make_audio_depthwise_conv1d_kernel() noexcept;

// #10: SiLU activation: y = x * sigmoid(x)
using AudioSiluKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*total_elements*/)>;

[[nodiscard]] AudioSiluKernel make_audio_silu_kernel() noexcept;

// #11: Add & Scale Residual: y = x * scale + residual
using AudioAddScaleKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*y*/,
    float /*scale*/,
    uint32_t /*total_elements*/)>;

[[nodiscard]] AudioAddScaleKernel make_audio_add_scale_kernel() noexcept;

// #12: Add Residual: y = x + residual
using AudioAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*total_elements*/)>;

[[nodiscard]] AudioAddKernel make_audio_add_kernel() noexcept;

// #13: Scatter audio soft tokens into language model sequence buffer
using AudioScatterTokensKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*h0*/,
    compute::Buffer<float> /*soft_tokens*/,
    uint32_t /*start_token_idx*/,
    uint32_t /*num_soft_tokens*/,
    uint32_t /*H*/)>;

[[nodiscard]] AudioScatterTokensKernel make_audio_scatter_tokens_kernel() noexcept;

}// namespace luisa::embed_mm
