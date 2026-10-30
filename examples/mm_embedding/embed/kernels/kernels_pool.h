#pragma once
// mm_embedding engine -- Mask-aware Mean Pooling, MRL truncation, and L2 normalization.
//
// Gemma 2 embedding output pipeline:
// 1. Mean Pooling:
//    pooled[b, d] = (sum_{s=0..S-1} mask[b, s] * hidden[b, s, d]) / max(sum(mask[b]), 1)
// 2. Projection (via GEMM):
//    projected[b, D_proj] = pooled[b, H] @ W_proj[H, D_proj]   (512 -> 768)
// 3. MRL Truncation + L2 Normalization:
//    out[b, 0..target_dim) = projected[b, 0..target_dim) / max(||projected[b, 0..target_dim)||_2, 1e-12)

#include <luisa/dsl/syntax.h>

namespace luisa::mm_embed {

inline constexpr uint32_t kPoolBlock = 256u;

// Mean pooling: pooled[b*H + d] = sum(mask * hidden) / count
// hidden: [B*S, H]
// mask: [B*S] (1.0 or 0.0)
// pooled: [B, H]
using MeanPoolKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*hidden*/,
    compute::Buffer<float> /*mask*/,
    compute::Buffer<float> /*pooled*/,
    uint32_t /*S*/,
    uint32_t /*H*/)>;

[[nodiscard]] MeanPoolKernel make_mean_pool_kernel() noexcept;

// MRL Truncation & L2 Normalize:
// One block per batch row, reduces sum of squares across target_dim, writes normalized vector.
// in_proj: [B, D_proj] (e.g. D_proj = 768)
// out_embed: [B, target_dim] (e.g. target_dim = 768, 512, 256, 128)
using MRLNormalizeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*in_proj*/,
    compute::Buffer<float> /*out_embed*/,
    uint32_t /*D_proj*/,
    uint32_t /*target_dim*/)>;

[[nodiscard]] MRLNormalizeKernel make_mrl_normalize_kernel() noexcept;

}// namespace luisa::mm_embed
