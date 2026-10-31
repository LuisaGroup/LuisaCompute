#pragma once
// embedding_mm engine -- Vision 2D Spatial Pooling, Sequence Mean Pooling, and MRL Normalization operators.
//
// 1. VisionPoolKernel: 3x3 average pooling across active patch grid + sqrt(768) scaling -> soft tokens
// 2. MeanPoolKernel: Mask-aware sequence mean pooling [B, S, H] -> [B, H]
// 3. MRLNormalizeKernel: Matryoshka Representation Learning truncation [B, D_proj] -> [B, target_dim] + L2 norm

#include <cstdint>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/shader.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kPoolBlock = 128u;

using VisionPoolKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*in_hidden*/,
    compute::Buffer<float> /*out_pooled*/,
    uint32_t /*grid_w_soft*/,
    uint32_t /*grid_h_soft*/,
    uint32_t /*patch_stride_w*/,
    uint32_t /*H*/,
    float /*scale_factor*/)>;

using MeanPoolKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*hidden*/,
    compute::Buffer<float> /*mask*/,
    compute::Buffer<float> /*pooled*/,
    uint32_t /*S*/,
    uint32_t /*H*/)>;

using MRLNormalizeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*in_proj*/,
    compute::Buffer<float> /*out_embed*/,
    uint32_t /*D_proj*/,
    uint32_t /*target_dim*/)>;

[[nodiscard]] VisionPoolKernel make_vision_pool_kernel() noexcept;
[[nodiscard]] MeanPoolKernel make_mean_pool_kernel() noexcept;
[[nodiscard]] MRLNormalizeKernel make_mrl_normalize_kernel() noexcept;

}// namespace luisa::embed_mm
