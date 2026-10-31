#pragma once
// embedding_mm engine -- normalization kernels: RMSNorm and L2Normalize.
//
// RMSNorm replaces LayerNorm in Gemma 2:
//   y[row] = (x[row] / sqrt(mean(x[row]^2) + eps)) * (1.0 + gamma)
// One block per row, block 256, fp32 reduction tree in shared memory.
// eps is 1e-6 for EmbeddingGemma-2.
//
// L2Normalize replaces F.normalize:
//   y[row] = x[row] / max(||x[row]||_2, 1e-12)

#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kNormBlock = 256u;

// rmsnorm: y[row] = (x[row] / sqrt(mean(x[row]^2) + eps)) * gamma
// Buffers: x [rows*H], y [rows*H], gamma [H].
// Scalars: H (hidden size), eps.
using RMSNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    compute::Buffer<float> /*gamma*/,
    uint32_t /*H*/,
    float /*eps*/)>;

[[nodiscard]] RMSNormKernel make_rmsnorm_kernel() noexcept;

// rmsnorm_no_weight: y[row] = x[row] / sqrt(mean(x[row]^2) + eps)
// Buffers: x [rows*H], y [rows*H].
// Scalars: H (hidden size), eps.
using RMSNormNoWeightKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*H*/,
    float /*eps*/)>;

[[nodiscard]] RMSNormNoWeightKernel make_rmsnorm_no_weight_kernel() noexcept;

// l2_normalize: y[row] = x[row] / max(||x[row]||_2, 1e-12).
// Buffers: x [rows*D], y [rows*D]. Scalar: D (row length).
using L2NormalizeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*D*/)>;

[[nodiscard]] L2NormalizeKernel make_l2_normalize_kernel() noexcept;

}// namespace luisa::embed_mm
