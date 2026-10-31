#pragma once
// embedding_mm engine -- Normalization operators: RMSNorm, RMSNormNoWeight, and L2Normalize.
//
// RMSNorm:
//   y[row] = (x[row] / sqrt(mean(x[row]^2) + eps)) * gamma
// RMSNormNoWeight (used for V-norm and modality connectors):
//   y[row] = x[row] / sqrt(mean(x[row]^2) + eps)
// L2Normalize:
//   y[row] = x[row] / max(||x[row]||_2, 1e-12)

#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kNormBlock = 256u;

using RMSNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    compute::Buffer<float> /*gamma*/,
    uint32_t /*H*/,
    float /*eps*/)>;

[[nodiscard]] RMSNormKernel make_rmsnorm_kernel() noexcept;

using RMSNormNoWeightKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*H*/,
    float /*eps*/)>;

[[nodiscard]] RMSNormNoWeightKernel make_rmsnorm_no_weight_kernel() noexcept;

using L2NormalizeKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*D*/)>;

[[nodiscard]] L2NormalizeKernel make_l2_normalize_kernel() noexcept;

}// namespace luisa::embed_mm
