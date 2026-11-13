#pragma once
// OCR engine -- normalization / reduction operators (Phase 2).
// Device-free DSL factories; FP32 activations and accumulation.
//
// All row norms operate on [rows, H] row-major FP32. One thread block per row
// is NOT assumed: the naive baseline uses one thread per row and loops H in
// thread (correct for arbitrary H, no shared memory, no warp requirement).
// QK-Norm is RMSNorm applied per head over head_dim (rows = tokens*heads).
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// LayerNorm: y = (x-mean)/sqrt(var+eps)*gamma + beta. gamma/beta are [H];
// bind a 1-element dummy view for the parameterless variant (has_weight=0).
using LayerNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    compute::Buffer<float> /*gamma*/,
    compute::Buffer<float> /*beta*/,
    uint32_t /*rows*/, uint32_t /*H*/, float /*eps*/, uint32_t /*has_weight*/)>;
[[nodiscard]] LayerNormKernel make_layer_norm_kernel() noexcept;

// RMSNorm: y = x/sqrt(mean(x^2)+eps)*gamma. gamma [H]; dummy for parameterless.
using RmsNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    compute::Buffer<float> /*gamma*/,
    uint32_t /*rows*/, uint32_t /*H*/, float /*eps*/, uint32_t /*has_weight*/)>;
[[nodiscard]] RmsNormKernel make_rms_norm_kernel() noexcept;

// Fused residual + RMSNorm. residual_out = projection + residual and
// norm_out = rmsnorm(residual_out) * gamma. residual_out may alias residual;
// projection and norm_out must be disjoint from both outputs.
using ResidualRmsNormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*projection*/, compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*gamma*/, compute::Buffer<float> /*residual_out*/,
    compute::Buffer<float> /*norm_out*/, uint32_t /*rows*/, uint32_t /*H*/, float /*eps*/)>;
[[nodiscard]] ResidualRmsNormKernel make_residual_rms_norm_kernel() noexcept;

// L2 normalize rows: y = x / max(||x||_2, floor).
using L2NormKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*rows*/, uint32_t /*H*/, float /*floor*/)>;
[[nodiscard]] L2NormKernel make_l2_norm_kernel() noexcept;

// Row sum reduction (FP32 accumulation): out[row] = sum_h x[row,h].
using RowSumKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*rows*/, uint32_t /*H*/)>;
[[nodiscard]] RowSumKernel make_row_sum_kernel() noexcept;

// Residual add: out[i] = a[i] + b[i] over count elements (aliasing a==out legal).
using ResidualAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*a*/,
    compute::Buffer<float> /*b*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*count*/)>;
[[nodiscard]] ResidualAddKernel make_residual_add_kernel() noexcept;

// Dispatch helpers (one thread per row / per element; TOTAL threads).
[[nodiscard]] Result<LaunchPlan1D> plan_norm_dispatch(uint64_t rows);
[[nodiscard]] Result<LaunchPlan1D> plan_residual_add_dispatch(uint64_t count);

[[nodiscard]] luisa::string norm_variant_key(luisa::string_view op);

}// namespace luisa::ocr