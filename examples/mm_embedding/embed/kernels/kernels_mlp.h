#pragma once
// mm_embedding engine -- MLP activation and residual add kernels.
//
// GeGLU activation in Gemma 2:
//   gelu_tanh(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   out[i] = gelu_tanh(gate[i]) * up[i]
//
// Residual add:
//   out[i] = x[i] + residual[i]

#include <luisa/dsl/syntax.h>

namespace luisa::mm_embed {

inline constexpr uint32_t kMLPBlock = 256u;

// geglu: out[i] = gelu_tanh(gate[i]) * up[i]
// Buffers: gate [N], up [N], out [N]. Scalar: N (total elements).
using GeGLUKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*gate*/,
    compute::Buffer<float> /*up*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*N*/)>;

[[nodiscard]] GeGLUKernel make_geglu_kernel() noexcept;

// residual_add: out[i] = x[i] + residual[i]
// Buffers: x [N], residual [N], out [N]. Scalar: N.
using ResidualAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*N*/)>;

[[nodiscard]] ResidualAddKernel make_residual_add_kernel() noexcept;

// ple_gate: out[i] = gelu_tanh(gate[i]) * ple_all[(token * 24 + layer_idx) * H + d]
// Buffers: gate [tokens*H], ple_all [tokens*24*H], out [tokens*H].
using PLEGateKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*gate*/,
    compute::Buffer<float> /*ple_all*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*layer_idx*/,
    uint32_t /*total_tokens*/,
    uint32_t /*H*/)>;

[[nodiscard]] PLEGateKernel make_ple_gate_kernel() noexcept;

// scale_add: out[i] = (x[i] + residual[i]) * scale
// Buffers: x [N], residual [N], out [N]. Scalars: scale, N.
using ScaleAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*out*/,
    float /*scale*/,
    uint32_t /*N*/)>;

[[nodiscard]] ScaleAddKernel make_scale_add_kernel() noexcept;

}// namespace luisa::mm_embed
