#pragma once
// embedding_mm engine -- Pointwise MLP activations, gating, clamping, and residual operators.
//
// 1. GeGLUKernel: out[i] = gelu_tanh(gate[i]) * up[i]
// 2. PLEGateKernel: out[i] = gelu_tanh(gate[i]) * ple_all[(t * 24 + l) * H + d]
// 3. ResidualAddKernel: out[i] = x[i] + residual[i]
// 4. ScaleAddKernel: out[i] = (x[i] + residual[i]) * scale
// 5. AudioClampKernel: y[i] = clamp(x[i], min_v, max_v)
// 6. AudioSiluKernel: y[i] = x[i] * sigmoid(x[i])
// 7. AudioGluKernel: y[t, c] = x[t, c] * sigmoid(x[t, H + c])
// 8. AudioAddScaleKernel: y[i] = x[i] * scale + residual[i]
// 9. AudioAddKernel: y[i] = x[i] + residual[i]

#include <cstdint>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/shader.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kMLPBlock = 256u;

using GeGLUKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*gate*/,
    compute::Buffer<float> /*up*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*N*/)>;

using ResidualAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*N*/)>;

using PLEGateKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*gate*/,
    compute::Buffer<float> /*ple_all*/,
    compute::Buffer<float> /*out*/,
    uint32_t /*layer_idx*/,
    uint32_t /*total_tokens*/,
    uint32_t /*H*/)>;

using ScaleAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*out*/,
    float /*scale*/,
    uint32_t /*N*/)>;

using AudioClampKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    float /*min_v*/,
    float /*max_v*/,
    uint32_t /*total_elements*/)>;

using AudioSiluKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*total_elements*/)>;

using AudioGluKernel = compute::Kernel2D<void(
    compute::Buffer<float> /*in_x*/,
    compute::Buffer<float> /*out_y*/,
    uint32_t /*T*/,
    uint32_t /*H*/)>;

using AudioAddScaleKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*y*/,
    float /*scale*/,
    uint32_t /*total_elements*/)>;

using AudioAddKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*residual*/,
    compute::Buffer<float> /*y*/,
    uint32_t /*total_elements*/)>;

[[nodiscard]] GeGLUKernel make_geglu_kernel() noexcept;
[[nodiscard]] ResidualAddKernel make_residual_add_kernel() noexcept;
[[nodiscard]] PLEGateKernel make_ple_gate_kernel() noexcept;
[[nodiscard]] ScaleAddKernel make_scale_add_kernel() noexcept;
[[nodiscard]] AudioClampKernel make_audio_clamp_kernel() noexcept;
[[nodiscard]] AudioSiluKernel make_audio_silu_kernel() noexcept;
[[nodiscard]] AudioGluKernel make_audio_glu_kernel() noexcept;
[[nodiscard]] AudioAddScaleKernel make_audio_add_scale_kernel() noexcept;
[[nodiscard]] AudioAddKernel make_audio_add_kernel() noexcept;

}// namespace luisa::embed_mm
