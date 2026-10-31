#pragma once
// embedding_mm engine -- Rotary Position Embedding (RoPE) kernel.
//
// RoPE applies rotary embeddings to Q and K tensors:
//   For channel c in [0, head_dim / 2):
//     theta = pos * (rope_theta)^(-2*c / head_dim)
//     q'[c]             = q[c] * cos(theta) - q[c + head_dim/2] * sin(theta)
//     q'[c + head_dim/2] = q[c + head_dim/2] * cos(theta) + q[c] * sin(theta)
//
// Gemma 2 uses:
// - sliding_attention layers: rope_theta = 10,000.0f
// - full_attention layers:    rope_theta = 1,000,000.0f

#include <luisa/dsl/syntax.h>

namespace luisa::embed_mm {

inline constexpr uint32_t kRoPEBlock = 256u;

// in_out: [B*S, num_heads * head_dim]
// Scalars: S, num_heads, head_dim, rope_theta
using RoPEKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*input*/,
    compute::Buffer<float> /*output*/,
    uint32_t /*S*/,
    uint32_t /*num_heads*/,
    uint32_t /*head_dim*/,
    float /*rope_theta*/)>;

[[nodiscard]] RoPEKernel make_rope_kernel() noexcept;

}// namespace luisa::embed_mm
