#pragma once
// OCR engine -- rotary position embedding operators (Phase 2).
// Device-free DSL factories; FP32 activations.
//
// SEPARATE explicit variants -- one is NEVER substituted for another:
//   rope_1d        : adjacent-pair 1D RoPE, pairs (2d, 2d+1), one position
//                    axis. This is the GPT-NeoX-"interleaved"/synthetic
//                    convention; it is NOT the Qwen2.5 HF text convention.
//   rope_rotate_half : rotate-half 1D RoPE, pairs (d, d+head_dim/2), one
//                    position axis. This is the Qwen2.5-VL HF text-decoder
//                    convention (HF rotate_half). The dense decoder selects
//                    it via DenseDecoderConfig::rope_pair_layout.
//   rope_2d        : 2D axial RoPE (vision), pairs (2d,2d+1), axis per pair
//                    selects height/width position (axes=2).
//   rope_mrope     : M-RoPE (Qwen2-VL style), pairs (2d,2d+1), axis per pair
//                    selects temporal/height/width position (axes=3).
//   rope_interleaved_mrope : interleaved MRoPE (Qwen3-VL / FireRed style),
//                    pairs (d, d+head_dim/2), axis per pair (axes=3).
//   rope_xd        : XD-RoPE (Hunyuan style), pairs (2d,2d+1), axis per pair
//                    over text/height/width/time (axes=4).
// The 2D/M-RoPE/XD variants share the axial kernel but are distinct named
// factories + variant keys; the axis count and axis_of_pair mapping are
// caller data, never hardcoded to one model. rope_rotate_half shares the
// rotate-half pair math with rope_interleaved_mrope but is a distinct named
// 1-axis factory so a text decoder never silently picks a multimodal variant.
//
// Position buffers: positions[t*axes + axis] (int32). axis_of_pair[d] (uint32)
// maps pair d to its axis. inv_freq[d] = theta^(-2d/head_dim).
//
// Also: sinusoidal absolute position table generator and learned/absolute
// position addition (the latter lives in ops_vision as add_position).
#include <cstdint>

#include <luisa/dsl/syntax.h>

#include "ocr/runtime/dispatch.h"

namespace luisa::ocr {

// Axial RoPE over [tokens, heads, head_dim]. `interleaved` is a compile-time
// host bool: false = pairs (2d,2d+1), true = pairs (d, d+head_dim/2).
// Kernel args: x, out, positions(int32), axis_of_pair(uint32),
//              tokens, heads, head_dim, axes, theta.
using RopeAxialKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*x*/,
    compute::Buffer<float> /*out*/,
    compute::Buffer<int32_t> /*positions [tokens*axes]*/,
    compute::Buffer<uint32_t> /*axis_of_pair [head_dim/2]*/,
    uint32_t /*tokens*/, uint32_t /*heads*/, uint32_t /*head_dim*/,
    uint32_t /*axes*/, float /*theta*/)>;

[[nodiscard]] RopeAxialKernel make_rope_1d_kernel() noexcept;
[[nodiscard]] RopeAxialKernel make_rope_rotate_half_kernel() noexcept;
[[nodiscard]] RopeAxialKernel make_rope_2d_kernel() noexcept;
[[nodiscard]] RopeAxialKernel make_rope_mrope_kernel() noexcept;
[[nodiscard]] RopeAxialKernel make_rope_interleaved_mrope_kernel() noexcept;
[[nodiscard]] RopeAxialKernel make_rope_xd_kernel() noexcept;

// Sinusoidal absolute position table [len, dim] (dim even). out[p,d].
using SinusoidalTableKernel = compute::Kernel1D<void(
    compute::Buffer<float> /*out*/,
    uint32_t /*len*/, uint32_t /*dim*/, float /*theta*/)>;
[[nodiscard]] SinusoidalTableKernel make_sinusoidal_table_kernel() noexcept;

// Dispatch: one thread per (token, head) row.
[[nodiscard]] Result<LaunchPlan1D> plan_rope_dispatch(uint64_t tokens, uint64_t heads);
[[nodiscard]] Result<LaunchPlan1D> plan_sinusoidal_dispatch(uint64_t len);

[[nodiscard]] luisa::string rope_variant_key(luisa::string_view variant);

}// namespace luisa::ocr